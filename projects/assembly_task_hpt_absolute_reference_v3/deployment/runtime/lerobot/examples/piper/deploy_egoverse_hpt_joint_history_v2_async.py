#!/usr/bin/env python
"""Non-blocking four-frame deployment for EgoVerse joint History V2.

This entrypoint intentionally leaves the audited synchronous History V2
deployment unchanged.  Camera sampling, model inference, and JSONL writing run
outside the 30 Hz command loop.  Every predicted chunk is anchored to the
absolute successful-action tick at which its observation was captured; when a
result arrives, expired prefix actions are skipped instead of replayed.

No FK or IK is used.  Execute mode retains the existing initial continuity
gate, calibrated joint limits, per-command driver clamp, and independent
binary gripper state machine.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import deploy_egoverse_hpt_joint as legacy
import deploy_egoverse_hpt_joint_history_v2 as history_v2


logger = logging.getLogger("deploy_egoverse_hpt_joint_history_v2_async")
CONTROL_PERIOD_S = 1.0 / 30.0


def make_cpu_history_batch(
    samples: list[dict[str, Any]],
    selection_report: dict[str, Any],
) -> dict[str, Any]:
    """Build one immutable CPU batch; the policy performs the only H2D copy.

    The synchronous entrypoint first moved all history tensors to CUDA, then
    ``prepare_rgb_history`` moved them back to CPU before a second H2D copy.
    Keeping the snapshot as NumPy/CPU data removes that round trip and keeps
    all CUDA work inside the inference worker.
    """
    if len(samples) != 4:
        raise ValueError(f"History V2 requires four samples, got {len(samples)}")
    states = np.stack(
        [
            np.asarray(
                [sample[f"joint_{index}.pos"] for index in range(7)],
                dtype=np.float32,
            )
            for sample in samples
        ],
        axis=0,
    )
    front = np.stack(
        [np.asarray(sample["435if"], dtype=np.uint8) for sample in samples],
        axis=0,
    )
    wrist = np.stack(
        [np.asarray(sample["405"], dtype=np.uint8) for sample in samples],
        axis=0,
    )
    if states.shape != (4, 7):
        raise RuntimeError(f"malformed history state shape {states.shape}")
    if front.shape != (4, 480, 640, 3) or wrist.shape != (4, 480, 640, 3):
        raise RuntimeError(
            "malformed RGB history: "
            f"front={front.shape}, wrist={wrist.shape}, expected=(4,480,640,3)"
        )
    if not np.isfinite(states).all():
        raise RuntimeError("history state contains NaN or Inf")
    return {
        "observation.state": states[-1],
        "observation.images.435if": front[-1],
        "observation.images.405": wrist[-1],
        "observation.history_v2.state": states,
        "observation.history_v2.images.435if": front,
        "observation.history_v2.images.405": wrist,
        "observation.history_v2.selection_ages_ms": torch.tensor(
            selection_report["selected_ages_ms"], dtype=torch.float32
        ).unsqueeze(0),
    }


class AsyncJsonlLogger:
    """Serialize and flush audit records without blocking the control loop."""

    _STOP = object()

    def __init__(self, path: Path, max_pending: int = 512):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        self.path = path
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_pending)
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="history-v2-jsonl-writer",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            with self.path.open("x") as stream:
                while True:
                    record = self._queue.get()
                    try:
                        if record is self._STOP:
                            return
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        stream.flush()
                    finally:
                        self._queue.task_done()
        except BaseException as exc:  # surfaced in the foreground on next write/close
            self._error = exc

    def write(self, record: dict[str, Any]) -> None:
        if self._error is not None:
            raise RuntimeError("asynchronous JSONL writer failed") from self._error
        try:
            self._queue.put_nowait(record)
        except queue.Full as exc:
            raise RuntimeError(
                "audit log queue filled; refusing to run without complete logging"
            ) from exc

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(self._STOP)
            self._queue.join()
            self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("asynchronous JSONL writer did not stop")
        if self._error is not None:
            raise RuntimeError("asynchronous JSONL writer failed") from self._error


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    request_tick: int
    submitted_ns: int
    samples: list[dict[str, Any]]
    selection_report: dict[str, Any]


@dataclass(frozen=True)
class InferenceResult:
    request_id: int
    request_tick: int
    submitted_ns: int
    finished_ns: int
    actions: np.ndarray | None
    prediction_report: dict[str, Any] | None
    timing: dict[str, float]
    error: BaseException | None


class BackgroundChunkPredictor:
    """Single-flight model worker; the control thread never waits for it."""

    _STOP = object()

    def __init__(self, policy):
        self.policy = policy
        self._requests: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._results: queue.Queue[InferenceResult] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._running_request_id: int | None = None
        self._queued_request_id: int | None = None
        self._fatal_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="history-v2-inference-worker",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        inference_device = torch.device(self.policy.config.device)
        if inference_device.type == "cuda":
            device_index = (
                inference_device.index
                if inference_device.index is not None
                else torch.cuda.current_device()
            )
            try:
                torch.cuda.set_device(device_index)
            except BaseException as exc:
                with self._lock:
                    self._fatal_error = exc
                return
        while True:
            request = self._requests.get()
            try:
                if request is self._STOP:
                    return
                with self._lock:
                    self._queued_request_id = None
                    self._running_request_id = request.request_id
                started_ns = time.monotonic_ns()
                pack_started = time.perf_counter()
                batch = make_cpu_history_batch(
                    request.samples, request.selection_report
                )
                pack_ms = (time.perf_counter() - pack_started) * 1e3
                predict_started = time.perf_counter()
                with torch.inference_mode():
                    actions = (
                        self.policy.predict_action_chunk(batch)
                        .detach()
                        .cpu()
                        .numpy()
                        .copy()
                    )
                policy_total_ms = (time.perf_counter() - predict_started) * 1e3
                report = copy.deepcopy(self.policy.last_prediction_report)
                model_ms = float((report or {}).get("model_inference_ms", 0.0))
                result = InferenceResult(
                    request_id=request.request_id,
                    request_tick=request.request_tick,
                    submitted_ns=request.submitted_ns,
                    finished_ns=time.monotonic_ns(),
                    actions=actions,
                    prediction_report=report,
                    timing={
                        "queue_wait_ms": (started_ns - request.submitted_ns) / 1e6,
                        "cpu_history_pack_ms": pack_ms,
                        "policy_total_ms": policy_total_ms,
                        "model_inference_ms": model_ms,
                        "preprocess_postprocess_ms": max(0.0, policy_total_ms - model_ms),
                    },
                    error=None,
                )
            except BaseException as exc:
                result = InferenceResult(
                    request_id=request.request_id,
                    request_tick=request.request_tick,
                    submitted_ns=request.submitted_ns,
                    finished_ns=time.monotonic_ns(),
                    actions=None,
                    prediction_report=None,
                    timing={},
                    error=exc,
                )
            finally:
                if request is not self._STOP:
                    with self._lock:
                        self._running_request_id = None
                    self._results.put(result)
                self._requests.task_done()

    def can_submit(self) -> bool:
        with self._lock:
            if self._fatal_error is not None:
                raise RuntimeError("inference worker initialization failed") from self._fatal_error
            return (
                self._queued_request_id is None
                and self._running_request_id is None
                and self._results.empty()
            )

    def submit(self, request: InferenceRequest) -> bool:
        with self._lock:
            if self._fatal_error is not None:
                raise RuntimeError("inference worker initialization failed") from self._fatal_error
            if (
                self._queued_request_id is not None
                or self._running_request_id is not None
                or not self._results.empty()
            ):
                return False
            self._queued_request_id = request.request_id
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            with self._lock:
                self._queued_request_id = None
            return False
        return True

    def poll(self) -> InferenceResult | None:
        with self._lock:
            if self._fatal_error is not None:
                raise RuntimeError("inference worker initialization failed") from self._fatal_error
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queued_request_id": self._queued_request_id,
                "running_request_id": self._running_request_id,
                "result_ready": not self._results.empty(),
                "worker_alive": self._thread.is_alive(),
                "fatal_error": (
                    repr(self._fatal_error) if self._fatal_error is not None else None
                ),
            }

    def close(self) -> None:
        if self._thread.is_alive():
            self._requests.put(self._STOP)
            self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("inference worker did not stop")


@dataclass
class ActivePlan:
    plan_id: int
    start_tick: int
    actions: np.ndarray
    prediction_report: dict[str, Any]


class AbsoluteTickActionController:
    """Execute only the newest time-aligned forecast without multi-chunk fusion."""

    def __init__(
        self,
        policy,
        min_suffix_actions: int,
        transition_blend_actions: int = 4,
    ):
        self.policy = policy
        self.min_suffix_actions = int(min_suffix_actions)
        self.transition_blend_actions = int(transition_blend_actions)
        self.action_tick = 0
        self.active_plan: ActivePlan | None = None
        self.last_velocity = np.zeros(6, dtype=np.float64)
        self.pending_velocity: np.ndarray | None = None
        self.last_raw_action: np.ndarray | None = None

    def install(
        self,
        *,
        plan_id: int,
        start_tick: int,
        actions: np.ndarray,
        prediction_report: dict[str, Any],
        result_age_ms: float,
        max_result_age_ms: float,
    ) -> dict[str, Any]:
        array = np.asarray(actions, dtype=np.float32).copy()
        if array.ndim != 2 or array.shape[1] != 7 or not np.isfinite(array).all():
            raise RuntimeError(f"invalid asynchronous action chunk {array.shape}")
        prefix = self.action_tick - int(start_tick)
        remaining = len(array) - prefix
        accepted = True
        reason = None
        if prefix < 0:
            accepted = False
            reason = "result_start_tick_is_in_the_future"
        elif result_age_ms > max_result_age_ms:
            accepted = False
            reason = "result_exceeded_wall_time_age_limit"
        elif remaining < self.min_suffix_actions:
            accepted = False
            reason = "result_has_too_few_time_aligned_suffix_actions"
        switch_reference = None
        switch_joint_rms_before = None
        switch_joint_rms_after = None
        blended_actions = 0
        if accepted and self.active_plan is not None:
            old_offset = self.action_tick - self.active_plan.start_tick
            if 0 <= old_offset < len(self.active_plan.actions):
                old_reference = self.active_plan.actions[old_offset].copy()
                switch_reference = "existing_time_aligned_plan"
            elif self.last_raw_action is not None:
                old_reference = self.last_raw_action.copy()
                switch_reference = "last_raw_action"
            else:
                old_reference = None
            if old_reference is not None and 0 <= prefix < len(array):
                before = array[prefix, :6].astype(np.float64)
                switch_joint_rms_before = float(
                    np.sqrt(np.mean(np.square(before - old_reference[:6])))
                )
                blended_actions = min(
                    self.transition_blend_actions,
                    len(array) - prefix,
                )
                for blend_index in range(blended_actions):
                    alpha = float(blend_index + 1) / float(blended_actions)
                    old_index = old_offset + blend_index
                    if 0 <= old_index < len(self.active_plan.actions):
                        old_action = self.active_plan.actions[old_index]
                    else:
                        old_action = old_reference
                    new_index = prefix + blend_index
                    array[new_index, :6] = (
                        (1.0 - alpha) * old_action[:6]
                        + alpha * array[new_index, :6]
                    )
                after = array[prefix, :6].astype(np.float64)
                switch_joint_rms_after = float(
                    np.sqrt(np.mean(np.square(after - old_reference[:6])))
                )
        if accepted:
            self.active_plan = ActivePlan(
                int(plan_id), int(start_tick), array.copy(), prediction_report
            )
        return {
            "accepted": accepted,
            "reason": reason,
            "plan_id": int(plan_id),
            "request_tick": int(start_tick),
            "install_tick": int(self.action_tick),
            "skipped_prefix_actions": int(max(0, prefix)),
            "remaining_suffix_actions": int(max(0, remaining)),
            "multi_chunk_ensemble": False,
            "transition_blend_actions_requested": int(
                self.transition_blend_actions
            ),
            "transition_blended_actions": int(blended_actions),
            "switch_reference": switch_reference,
            "switch_joint_rms_before": switch_joint_rms_before,
            "switch_joint_rms_after": switch_joint_rms_after,
            "result_age_ms": float(result_age_ms),
        }

    def remaining_actions(self) -> int:
        if self.active_plan is None:
            return 0
        return max(
            0,
            len(self.active_plan.actions)
            - (self.action_tick - self.active_plan.start_tick),
        )

    def should_request(self, lead_ticks: int) -> bool:
        return self.active_plan is None or self.remaining_actions() <= int(lead_ticks)

    def status(self) -> dict[str, Any]:
        plan_offset = (
            self.action_tick - self.active_plan.start_tick
            if self.active_plan is not None
            else None
        )
        return {
            "action_tick": int(self.action_tick),
            "active_plan_id": (
                self.active_plan.plan_id if self.active_plan is not None else None
            ),
            "active_plan_start_tick": (
                self.active_plan.start_tick if self.active_plan is not None else None
            ),
            "source_offset": plan_offset,
            "remaining_actions": int(self.remaining_actions()),
            "multi_chunk_ensemble": False,
        }

    def peek(self, current_driver: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        current = np.asarray(current_driver, dtype=np.float64).reshape(7)
        plan_offset = None
        gap_hold = False
        if self.active_plan is not None:
            plan_offset = self.action_tick - self.active_plan.start_tick
        if (
            self.active_plan is not None
            and plan_offset is not None
            and 0 <= plan_offset < len(self.active_plan.actions)
        ):
            raw_action = self.active_plan.actions[plan_offset].copy()
            self.last_raw_action = raw_action.copy()
        else:
            gap_hold = True
            raw_action = np.empty(7, dtype=np.float32)
            raw_action[:6] = current[:6]
            raw_action[6] = (
                float(self.last_raw_action[6])
                if self.last_raw_action is not None
                else (100.0 if current[6] >= 50.0 else 0.0)
            )
        safe_action, pending_velocity = self.policy._filter_driver_action(
            raw_action,
            current[:6],
            self.last_velocity,
        )
        self.pending_velocity = pending_velocity
        report = self.status()
        report["source_offset"] = plan_offset
        report["gap_hold"] = gap_hold
        return safe_action, report

    def advance(self) -> None:
        if self.pending_velocity is None:
            raise RuntimeError("cannot advance before peeking an action")
        self.last_velocity = self.pending_velocity
        self.pending_velocity = None
        self.action_tick += 1

    def cancel_pending(self) -> None:
        self.pending_velocity = None


class AlignedGripperStateMachine(legacy.IndependentGripperStateMachine):
    """Add forecasts at their observation tick rather than arrival time."""

    def observe_prediction_horizon_at(
        self,
        chunk_id: int,
        start_tick: int,
        model_commands_pct: Any,
    ) -> dict[str, Any]:
        commands = self._validate_prediction_horizon(model_commands_pct)
        chunk = int(chunk_id)
        if any(existing == chunk for existing, _, _ in self._prediction_horizons):
            raise RuntimeError(f"duplicate gripper prediction horizon {chunk}")
        self._prediction_horizons.append(
            (chunk, int(start_tick), commands.copy())
        )
        return {
            "chunk_id": chunk,
            "absolute_start_tick": int(start_tick),
            "install_tick": int(self.prediction_tick),
            "skipped_prefix_actions": int(max(0, self.prediction_tick - start_tick)),
            "horizon_steps": int(commands.size),
            "commands_pct": commands.tolist(),
        }


def history_snapshot(
    observation: dict[str, Any],
    history_buffer: history_v2.CausalObservationHistoryV2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchor_ns = observation.get("_history_v2_timestamp_ns")
    if anchor_ns is None:
        raise RuntimeError("History V2 observation lacks a causal anchor timestamp")
    return history_buffer.select(int(anchor_ns))


def current_driver_state(
    observation: dict[str, Any],
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> np.ndarray:
    return legacy.observation_to_driver_state(
        observation, joint_limits, gripper_stroke_mm
    )


def run_live_async(
    args: argparse.Namespace,
    policy,
    log: AsyncJsonlLogger,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
    history_buffer: history_v2.CausalObservationHistoryV2,
) -> None:
    execute = args.mode == "execute"
    if execute and not args.allow_motion:
        raise RuntimeError("execute mode requires --allow-motion")

    robot = history_v2.make_history_robot(args, joint_limits, gripper_stroke_mm)
    robot.connect(calibrate=False)
    predictor: BackgroundChunkPredictor | None = None
    gripper = AlignedGripperStateMachine(
        mode=args.gripper_control_mode,
        close_confirm_frames=args.gripper_close_confirm_frames,
        open_confirm_frames=args.gripper_open_confirm_frames,
        min_close_hold_frames=args.gripper_min_close_hold_frames,
        min_open_hold_frames=args.gripper_min_open_hold_frames,
        close_feedback_threshold=args.gripper_close_feedback_threshold,
        open_feedback_threshold=args.gripper_open_feedback_threshold,
        settle_frames=args.gripper_settle_frames,
        settle_tolerance=args.gripper_settle_tolerance,
        terminal_open_after_release=args.gripper_terminal_open_after_release,
    )
    controller = AbsoluteTickActionController(
        policy,
        args.async_min_suffix_actions,
        args.async_transition_blend_actions,
    )
    tracking_gate = legacy.ArmTrackingGate(
        tolerance=args.tracking_joint_tolerance,
        min_progress=args.tracking_min_progress,
        stall_frames=args.tracking_stall_frames,
        max_hold_frames=args.tracking_max_hold_frames,
    )
    pending_model_action: dict[str, float] | None = None
    pending_plan_id: int | None = None
    deferred_result: InferenceResult | None = None
    next_request_id = 1
    requested_plan_id: int | None = None
    try:
        camera_report = legacy.configure_front_camera_auto_controls(robot)
        log.write(
            {
                "event": "camera_config",
                "camera": "435if",
                **camera_report,
                "command_sent": False,
            }
        )

        warmup_observation = robot.get_observation()
        warmup_samples, warmup_selection = history_snapshot(
            warmup_observation, history_buffer
        )
        warmup_started = time.perf_counter()
        warmup_batch = make_cpu_history_batch(
            warmup_samples, warmup_selection
        )
        warmup_actions = (
            policy.predict_action_chunk(warmup_batch)
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        warmup_report = copy.deepcopy(policy.last_prediction_report)
        install_report = controller.install(
            plan_id=0,
            start_tick=0,
            actions=warmup_actions,
            prediction_report=warmup_report,
            result_age_ms=0.0,
            max_result_age_ms=args.async_max_result_age_ms,
        )
        if not install_report["accepted"]:
            raise RuntimeError(f"warmup chunk was rejected: {install_report}")
        warmup_driver = current_driver_state(
            warmup_observation, joint_limits, gripper_stroke_mm
        )
        warmup_action, warmup_queue = controller.peek(warmup_driver)
        continuity = legacy.initial_continuity_report(
            warmup_observation,
            torch.from_numpy(warmup_action),
            joint_limits,
            args.max_initial_joint_delta_deg,
        )
        controller.cancel_pending()
        warmup_gripper = None
        if args.gripper_prediction_source == "full-horizon":
            warmup_gripper = gripper.observe_prediction_horizon_at(
                0,
                0,
                legacy.full_gripper_horizon_from_report(warmup_report),
            )
        log.write(
            {
                "event": "warmup",
                "mode": args.mode,
                "total_step_ms": (time.perf_counter() - warmup_started) * 1e3,
                "first_action": warmup_action.tolist(),
                "initial_continuity": continuity,
                "chunk_report": warmup_report,
                "absolute_tick_install": install_report,
                "queue_status": warmup_queue,
                "gripper_horizon_update": warmup_gripper,
                "history_selection": warmup_selection,
                "command_sent": False,
            }
        )
        predictor = BackgroundChunkPredictor(policy)

        if execute:
            held_position = robot.enable_motion_hold_current()
            log.write(
                {
                    "event": "motion_enable",
                    "held_position": held_position,
                    "command_sent": True,
                }
            )

        started_run = time.perf_counter()
        for frame_index in range(args.max_steps):
            loop_started = time.perf_counter()
            observation_started = time.perf_counter()
            observation = robot.get_observation()
            observation_wait_ms = (
                time.perf_counter() - observation_started
            ) * 1e3
            driver_state = current_driver_state(
                observation, joint_limits, gripper_stroke_mm
            )

            result_event = None
            new_chunk_report = None
            gripper_horizon_update = None
            poll_started = time.perf_counter()
            result = predictor.poll()
            result_poll_ms = (time.perf_counter() - poll_started) * 1e3
            if result is None and deferred_result is not None and not tracking_gate.pending:
                result = deferred_result
                deferred_result = None
            if result is not None and tracking_gate.pending:
                if deferred_result is not None:
                    raise RuntimeError(
                        "received a second asynchronous result before the pending "
                        "action boundary was committed"
                    )
                deferred_result = result
                result_event = {
                    "request_id": int(result.request_id),
                    "request_tick": int(result.request_tick),
                    "deferred_until_action_boundary": True,
                }
                result = None
            if result is not None:
                if result.error is not None:
                    raise RuntimeError(
                        f"asynchronous inference request {result.request_id} failed"
                    ) from result.error
                result_age_ms = (time.monotonic_ns() - result.submitted_ns) / 1e6
                result_event = controller.install(
                    plan_id=result.request_id,
                    start_tick=result.request_tick,
                    actions=result.actions,
                    prediction_report=result.prediction_report,
                    result_age_ms=result_age_ms,
                    max_result_age_ms=args.async_max_result_age_ms,
                )
                result_event["inference_timing"] = result.timing
                new_chunk_report = result.prediction_report
                if result_event["accepted"]:
                    requested_plan_id = None
                    if args.gripper_prediction_source == "full-horizon":
                        gripper_horizon_update = gripper.observe_prediction_horizon_at(
                            result.request_id,
                            result.request_tick,
                            legacy.full_gripper_horizon_from_report(
                                result.prediction_report
                            ),
                        )
                else:
                    requested_plan_id = None

            request_event = None
            snapshot_ms = 0.0
            active_plan_id = (
                controller.active_plan.plan_id
                if controller.active_plan is not None
                else None
            )
            if (
                controller.should_request(args.async_replan_lead_ticks)
                and requested_plan_id != active_plan_id
                and deferred_result is None
                and predictor.can_submit()
            ):
                snapshot_started = time.perf_counter()
                samples, selection_report = history_snapshot(
                    observation, history_buffer
                )
                snapshot_ms = (time.perf_counter() - snapshot_started) * 1e3
                request = InferenceRequest(
                    request_id=next_request_id,
                    request_tick=controller.action_tick,
                    submitted_ns=time.monotonic_ns(),
                    samples=samples,
                    selection_report=selection_report,
                )
                if predictor.submit(request):
                    requested_plan_id = active_plan_id
                    request_event = {
                        "request_id": next_request_id,
                        "request_tick": int(controller.action_tick),
                        "active_plan_id": active_plan_id,
                        "active_remaining_actions": int(
                            controller.remaining_actions()
                        ),
                        "history_selection": selection_report,
                    }
                    next_request_id += 1

            filter_started = time.perf_counter()
            if pending_model_action is None:
                model_action_array, queue_status = controller.peek(driver_state)
                model_action = legacy.action_to_robot_dict(
                    torch.from_numpy(model_action_array)
                )
                pending_plan_id = queue_status["active_plan_id"]
            else:
                queue_status = controller.status()
                queue_status["gap_hold"] = False
                model_action = dict(pending_model_action)
            filter_ms = (time.perf_counter() - filter_started) * 1e3
            if args.gripper_prediction_source == "full-horizon":
                gripper_state = gripper.update_from_prediction_horizons(
                    model_action["joint_6.pos"],
                    driver_state[6],
                    active_chunk_id=pending_plan_id,
                    active_tick=controller.action_tick,
                )
            else:
                gripper_state = gripper.update(
                    model_action["joint_6.pos"], driver_state[6]
                )
            robot_action = dict(model_action)
            robot_action["joint_6.pos"] = gripper_state["command_pct"]

            tracking_status = (
                tracking_gate.observe(driver_state[:6])
                if args.feedback_tracking
                else legacy.disabled_arm_tracking_status()
            )
            tracking_abort = tracking_status["abort_reason"]

            send_started = time.perf_counter()
            sent_action = None
            if execute and tracking_abort is None:
                sent_action = robot.send_action(
                    robot_action,
                    present_position=legacy.driver_present_position(driver_state),
                )
            command_send_ms = (time.perf_counter() - send_started) * 1e3
            clip_report = legacy.command_clip_report(
                robot_action, sent_action, args.driver_clip_tolerance
            )
            arm_clipped = clip_report["arm_clipped"]
            gripper_clipped = clip_report["gripper_clipped"]

            action_advanced = False
            if not execute:
                controller.advance()
                gripper.advance_prediction_tick()
                action_advanced = True
                pending_model_action = None
                pending_plan_id = None
            elif not args.feedback_tracking:
                if gripper_state["target_reached"]:
                    controller.advance()
                    gripper.advance_prediction_tick()
                    pending_model_action = None
                    pending_plan_id = None
                    action_advanced = True
                else:
                    pending_model_action = dict(model_action)
            elif not tracking_gate.pending:
                tracking_gate.begin(
                    [robot_action[f"joint_{index}.pos"] for index in range(6)]
                )
                pending_model_action = dict(model_action)
            elif (
                tracking_status["target_reached"]
                and gripper_state["target_reached"]
            ):
                controller.advance()
                gripper.advance_prediction_tick()
                tracking_gate.clear()
                pending_model_action = None
                pending_plan_id = None
                action_advanced = True

            control_step_ms = (time.perf_counter() - loop_started) * 1e3
            log.write(
                {
                    "mode": args.mode,
                    "frame": frame_index,
                    "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": time.perf_counter() - started_run,
                    "control_step_ms": control_step_ms,
                    "stage_timing_ms": {
                        "observation_wait": observation_wait_ms,
                        "result_poll": result_poll_ms,
                        "history_snapshot": snapshot_ms,
                        "command_filter": filter_ms,
                        "command_send": command_send_ms,
                    },
                    "raw_state": [
                        float(observation[f"joint_{index}.pos"])
                        for index in range(7)
                    ],
                    "driver_state": driver_state.tolist(),
                    "model_action": model_action,
                    "proposed_action": robot_action,
                    "sent_action": sent_action,
                    "arm_command_was_clipped": arm_clipped,
                    "gripper_command_was_clipped": gripper_clipped,
                    "driver_clip_report": clip_report,
                    "arm_tracking": tracking_status,
                    "action_advanced": action_advanced,
                    "queue_status": queue_status,
                    "gripper_state_machine": gripper_state,
                    "gripper_horizon_update": gripper_horizon_update,
                    "async_request": request_event,
                    "async_result": result_event,
                    "async_worker": predictor.status(),
                    "new_chunk_report": new_chunk_report,
                    "command_sent": bool(execute and sent_action is not None),
                }
            )
            if tracking_abort is not None:
                raise RuntimeError(
                    "feedback-aware asynchronous execution stopped: "
                    f"{tracking_abort}; tracking={tracking_status}; "
                    f"clip={clip_report}"
                )
            legacy.precise_sleep(
                CONTROL_PERIOD_S - (time.perf_counter() - loop_started)
            )
    finally:
        if predictor is not None:
            predictor.close()
        robot.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--async-replan-lead-ticks", type=int, default=4)
    parser.add_argument("--async-min-suffix-actions", type=int, default=5)
    parser.add_argument("--async-transition-blend-actions", type=int, default=4)
    parser.add_argument("--async-max-result-age-ms", type=float, default=1000.0)
    parser.add_argument("--async-log-queue-size", type=int, default=512)
    async_args, remaining = parser.parse_known_args()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = history_v2.parse_args()
    finally:
        sys.argv = old_argv
    for key, value in vars(async_args).items():
        setattr(args, key, value)
    if args.list_checkpoint_profiles:
        return args
    if args.temporal_ensemble:
        raise ValueError(
            "the async entrypoint already performs absolute-time replanning; "
            "do not combine it with --temporal-ensemble"
        )
    if not 1 <= args.async_replan_lead_ticks < args.query_frequency:
        raise ValueError(
            "--async-replan-lead-ticks must be in [1, query_frequency-1]"
        )
    if not 1 <= args.async_min_suffix_actions <= args.query_frequency:
        raise ValueError(
            "--async-min-suffix-actions must be in [1, query_frequency]"
        )
    if not 1 <= args.async_transition_blend_actions <= args.query_frequency:
        raise ValueError(
            "--async-transition-blend-actions must be in [1, query_frequency]"
        )
    if args.async_max_result_age_ms <= 0:
        raise ValueError("--async-max-result-age-ms must be positive")
    if args.async_log_queue_size < 16:
        raise ValueError("--async-log-queue-size must be at least 16")
    return args


def main() -> None:
    global logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    if args.list_checkpoint_profiles:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    actual_sha256 = legacy.sha256_file(args.checkpoint)
    if actual_sha256 != args.checkpoint_sha256:
        raise RuntimeError(
            "checkpoint SHA-256 mismatch: "
            f"expected {args.checkpoint_sha256}, got {actual_sha256}"
        )
    joint_limits, gripper_stroke_mm = legacy.load_calibration(
        args.calibration_json
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    policy = history_v2.make_policy(args, joint_limits, gripper_stroke_mm)
    if args.mode == "checkpoint":
        logger.info(
            "Async History V2 checkpoint loaded without constructing a robot: %s",
            policy.checkpoint_report,
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = AsyncJsonlLogger(
        args.log_dir / f"history_v2_async_{args.mode}_{timestamp}.jsonl",
        max_pending=args.async_log_queue_size,
    )
    try:
        log.write(
            {
                "event": "start",
                "mode": args.mode,
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": actual_sha256,
                "checkpoint_contract": policy.checkpoint_report,
                "query_frequency": args.query_frequency,
                "observation_history": args.observation_history,
                "observation_history_step_ms": args.observation_history_step_ms,
                "observation_history_max_skew_ms": (
                    args.observation_history_max_skew_ms
                ),
                "async_replan_lead_ticks": args.async_replan_lead_ticks,
                "async_min_suffix_actions": args.async_min_suffix_actions,
                "async_transition_blend_actions": args.async_transition_blend_actions,
                "async_max_result_age_ms": args.async_max_result_age_ms,
                "control_frequency_hz": 30,
                "absolute_tick_alignment": True,
                "absolute_time_multi_chunk_ensemble": False,
                "chunk_switching": "latest_time_aligned_forecast_short_boundary_blend",
                "driver_clip_tolerance": args.driver_clip_tolerance,
                "feedback_tracking": args.feedback_tracking,
                "tracking_joint_tolerance": args.tracking_joint_tolerance,
                "tracking_min_progress": args.tracking_min_progress,
                "tracking_stall_frames": args.tracking_stall_frames,
                "tracking_max_hold_frames": args.tracking_max_hold_frames,
                "arm_gripper_plan_alignment": True,
                "inference_thread": True,
                "camera_sampler_thread": True,
                "jsonl_writer_thread": True,
                "command_sent": False,
            }
        )
        if args.mode == "offline":
            history_v2.run_offline(args, policy, log, device)
        else:
            history_buffer = history_v2.CausalObservationHistoryV2(
                args.observation_history,
                args.observation_history_step_ms,
                args.observation_history_max_skew_ms,
            )
            history_v2._ACTIVE_HISTORY = history_buffer
            run_live_async(
                args,
                policy,
                log,
                joint_limits,
                gripper_stroke_mm,
                history_buffer,
            )
    finally:
        log.close()
        logger.info("wrote Async History V2 audit log to %s", log.path)


if __name__ == "__main__":
    main()
