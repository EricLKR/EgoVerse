#!/usr/bin/env python
"""Non-blocking single-frame deployment for EgoVerse joint Clean V2.

This is an isolated entrypoint.  It does not modify the audited synchronous
single-frame or History V2 deployments.  Camera sampling, model inference and
JSONL writing run outside the 30 Hz command loop.  Every prediction is anchored
to the absolute successful-action tick at which its observation was captured;
expired prefix actions are skipped when the result arrives.

The model contract remains single-frame RGB + proprio.  No FK or IK is used.
Execute mode retains the initial continuity gate, calibrated joint limits,
per-command driver clamp and independent binary gripper state machine.
"""

from __future__ import annotations

import argparse
import copy
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

import deploy_egoverse_hpt_joint as legacy
import deploy_egoverse_hpt_joint_history_v2_async as async_runtime


logger = logging.getLogger("deploy_egoverse_hpt_joint_async")
CONTROL_PERIOD_S = 1.0 / 30.0
SINGLE_FRAME_CHECKPOINT_PROFILES = {
    name: legacy.CHECKPOINT_PROFILES[name]
    for name in (
        "piper-clean-v2",
        "mixed-clean-v2",
        "mixed-clean-v2-trimmed-15k",
        "residual-v1-step12k",
    )
}


def make_cpu_single_frame_batch(observation: dict[str, Any]) -> dict[str, Any]:
    """Copy one observation into an independently owned CPU snapshot."""
    state = np.asarray(
        [observation[f"joint_{index}.pos"] for index in range(7)],
        dtype=np.float32,
    )
    front = np.ascontiguousarray(observation["435if"], dtype=np.uint8).copy()
    wrist = np.ascontiguousarray(observation["405"], dtype=np.uint8).copy()
    if state.shape != (7,) or not np.isfinite(state).all():
        raise RuntimeError(f"malformed single-frame state: {state}")
    if front.shape != (480, 640, 3) or wrist.shape != (480, 640, 3):
        raise RuntimeError(
            "malformed single-frame RGB input: "
            f"front={front.shape}, wrist={wrist.shape}, expected=(480,640,3)"
        )
    return {
        "observation.state": state,
        "observation.images.435if": front,
        "observation.images.405": wrist,
    }


class LatestObservationRobot:
    """Delegate robot I/O while sampling cameras/proprio at 30 Hz in a thread."""

    def __init__(self, robot, wrist_color_normalizer=None):
        self._robot = robot
        self._wrist_color_normalizer = wrist_color_normalizer
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._latest: dict[str, Any] | None = None
        self._sequence = 0
        self._last_returned_sequence = -1

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def connect(self, *args, **kwargs):
        self._stop.clear()
        self._thread = None
        self._worker_error = None
        self._latest = None
        self._sequence = 0
        self._last_returned_sequence = -1
        return self._robot.connect(*args, **kwargs)

    def _start_sampler(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="single-frame-camera-sampler",
            daemon=True,
        )
        self._thread.start()

    def calibrate_wrist_color(self) -> dict[str, Any] | None:
        if self._thread is not None:
            raise RuntimeError("wrist color calibration must precede camera sampling")
        if self._wrist_color_normalizer is None:
            return None
        observation = None
        for _ in range(self._wrist_color_normalizer.calibration_frames):
            observation = self._robot.get_observation()
            self._wrist_color_normalizer.observe(observation["405"])
        assert observation is not None
        report = self._wrist_color_normalizer.finalize()
        corrected = self._wrist_color_normalizer.apply(observation["405"])
        report["last_raw_rgb_mean"] = (
            np.asarray(observation["405"]).mean(axis=(0, 1)).tolist()
        )
        report["last_corrected_rgb_mean"] = corrected.mean(axis=(0, 1)).tolist()
        return report

    def _copy_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        copied = dict(observation)
        copied["435if"] = np.asarray(observation["435if"], dtype=np.uint8).copy()
        wrist = np.asarray(observation["405"], dtype=np.uint8)
        if self._wrist_color_normalizer is not None:
            wrist = self._wrist_color_normalizer.apply(wrist)
        copied["405"] = np.ascontiguousarray(wrist, dtype=np.uint8).copy()
        copied["_single_frame_timestamp_ns"] = time.monotonic_ns()
        return copied

    def _sample_loop(self) -> None:
        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                observation = self._copy_observation(self._robot.get_observation())
                with self._condition:
                    self._latest = observation
                    self._sequence += 1
                    self._condition.notify_all()
                self._stop.wait(max(0.0, CONTROL_PERIOD_S - (time.perf_counter() - started)))
        except BaseException as exc:
            with self._condition:
                self._worker_error = exc
                self._condition.notify_all()

    def get_observation(self) -> dict[str, Any]:
        self._start_sampler()
        deadline = time.monotonic() + 10.0
        while True:
            with self._condition:
                if self._worker_error is not None:
                    raise RuntimeError("single-frame camera sampler failed") from self._worker_error
                if self._latest is not None and self._sequence > self._last_returned_sequence:
                    self._last_returned_sequence = self._sequence
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("single-frame camera sampler produced no frame for 10 seconds")
                self._condition.wait(timeout=min(remaining, 0.1))

    def sampler_status(self) -> dict[str, Any]:
        with self._condition:
            return {
                "sequence": int(self._sequence),
                "worker_alive": bool(self._thread and self._thread.is_alive()),
                "fatal_error": repr(self._worker_error) if self._worker_error else None,
            }

    def disconnect(self, *args, **kwargs):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
            if thread.is_alive():
                raise RuntimeError("single-frame camera sampler did not stop")
        return self._robot.disconnect(*args, **kwargs)


@dataclass(frozen=True)
class SingleFrameInferenceRequest:
    request_id: int
    request_tick: int
    observation_timestamp_ns: int
    submitted_ns: int
    batch: dict[str, Any]
    snapshot_pack_ms: float


class BackgroundSingleFramePredictor:
    """Single-flight model worker; the command thread never waits for CUDA."""

    _STOP = object()

    def __init__(self, policy):
        self.policy = policy
        self._requests: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._results: queue.Queue[async_runtime.InferenceResult] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._running_request_id: int | None = None
        self._queued_request_id: int | None = None
        self._fatal_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="single-frame-inference-worker",
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
                predict_started = time.perf_counter()
                with torch.inference_mode():
                    actions = (
                        self.policy.predict_action_chunk(request.batch)
                        .detach()
                        .cpu()
                        .numpy()
                        .copy()
                    )
                policy_total_ms = (time.perf_counter() - predict_started) * 1e3
                report = copy.deepcopy(self.policy.last_prediction_report)
                model_ms = float((report or {}).get("model_inference_ms", 0.0))
                result = async_runtime.InferenceResult(
                    request_id=request.request_id,
                    request_tick=request.request_tick,
                    submitted_ns=request.observation_timestamp_ns,
                    finished_ns=time.monotonic_ns(),
                    actions=actions,
                    prediction_report=report,
                    timing={
                        "queue_wait_ms": (started_ns - request.submitted_ns) / 1e6,
                        "cpu_snapshot_pack_ms": float(request.snapshot_pack_ms),
                        "policy_total_ms": policy_total_ms,
                        "model_inference_ms": model_ms,
                        "preprocess_postprocess_ms": max(0.0, policy_total_ms - model_ms),
                    },
                    error=None,
                )
            except BaseException as exc:
                result = async_runtime.InferenceResult(
                    request_id=request.request_id,
                    request_tick=request.request_tick,
                    submitted_ns=request.observation_timestamp_ns,
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
                raise RuntimeError("single-frame inference worker initialization failed") from self._fatal_error
            return (
                self._queued_request_id is None
                and self._running_request_id is None
                and self._results.empty()
            )

    def submit(self, request: SingleFrameInferenceRequest) -> bool:
        with self._lock:
            if self._fatal_error is not None:
                raise RuntimeError("single-frame inference worker initialization failed") from self._fatal_error
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

    def poll(self) -> async_runtime.InferenceResult | None:
        with self._lock:
            if self._fatal_error is not None:
                raise RuntimeError("single-frame inference worker initialization failed") from self._fatal_error
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
                "fatal_error": repr(self._fatal_error) if self._fatal_error else None,
            }

    def close(self) -> None:
        if self._thread.is_alive():
            self._requests.put(self._STOP)
            self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("single-frame inference worker did not stop")


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
    log: async_runtime.AsyncJsonlLogger,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> None:
    execute = args.mode == "execute"
    if execute and not args.allow_motion:
        raise RuntimeError("execute mode requires --allow-motion")

    robot = LatestObservationRobot(
        legacy.make_robot(args, joint_limits, gripper_stroke_mm),
        legacy.make_wrist_color_normalizer(args),
    )
    robot.connect(calibrate=False)
    predictor: BackgroundSingleFramePredictor | None = None
    gripper = async_runtime.AlignedGripperStateMachine(
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
    controller = async_runtime.AbsoluteTickActionController(
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
    deferred_result: async_runtime.InferenceResult | None = None
    next_request_id = 1
    requested_plan_id: int | None = None
    try:
        camera_reports = legacy.configure_camera_auto_controls(
            robot,
            front_exposure_mode=args.front_exposure_mode,
            front_exposure=args.front_exposure,
            front_gain=args.front_gain,
        )
        for camera_name, camera_report in camera_reports.items():
            log.write(
                {
                    "event": "camera_config",
                    "camera": camera_name,
                    **camera_report,
                    "command_sent": False,
                }
            )

        color_report = robot.calibrate_wrist_color()
        if color_report is not None:
            log.write(
                {
                    "event": "wrist_color_normalization",
                    **color_report,
                    "command_sent": False,
                }
            )

        warmup_observation = robot.get_observation()
        warmup_started = time.perf_counter()
        warmup_batch = make_cpu_single_frame_batch(warmup_observation)
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
                "observation_timestamp_ns": warmup_observation.get(
                    "_single_frame_timestamp_ns"
                ),
                "command_sent": False,
            }
        )
        predictor = BackgroundSingleFramePredictor(policy)

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
            observation_wait_ms = (time.perf_counter() - observation_started) * 1e3
            observation_timestamp_ns = int(
                observation.get("_single_frame_timestamp_ns", time.monotonic_ns())
            )
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
                requested_plan_id = None
                if (
                    result_event["accepted"]
                    and args.gripper_prediction_source == "full-horizon"
                ):
                    gripper_horizon_update = gripper.observe_prediction_horizon_at(
                        result.request_id,
                        result.request_tick,
                        legacy.full_gripper_horizon_from_report(
                            result.prediction_report
                        ),
                    )

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
                batch = make_cpu_single_frame_batch(observation)
                snapshot_ms = (time.perf_counter() - snapshot_started) * 1e3
                request = SingleFrameInferenceRequest(
                    request_id=next_request_id,
                    request_tick=controller.action_tick,
                    observation_timestamp_ns=observation_timestamp_ns,
                    submitted_ns=time.monotonic_ns(),
                    batch=batch,
                    snapshot_pack_ms=snapshot_ms,
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
                        "observation_capture_age_ms": (
                            time.monotonic_ns() - observation_timestamp_ns
                        )
                        / 1e6,
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
                        "single_frame_snapshot": snapshot_ms,
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
                    "camera_sampler": robot.sampler_status(),
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
    parser.add_argument("--async-max-result-age-ms", type=float, default=250.0)
    parser.add_argument("--async-log-queue-size", type=int, default=512)
    async_args, remaining = parser.parse_known_args()
    old_profiles = legacy.CHECKPOINT_PROFILES
    old_argv = sys.argv
    try:
        legacy.CHECKPOINT_PROFILES = SINGLE_FRAME_CHECKPOINT_PROFILES
        sys.argv = [old_argv[0], *remaining]
        args = legacy.parse_args()
    finally:
        legacy.CHECKPOINT_PROFILES = old_profiles
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
    if args.save_inference_images:
        raise ValueError(
            "the async entrypoint does not synchronously save inference images"
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
    policy = legacy.make_policy(args, joint_limits, gripper_stroke_mm)
    if args.mode == "checkpoint":
        logger.info(
            "Async single-frame checkpoint loaded without constructing a robot: %s",
            policy.checkpoint_report,
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = async_runtime.AsyncJsonlLogger(
        args.log_dir / f"single_frame_async_{args.mode}_{timestamp}.jsonl",
        max_pending=args.async_log_queue_size,
    )
    try:
        log.write(
            {
                "event": "start",
                "mode": args.mode,
                "checkpoint_profile": args.checkpoint_profile or "custom",
                "checkpoint_description": args.checkpoint_description,
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": actual_sha256,
                "checkpoint_contract": policy.checkpoint_report,
                "observation_contract": "single_rgb_proprio",
                "query_frequency": args.query_frequency,
                "inference_steps": args.inference_steps,
                "seed": args.seed,
                "async_replan_lead_ticks": args.async_replan_lead_ticks,
                "async_min_suffix_actions": args.async_min_suffix_actions,
                "async_transition_blend_actions": args.async_transition_blend_actions,
                "async_max_result_age_ms": args.async_max_result_age_ms,
                "absolute_time_multi_chunk_ensemble": False,
                "chunk_switching": "latest_time_aligned_forecast_short_boundary_blend",
                "driver_clip_tolerance": args.driver_clip_tolerance,
                "feedback_tracking": args.feedback_tracking,
                "tracking_joint_tolerance": args.tracking_joint_tolerance,
                "tracking_min_progress": args.tracking_min_progress,
                "tracking_stall_frames": args.tracking_stall_frames,
                "tracking_max_hold_frames": args.tracking_max_hold_frames,
                "arm_gripper_plan_alignment": True,
                "control_frequency_hz": 30,
                "absolute_tick_alignment": True,
                "inference_thread": True,
                "camera_sampler_thread": True,
                "jsonl_writer_thread": True,
                "command_sent": False,
            }
        )
        if args.mode == "offline":
            legacy.run_offline(args, policy, log, device)
        else:
            run_live_async(
                args,
                policy,
                log,
                joint_limits,
                gripper_stroke_mm,
            )
    finally:
        log.close()
        logger.info("wrote async single-frame audit log to %s", log.path)


if __name__ == "__main__":
    main()
