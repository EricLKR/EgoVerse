#!/usr/bin/env python
"""Decoded-joint RTC deployment for single-frame Mixed Clean V2.

This additive entrypoint deliberately leaves the audited synchronous and
asynchronous HPT deployment files unchanged.  Mixed Clean V2 has a
deterministic 30-step residual-joint decoder rather than an iterative Flow
Matching sampler, so generative latent guidance is not available.  RTC is
therefore applied after decoding:

* inference runs in the existing background worker;
* every forecast is anchored to the successful action tick at observation
  capture time;
* actions that expired during inference are discarded;
* overlapping old/new forecasts are blended over a short cosine ramp in
  decoded physical 6-D joint-radian space;
* the gripper remains binary and is handled by the existing absolute-time,
  feedback-aware state machine;
* policy time advances only when the underlying guarded runtime commits an
  action tick.

``checkpoint`` and ``offline`` never construct a robot. ``live-dry-run`` is
observation-only. ``execute`` still requires ``--allow-motion``.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# This dedicated HPT entrypoint must never inherit a pi0.5 training checkout
# from a long-lived terminal.  Both trees expose the same top-level
# ``egomimic`` package, but only the HPT tree contains hpt_mixed_clean_v2.
_DEFAULT_HPT_ROOT = Path(__file__).resolve().parents[3] / "egoverse_piper" / "EgoVerse"
_HPT_TRAINING_ROOT = Path(
    os.environ.get("EGOVERSE_HPT_TRAINING_ROOT", str(_DEFAULT_HPT_ROOT))
).resolve()
_HPT_TARGET = _HPT_TRAINING_ROOT / "egomimic" / "algo" / "hpt_mixed_clean_v2.py"
if not _HPT_TARGET.is_file():
    raise RuntimeError(
        "HPT Mixed training checkout is incomplete: expected "
        f"{_HPT_TARGET}. Set EGOVERSE_HPT_TRAINING_ROOT to the correct "
        "EgoVerse checkout."
    )
sys.path[:] = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != _HPT_TRAINING_ROOT
]
sys.path.insert(0, str(_HPT_TRAINING_ROOT))
os.environ["EGOVERSE_TRAINING_ROOT"] = str(_HPT_TRAINING_ROOT)

import numpy as np
import torch

import deploy_egoverse_hpt_joint as legacy
import deploy_egoverse_hpt_joint_async as single_async
import deploy_egoverse_hpt_joint_history_v2_async as async_runtime


logger = logging.getLogger("deploy_egoverse_hpt_joint_rtc")
MODEL_HORIZON = 30


def _physical_chunk_from_report(
    prediction_report: dict[str, Any], expected_steps: int
) -> np.ndarray:
    if prediction_report is None:
        raise RuntimeError("decoded-joint RTC requires a prediction report")
    physical = np.asarray(
        prediction_report.get("selected_physical_actions"), dtype=np.float32
    )
    expected = (int(expected_steps), 7)
    if physical.shape != expected or not np.isfinite(physical).all():
        raise RuntimeError(
            "decoded-joint RTC requires aligned selected physical actions: "
            f"expected={expected}, got={physical.shape}"
        )
    return physical.copy()


class HPTDecodedJointRTCController(async_runtime.AbsoluteTickActionController):
    """Absolute-time controller with a decoded physical-joint RTC overlap."""

    def __init__(
        self,
        policy,
        min_suffix_actions: int,
        transition_blend_actions: int = 8,
    ) -> None:
        super().__init__(policy, min_suffix_actions, transition_blend_actions)
        self.active_physical: np.ndarray | None = None

    @staticmethod
    def _cosine_take_new(blend_index: int, count: int) -> float:
        # Do not reach exactly one inside the overlap: the first pure-new
        # sample after the ramp then remains a small, bounded final step.
        phase = float(blend_index + 1) / float(count + 1)
        return 0.5 - 0.5 * math.cos(math.pi * phase)

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
            raise RuntimeError(f"invalid RTC action chunk {array.shape}")
        physical = _physical_chunk_from_report(prediction_report, len(array))
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

        reference = None
        old_offset = None
        blend_count = 0
        before_rad = None
        after_rad = None
        before_driver = None
        after_driver = None
        blend_weights: list[float] = []
        if accepted and self.active_plan is not None:
            old_offset = self.action_tick - self.active_plan.start_tick
            if (
                self.active_physical is not None
                and 0 <= old_offset < len(self.active_plan.actions)
                and 0 <= prefix < len(array)
            ):
                reference = "existing_absolute_time_joint_plan"
                old_driver_reference = self.active_plan.actions[old_offset, :6].copy()
                old_physical_reference = self.active_physical[old_offset, :6].copy()
                before_rad = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                physical[prefix, :6].astype(np.float64)
                                - old_physical_reference.astype(np.float64)
                            )
                        )
                    )
                )
                before_driver = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                array[prefix, :6].astype(np.float64)
                                - old_driver_reference.astype(np.float64)
                            )
                        )
                    )
                )
                blend_count = min(
                    self.transition_blend_actions,
                    len(array) - prefix,
                    len(self.active_plan.actions) - old_offset,
                )
                for blend_index in range(blend_count):
                    alpha = self._cosine_take_new(blend_index, blend_count)
                    blend_weights.append(alpha)
                    old_index = old_offset + blend_index
                    new_index = prefix + blend_index
                    blended_joint = (
                        (1.0 - alpha) * self.active_physical[old_index, :6]
                        + alpha * physical[new_index, :6]
                    )
                    physical[new_index, :6] = blended_joint
                    array[new_index, :6] = self.policy._joint_rad_to_driver(
                        blended_joint
                    )
                after_rad = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                physical[prefix, :6].astype(np.float64)
                                - old_physical_reference.astype(np.float64)
                            )
                        )
                    )
                )
                after_driver = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                array[prefix, :6].astype(np.float64)
                                - old_driver_reference.astype(np.float64)
                            )
                        )
                    )
                )

        if accepted:
            self.active_plan = async_runtime.ActivePlan(
                int(plan_id), int(start_tick), array.copy(), prediction_report
            )
            self.active_physical = physical.copy()
        return {
            "accepted": bool(accepted),
            "reason": reason,
            "plan_id": int(plan_id),
            "request_tick": int(start_tick),
            "install_tick": int(self.action_tick),
            "inference_delay_action_ticks": int(max(0, prefix)),
            "skipped_prefix_actions": int(max(0, prefix)),
            "remaining_suffix_actions": int(max(0, remaining)),
            "rtc": True,
            "rtc_space": "decoded_physical_6d_joint_radians",
            "rtc_gripper_guidance": False,
            "rtc_overlap_reference": reference,
            "rtc_old_source_offset": old_offset,
            "rtc_blend_actions_requested": int(self.transition_blend_actions),
            "rtc_blended_actions": int(blend_count),
            "rtc_cosine_take_new_weights": blend_weights,
            "switch_joint_rms_before_rad": before_rad,
            "switch_joint_rms_after_rad": after_rad,
            "switch_joint_rms_before_deg": (
                None if before_rad is None else math.degrees(before_rad)
            ),
            "switch_joint_rms_after_deg": (
                None if after_rad is None else math.degrees(after_rad)
            ),
            "switch_driver_rms_before": before_driver,
            "switch_driver_rms_after": after_driver,
            "result_age_ms": float(result_age_ms),
        }

    def should_request(self, replan_interval_ticks: int) -> bool:
        if self.active_plan is None:
            return True
        interval = int(replan_interval_ticks)
        if interval <= 0:
            raise ValueError("RTC replan interval must be positive")
        elapsed = self.action_tick - self.active_plan.start_tick
        return elapsed >= interval or self.remaining_actions() <= self.min_suffix_actions

    def status(self) -> dict[str, Any]:
        report = super().status()
        report.update(
            {
                "rtc": True,
                "rtc_space": "decoded_physical_6d_joint_radians",
                "rtc_blend_actions": int(self.transition_blend_actions),
            }
        )
        return report


class HPTAlignedGripperCompatibility(
    async_runtime.AlignedGripperStateMachine
):
    """Keep the legacy HPT gripper contract while reusing the async loop.

    The isolated async entrypoint was later taught to pass ``settle_*``
    keywords used by another deployment family, but the audited HPT binary
    state machine never implemented those arguments.  Accepting and recording
    them here fixes that interface drift without altering its open/close
    decisions or editing either legacy file.
    """

    def __init__(
        self,
        *,
        settle_frames: int = 0,
        settle_tolerance: float = 0.0,
        contact_detection: bool = True,
        contact_max_feedback: float = 70.0,
        contact_stall_frames: int = 15,
        contact_motion_tolerance: float = 0.5,
        contact_min_close_frames: int = 10,
        contact_min_drop: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rtc_ignored_settle_frames = int(settle_frames)
        self.rtc_ignored_settle_tolerance = float(settle_tolerance)
        self.contact_detection = bool(contact_detection)
        self.contact_max_feedback = float(contact_max_feedback)
        self.contact_stall_frames = int(contact_stall_frames)
        self.contact_motion_tolerance = float(contact_motion_tolerance)
        self.contact_min_close_frames = int(contact_min_close_frames)
        self.contact_min_drop = float(contact_min_drop)
        self._contact_feedback_window: deque[float] = deque(
            maxlen=self.contact_stall_frames
        )
        self._close_start_feedback: float | None = None
        self._contact_latched = False

    def _reset_contact_tracking(self) -> None:
        self._contact_feedback_window.clear()
        self._close_start_feedback = None
        self._contact_latched = False

    def update(
        self, model_command_pct: float, feedback_pct: float
    ) -> dict[str, Any]:
        report = super().update(model_command_pct, feedback_pct)
        feedback = float(report["feedback_pct"])
        if report["state"] != "close":
            self._reset_contact_tracking()
        else:
            if self._close_start_feedback is None:
                self._close_start_feedback = feedback
            else:
                self._close_start_feedback = max(
                    self._close_start_feedback, feedback
                )
            self._contact_feedback_window.append(feedback)
            window_full = (
                len(self._contact_feedback_window) == self.contact_stall_frames
            )
            window_span = (
                float(
                    max(self._contact_feedback_window)
                    - min(self._contact_feedback_window)
                )
                if self._contact_feedback_window
                else math.inf
            )
            close_drop = float(self._close_start_feedback - feedback)
            contact_candidate = bool(
                self.contact_detection
                and not report["close_reached"]
                and report["state_age_frames"] >= self.contact_min_close_frames
                and feedback <= self.contact_max_feedback
                and close_drop >= self.contact_min_drop
                and window_full
                and window_span <= self.contact_motion_tolerance
            )
            if contact_candidate:
                self._contact_latched = True
            if self._contact_latched:
                report["target_reached"] = True
        report.update(
            {
                "contact_detection_enabled": self.contact_detection,
                "contact_reached": bool(self._contact_latched),
                "contact_completion_source": (
                    "physical_close_threshold"
                    if report["close_reached"]
                    else "stalled_on_object"
                    if self._contact_latched
                    else None
                ),
                "contact_max_feedback": self.contact_max_feedback,
                "contact_stall_frames": self.contact_stall_frames,
                "contact_motion_tolerance": self.contact_motion_tolerance,
                "contact_min_close_frames": self.contact_min_close_frames,
                "contact_min_drop": self.contact_min_drop,
                "contact_close_start_feedback": self._close_start_feedback,
                "contact_feedback_window_size": len(
                    self._contact_feedback_window
                ),
                "contact_feedback_window_span": (
                    float(
                        max(self._contact_feedback_window)
                        - min(self._contact_feedback_window)
                    )
                    if self._contact_feedback_window
                    else None
                ),
            }
        )
        return report

    def update_from_prediction_horizons(
        self,
        fallback_model_command_pct: float,
        feedback_pct: float,
        *,
        active_chunk_id: int | None = None,
        active_tick: int | None = None,
    ) -> dict[str, Any]:
        report = super().update_from_prediction_horizons(
            fallback_model_command_pct,
            feedback_pct,
        )
        report.update(
            {
                "rtc_active_arm_chunk_id": active_chunk_id,
                "rtc_active_arm_tick": active_tick,
            }
        )
        return report


class HPTArmTrackingGateCompatibility:
    """Small feedback gate expected by the isolated async runtime."""

    def __init__(
        self,
        *,
        tolerance: float,
        min_progress: float,
        stall_frames: int,
        max_hold_frames: int,
    ) -> None:
        self.tolerance = float(tolerance)
        self.min_progress = float(min_progress)
        self.stall_frames = int(stall_frames)
        self.max_hold_frames = int(max_hold_frames)
        self.pending = False
        self.target: np.ndarray | None = None
        self.frames = 0
        self.stalled_frames = 0
        self.best_rms = math.inf

    def begin(self, target: Any) -> None:
        values = np.asarray(target, dtype=np.float64).reshape(6)
        if not np.isfinite(values).all():
            raise RuntimeError("arm tracking target contains NaN or Inf")
        self.pending = True
        self.target = values.copy()
        self.frames = 0
        self.stalled_frames = 0
        self.best_rms = math.inf

    def clear(self) -> None:
        self.pending = False
        self.target = None
        self.frames = 0
        self.stalled_frames = 0
        self.best_rms = math.inf

    def observe(self, current: Any) -> dict[str, Any]:
        if not self.pending or self.target is None:
            return {
                "enabled": True,
                "pending": False,
                "target_reached": False,
                "max_error": None,
                "rms_error": None,
                "best_rms": None,
                "frames": 0,
                "stalled_frames": 0,
                "abort_reason": None,
            }
        observed = np.asarray(current, dtype=np.float64).reshape(6)
        error = np.abs(self.target - observed)
        max_error = float(error.max())
        rms_error = float(np.sqrt(np.mean(np.square(error))))
        reached = max_error <= self.tolerance
        self.frames += 1
        if rms_error < self.best_rms - self.min_progress:
            self.best_rms = rms_error
            self.stalled_frames = 0
        else:
            self.stalled_frames += 1
        abort_reason = None
        if not reached and self.stalled_frames >= self.stall_frames:
            abort_reason = "arm tracking error made no sufficient progress"
        elif not reached and self.frames >= self.max_hold_frames:
            abort_reason = "arm target was not reached before the hold limit"
        return {
            "enabled": True,
            "pending": True,
            "target_reached": bool(reached),
            "max_error": max_error,
            "rms_error": rms_error,
            "best_rms": float(self.best_rms),
            "frames": int(self.frames),
            "stalled_frames": int(self.stalled_frames),
            "abort_reason": abort_reason,
        }


def _disabled_arm_tracking_status() -> dict[str, Any]:
    return {
        "enabled": False,
        "pending": False,
        "target_reached": True,
        "max_error": None,
        "rms_error": None,
        "best_rms": None,
        "frames": 0,
        "stalled_frames": 0,
        "abort_reason": None,
    }


def _driver_present_position(driver_state: Any) -> dict[str, float]:
    values = np.asarray(driver_state, dtype=np.float64).reshape(7)
    return {f"joint_{index}.pos": float(value) for index, value in enumerate(values)}


def _command_clip_report(
    proposed: dict[str, float],
    sent: dict[str, float] | None,
    tolerance: float,
) -> dict[str, Any]:
    if sent is None:
        return {
            "arm_clipped": False,
            "gripper_clipped": False,
            "max_abs_difference": 0.0,
            "differences": {},
            "command_sent": False,
        }
    differences = {
        key: float(sent.get(key, proposed[key]) - proposed[key])
        for key in proposed
    }
    arm_clipped = any(
        abs(differences.get(f"joint_{index}.pos", 0.0)) > tolerance
        for index in range(6)
    )
    gripper_clipped = (
        abs(differences.get("joint_6.pos", 0.0)) > tolerance
    )
    return {
        "arm_clipped": bool(arm_clipped),
        "gripper_clipped": bool(gripper_clipped),
        "max_abs_difference": float(
            max((abs(value) for value in differences.values()), default=0.0)
        ),
        "differences": differences,
        "command_sent": True,
    }


def run_live_rtc(
    args: argparse.Namespace,
    policy,
    log: async_runtime.AsyncJsonlLogger,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> None:
    """Reuse the guarded async runtime with only its controller substituted."""
    original_controller = async_runtime.AbsoluteTickActionController
    original_gripper = async_runtime.AlignedGripperStateMachine
    missing = object()
    compatibility = {
        "ArmTrackingGate": HPTArmTrackingGateCompatibility,
        "disabled_arm_tracking_status": _disabled_arm_tracking_status,
        "driver_present_position": _driver_present_position,
        "command_clip_report": _command_clip_report,
    }
    previous = {
        name: getattr(legacy, name, missing) for name in compatibility
    }
    def make_gripper(**kwargs: Any) -> HPTAlignedGripperCompatibility:
        return HPTAlignedGripperCompatibility(
            contact_detection=args.gripper_contact_detection,
            contact_max_feedback=args.gripper_contact_max_feedback,
            contact_stall_frames=args.gripper_contact_stall_frames,
            contact_motion_tolerance=args.gripper_contact_motion_tolerance,
            contact_min_close_frames=args.gripper_contact_min_close_frames,
            contact_min_drop=args.gripper_contact_min_drop,
            **kwargs,
        )

    async_runtime.AbsoluteTickActionController = HPTDecodedJointRTCController
    async_runtime.AlignedGripperStateMachine = make_gripper
    for name, value in compatibility.items():
        setattr(legacy, name, value)
    try:
        single_async.run_live_async(
            args,
            policy,
            log,
            joint_limits,
            gripper_stroke_mm,
        )
    finally:
        async_runtime.AbsoluteTickActionController = original_controller
        async_runtime.AlignedGripperStateMachine = original_gripper
        for name, value in previous.items():
            if value is missing:
                delattr(legacy, name)
            else:
                setattr(legacy, name, value)


def run_offline_rtc_pair(
    args: argparse.Namespace,
    policy,
    log: async_runtime.AsyncJsonlLogger,
    device: torch.device,
) -> None:
    """Measure one time-aligned HPT chunk replacement without robot I/O."""
    if args.offline_npz is None:
        raise ValueError("offline RTC mode requires --offline-npz")
    keys = (
        "observation.state",
        "observation.images.435if",
        "observation.images.405",
    )
    with np.load(args.offline_npz, allow_pickle=False) as archive:
        missing = [key for key in keys if key not in archive]
        if missing:
            raise KeyError(f"offline NPZ is missing {missing}")
        arrays = {key: archive[key] for key in keys}
    request_tick = int(args.rtc_offline_request_tick)
    install_tick = request_tick + int(args.rtc_offline_inference_delay_ticks)
    count = min(len(arrays[key]) for key in keys)
    if count <= request_tick:
        raise ValueError(
            f"offline NPZ needs frame {request_tick}, but contains {count} frames"
        )

    first_batch = legacy.make_policy_batch(
        arrays[keys[0]][0], arrays[keys[1]][0], arrays[keys[2]][0], device
    )
    first_actions = policy.predict_action_chunk(first_batch).detach().cpu().numpy()
    first_report = copy.deepcopy(policy.last_prediction_report)
    second_batch = legacy.make_policy_batch(
        arrays[keys[0]][request_tick],
        arrays[keys[1]][request_tick],
        arrays[keys[2]][request_tick],
        device,
    )
    second_actions = policy.predict_action_chunk(second_batch).detach().cpu().numpy()
    second_report = copy.deepcopy(policy.last_prediction_report)

    controller = HPTDecodedJointRTCController(
        policy,
        args.rtc_min_suffix_actions,
        args.rtc_blend_actions,
    )
    initial = controller.install(
        plan_id=0,
        start_tick=0,
        actions=first_actions,
        prediction_report=first_report,
        result_age_ms=0.0,
        max_result_age_ms=args.rtc_max_result_age_ms,
    )
    if not initial["accepted"]:
        raise RuntimeError(f"offline initial RTC plan was rejected: {initial}")
    controller.action_tick = install_tick
    replacement = controller.install(
        plan_id=1,
        start_tick=request_tick,
        actions=second_actions,
        prediction_report=second_report,
        result_age_ms=0.0,
        max_result_age_ms=args.rtc_max_result_age_ms,
    )
    if not replacement["accepted"]:
        raise RuntimeError(f"offline replacement RTC plan was rejected: {replacement}")
    log.write(
        {
            "event": "hpt_mixed_decoded_joint_rtc_offline_pair",
            "request_tick": request_tick,
            "install_tick": install_tick,
            "first_chunk_report": first_report,
            "second_chunk_report": second_report,
            "initial_install": initial,
            "replacement_install": replacement,
            "command_sent": False,
        }
    )


def parse_args() -> argparse.Namespace:
    rtc_parser = argparse.ArgumentParser(add_help=False)
    rtc_parser.add_argument("--rtc-replan-interval", type=int, default=10)
    rtc_parser.add_argument("--rtc-blend-actions", type=int, default=8)
    rtc_parser.add_argument("--rtc-min-suffix-actions", type=int, default=8)
    rtc_parser.add_argument("--rtc-max-result-age-ms", type=float, default=500.0)
    rtc_parser.add_argument(
        "--feedback-tracking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    rtc_parser.add_argument("--tracking-joint-tolerance", type=float, default=1.0)
    rtc_parser.add_argument("--tracking-min-progress", type=float, default=0.01)
    rtc_parser.add_argument("--tracking-stall-frames", type=int, default=300)
    rtc_parser.add_argument("--tracking-max-hold-frames", type=int, default=500)
    rtc_parser.add_argument("--driver-clip-tolerance", type=float, default=1e-3)
    rtc_parser.add_argument(
        "--gripper-contact-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    rtc_parser.add_argument(
        "--gripper-contact-max-feedback", type=float, default=70.0
    )
    rtc_parser.add_argument(
        "--gripper-contact-stall-frames", type=int, default=15
    )
    rtc_parser.add_argument(
        "--gripper-contact-motion-tolerance", type=float, default=0.5
    )
    rtc_parser.add_argument(
        "--gripper-contact-min-close-frames", type=int, default=10
    )
    rtc_parser.add_argument(
        "--gripper-contact-min-drop", type=float, default=10.0
    )
    rtc_parser.add_argument("--rtc-offline-request-tick", type=int, default=3)
    rtc_parser.add_argument(
        "--rtc-offline-inference-delay-ticks", type=int, default=3
    )
    rtc_args, remaining = rtc_parser.parse_known_args()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = single_async.parse_args()
    finally:
        sys.argv = old_argv
    for key, value in vars(rtc_args).items():
        setattr(args, key, value)
    if args.list_checkpoint_profiles:
        return args
    allowed_profiles = {
        "mixed-clean-v2",
        "mixed-clean-v2-trimmed-15k",
    }
    if args.checkpoint_profile not in allowed_profiles:
        raise ValueError(
            "this RTC entrypoint is intentionally restricted to "
            "audited Mixed Clean V2 profiles: "
            f"{sorted(allowed_profiles)}"
        )
    if int(args.query_frequency) != MODEL_HORIZON:
        raise ValueError("HPT decoded-joint RTC requires --query-frequency 30")
    if args.chunk_smoothing:
        raise ValueError(
            "do not combine decoded-joint RTC with --chunk-smoothing"
        )
    if not 1 <= args.rtc_replan_interval < MODEL_HORIZON:
        raise ValueError("--rtc-replan-interval must be in [1,29]")
    if not 1 <= args.rtc_blend_actions < MODEL_HORIZON:
        raise ValueError("--rtc-blend-actions must be in [1,29]")
    if not 1 <= args.rtc_min_suffix_actions <= MODEL_HORIZON:
        raise ValueError("--rtc-min-suffix-actions must be in [1,30]")
    if args.rtc_max_result_age_ms <= 0:
        raise ValueError("--rtc-max-result-age-ms must be positive")
    if args.tracking_joint_tolerance <= 0:
        raise ValueError("--tracking-joint-tolerance must be positive")
    if args.tracking_min_progress < 0:
        raise ValueError("--tracking-min-progress must be non-negative")
    if args.tracking_stall_frames <= 0 or args.tracking_max_hold_frames <= 0:
        raise ValueError("tracking frame limits must be positive")
    if args.driver_clip_tolerance < 0:
        raise ValueError("--driver-clip-tolerance must be non-negative")
    if not 0 <= args.gripper_contact_max_feedback <= 100:
        raise ValueError("--gripper-contact-max-feedback must be in [0,100]")
    if args.gripper_contact_stall_frames <= 1:
        raise ValueError("--gripper-contact-stall-frames must exceed one")
    if args.gripper_contact_motion_tolerance < 0:
        raise ValueError("--gripper-contact-motion-tolerance must be non-negative")
    if args.gripper_contact_min_close_frames < 0:
        raise ValueError("--gripper-contact-min-close-frames must be non-negative")
    if not 0 <= args.gripper_contact_min_drop <= 100:
        raise ValueError("--gripper-contact-min-drop must be in [0,100]")
    if not 0 <= args.rtc_offline_request_tick < MODEL_HORIZON:
        raise ValueError("--rtc-offline-request-tick must be in [0,29]")
    if not 0 <= args.rtc_offline_inference_delay_ticks < MODEL_HORIZON:
        raise ValueError("--rtc-offline-inference-delay-ticks must be in [0,29]")
    if (
        args.rtc_offline_request_tick
        + args.rtc_offline_inference_delay_ticks
        >= MODEL_HORIZON
    ):
        raise ValueError("offline RTC install tick must remain within 30 actions")

    # The reused guarded runtime passes these fields to the substituted RTC
    # controller.  Their async names remain internal implementation details.
    args.async_replan_lead_ticks = int(args.rtc_replan_interval)
    args.async_min_suffix_actions = int(args.rtc_min_suffix_actions)
    args.async_transition_blend_actions = int(args.rtc_blend_actions)
    args.async_max_result_age_ms = float(args.rtc_max_result_age_ms)
    # Compatibility-only fields expected by the isolated async loop.  The
    # legacy HPT state machine intentionally keeps its original semantics.
    args.gripper_settle_frames = 0
    args.gripper_settle_tolerance = 0.0
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
        print(
            "HPT_MIXED_DECODED_JOINT_RTC_CHECKPOINT_OK "
            f"profile={args.checkpoint_profile} sha256={actual_sha256} "
            "robot_constructed=false camera_opened=false can_opened=false"
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = async_runtime.AsyncJsonlLogger(
        args.log_dir / f"hpt_mixed_joint_rtc_{args.mode}_{timestamp}.jsonl",
        max_pending=args.async_log_queue_size,
    )
    try:
        log.write(
            {
                "event": "start",
                "mode": args.mode,
                "checkpoint_profile": args.checkpoint_profile,
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": actual_sha256,
                "checkpoint_contract": policy.checkpoint_report,
                "rtc": True,
                "rtc_space": "decoded_physical_6d_joint_radians",
                "rtc_replan_interval": args.rtc_replan_interval,
                "rtc_blend_actions": args.rtc_blend_actions,
                "rtc_min_suffix_actions": args.rtc_min_suffix_actions,
                "rtc_max_result_age_ms": args.rtc_max_result_age_ms,
                "gripper_rtc_guidance": False,
                "gripper_timeline": "absolute_tick_binary_state_machine",
                "gripper_contact_detection": args.gripper_contact_detection,
                "gripper_contact_max_feedback": args.gripper_contact_max_feedback,
                "gripper_contact_stall_frames": args.gripper_contact_stall_frames,
                "gripper_contact_motion_tolerance": (
                    args.gripper_contact_motion_tolerance
                ),
                "gripper_contact_min_close_frames": (
                    args.gripper_contact_min_close_frames
                ),
                "gripper_contact_min_drop": args.gripper_contact_min_drop,
                "control_frequency_hz": 30,
                "successful_action_tick_clock": True,
                "inference_thread": True,
                "camera_sampler_thread": True,
                "jsonl_writer_thread": True,
                "command_sent": False,
            }
        )
        if args.mode == "offline":
            run_offline_rtc_pair(args, policy, log, device)
        else:
            run_live_rtc(
                args,
                policy,
                log,
                joint_limits,
                gripper_stroke_mm,
            )
    finally:
        log.close()
        logger.info("wrote HPT Mixed RTC audit log to %s", log.path)


if __name__ == "__main__":
    main()
