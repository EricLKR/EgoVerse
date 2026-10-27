#!/usr/bin/env python
"""Feedback-paced trajectory retiming for Assembly New1 asynchronous HPT RTC.

This is an additive entrypoint.  It leaves the trained policy, decoded physical
joint path, gripper state machine, homing, command mapping, and hardware safety
layer unchanged.  It only changes how decoded 6-D joint waypoints are timed:

* each policy waypoint is linearly subdivided in physical joint-angle space;
* a sub-command is never allowed to lead measured feedback by more than the
  configured physical-angle bound;
* policy/gripper time advances only after the complete original waypoint has
  been accepted, so inserted sub-commands do not consume the 30-step chunk;
* a buffered RTC chunk is rejected at its boundary when the decoded-joint jump
  or the bridge correction exceeds the configured continuity authority.

The existing SDK relative-target limiter remains the final hardware guard.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

import deploy_assembly_new1_hpt_joint_async_rtc as base
import deploy_assembly_new1_hpt_joint_async_rtc_boundary_bridge as boundary


@dataclass(frozen=True)
class RetimeOptions:
    advance_mode: str
    max_joint_step_deg: float
    max_feedback_lead_deg: float
    max_substeps_per_action: int
    max_switch_joint_rms_deg: float
    max_bridge_correction_deg: float
    model_limit_soft_clamp_tolerance_deg: float
    command_offsets_deg: tuple[float, float, float, float, float, float]


_OPTIONS: RetimeOptions | None = None
_ACTIVE_CONTROLLER: "FeedbackRetimedDecodedJointRtcController | None" = None


def _parse_retime_options() -> RetimeOptions:
    argv_snapshot = list(sys.argv[1:])

    def _last_base_float(option: str, default: float = 0.0) -> float:
        result = float(default)
        for index, token in enumerate(argv_snapshot):
            if token == option and index + 1 < len(argv_snapshot):
                result = float(argv_snapshot[index + 1])
            elif token.startswith(option + "="):
                result = float(token.split("=", 1)[1])
        return result

    # ``--mode`` belongs to the base deployment parser.  Abbreviation must be
    # disabled here or argparse treats ``--mode`` as an abbreviation of the
    # custom ``--model-limit-soft-clamp-tolerance-deg`` option.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--retime-advance-mode",
        choices=("feedback-complete", "command-substep", "control-frame"),
        default="feedback-complete",
    )
    parser.add_argument("--retime-max-joint-step-deg", type=float, default=0.35)
    parser.add_argument(
        "--retime-max-feedback-lead-deg", type=float, default=0.70
    )
    parser.add_argument("--retime-max-substeps-per-action", type=int, default=64)
    parser.add_argument(
        "--rtc-max-switch-joint-rms-deg", type=float, default=5.0
    )
    parser.add_argument(
        "--rtc-max-bridge-correction-deg", type=float, default=10.0
    )
    parser.add_argument(
        "--model-limit-soft-clamp-tolerance-deg", type=float, default=0.25
    )
    known, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    values = {
        "retime_max_joint_step_deg": known.retime_max_joint_step_deg,
        "retime_max_feedback_lead_deg": known.retime_max_feedback_lead_deg,
        "retime_max_substeps_per_action": known.retime_max_substeps_per_action,
        "rtc_max_switch_joint_rms_deg": known.rtc_max_switch_joint_rms_deg,
        "rtc_max_bridge_correction_deg": known.rtc_max_bridge_correction_deg,
        "model_limit_soft_clamp_tolerance_deg": (
            known.model_limit_soft_clamp_tolerance_deg
        ),
    }
    for name, value in values.items():
        if float(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    return RetimeOptions(
        advance_mode=str(known.retime_advance_mode),
        max_joint_step_deg=float(known.retime_max_joint_step_deg),
        max_feedback_lead_deg=float(known.retime_max_feedback_lead_deg),
        max_substeps_per_action=int(known.retime_max_substeps_per_action),
        max_switch_joint_rms_deg=float(known.rtc_max_switch_joint_rms_deg),
        max_bridge_correction_deg=float(known.rtc_max_bridge_correction_deg),
        model_limit_soft_clamp_tolerance_deg=float(
            known.model_limit_soft_clamp_tolerance_deg
        ),
        command_offsets_deg=(
            0.0,
            _last_base_float("--j2-command-offset-deg"),
            _last_base_float("--j3-command-offset-deg"),
            0.0,
            _last_base_float("--j5-command-offset-deg"),
            0.0,
        ),
    )


class SoftClampJointSpaceAdapter(base.legacy.JointSpaceAdapter):
    """Apply command offsets in physical space before driver encoding.

    The legacy entrypoint applies J2/J3/J5 offsets after this adapter has
    already encoded a physical target into the hardware driver's ``[-100,100]``
    range.  That ordering rejects a harmless intermediate target such as
    J5=86.514 deg even when the configured -1 deg offset makes the command
    actually sent to firmware 85.514 deg.  This additive adapter applies all
    three offsets first, validates/clamps the final physical target, and only
    then lets the base adapter encode it.
    """

    def model_degrees_to_hardware_degrees(
        self, degrees
    ) -> np.ndarray:
        if _OPTIONS is None:
            raise RuntimeError("retime options were not initialized")
        unshifted_hardware = self._base_model_to_hardware(degrees) + np.asarray(
            self.hardware_minus_model_zero_offset_deg, dtype=np.float64
        )
        low = np.asarray(
            [pair[0] for pair in self.hardware_limits_deg], dtype=np.float64
        )
        high = np.asarray(
            [pair[1] for pair in self.hardware_limits_deg], dtype=np.float64
        )
        if unshifted_hardware.shape != (6,) or not np.isfinite(
            unshifted_hardware
        ).all():
            raise ValueError(
                "invalid model physical joint target: "
                f"{unshifted_hardware.tolist()}"
            )
        tolerance = float(_OPTIONS.model_limit_soft_clamp_tolerance_deg)
        command_offsets = np.asarray(
            _OPTIONS.command_offsets_deg, dtype=np.float64
        )
        final_hardware = unshifted_hardware + command_offsets
        outside = np.flatnonzero(
            (final_hardware < low - tolerance)
            | (final_hardware > high + tolerance)
        )
        if len(outside):
            details = {
                int(index): {
                    "unshifted_degrees": float(unshifted_hardware[index]),
                    "command_offset_deg": float(command_offsets[index]),
                    "final_degrees": float(final_hardware[index]),
                    "allowed": [float(low[index]), float(high[index])],
                    "soft_clamp_tolerance_deg": tolerance,
                }
                for index in outside
            }
            raise RuntimeError(
                "model physical joint target materially exceeds original-leader "
                f"physical limits: {details}"
            )
        clipped = np.clip(final_hardware, low, high)
        clipped_indices = np.flatnonzero(
            np.abs(clipped - final_hardware) > 1e-12
        )
        if len(clipped_indices):
            base.logger.warning(
                "Retimed deployment soft-clamped tiny model limit overshoot: %s",
                {
                    int(index): {
                        "unshifted_predicted_deg": float(
                            unshifted_hardware[index]
                        ),
                        "command_offset_deg": float(command_offsets[index]),
                        "final_predicted_deg": float(final_hardware[index]),
                        "final_sent_deg": float(clipped[index]),
                        "correction_deg": float(
                            clipped[index] - final_hardware[index]
                        ),
                    }
                    for index in clipped_indices
                },
            )
        return clipped.copy()

    def report(self) -> dict[str, Any]:
        report = super().report()
        report["model_limit_soft_clamp_tolerance_deg"] = (
            None
            if _OPTIONS is None
            else float(_OPTIONS.model_limit_soft_clamp_tolerance_deg)
        )
        report["model_limit_soft_clamp_scope"] = (
            "J2/J3/J5 offsets applied in physical degrees before driver encoding; "
            "material final-target violations fail"
        )
        report["command_offsets_applied_before_driver_encoding_deg"] = list(
            _OPTIONS.command_offsets_deg if _OPTIONS is not None else (0.0,) * 6
        )
        return report


def _offset_already_applied_to_hardware_action(
    joint_name: str,
    action: dict[str, float],
    offset_deg: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """No-op legacy stage after the adapter has applied physical offsets."""

    offset = float(offset_deg)
    if not math.isfinite(offset):
        raise ValueError(f"{joint_name} command offset must be finite")
    return dict(action), {
        f"{joint_name.lower()}_command_offset_deg": offset,
        "already_applied": True,
        "application_space": "physical_degrees_before_driver_encoding",
    }


def _j2_offset_already_applied_to_hardware_action(
    action: dict[str, float],
    offset_deg: float,
    _joint_limits,
):
    return _offset_already_applied_to_hardware_action("J2", action, offset_deg)


def _j3_offset_already_applied_to_hardware_action(
    action: dict[str, float],
    offset_deg: float,
    _joint_limits,
):
    return _offset_already_applied_to_hardware_action("J3", action, offset_deg)


def _j5_offset_already_applied_to_hardware_action(
    action: dict[str, float],
    offset_deg: float,
    _joint_limits,
):
    return _offset_already_applied_to_hardware_action("J5", action, offset_deg)


def _offset_already_applied_to_initial_pose(
    pose,
    offset_deg: float,
    _joint_limits,
):
    """Keep Home unchanged because the adapter already returned final angles."""

    if not math.isfinite(float(offset_deg)):
        raise ValueError("command offset must be finite")
    return pose


class FeedbackRetimedDecodedJointRtcController(
    boundary.BoundaryBridgedDecodedJointRtcController
):
    """Preserve decoded waypoints while pacing them against measured feedback."""

    def __init__(self, *args, **kwargs) -> None:
        if _OPTIONS is None:
            raise RuntimeError("retime options were not initialized")
        super().__init__(*args, **kwargs)
        self.options = _OPTIONS
        self._retime_key: tuple[int, int, int] | None = None
        self._retime_anchor_rad: np.ndarray | None = None
        self._retime_target_rad: np.ndarray | None = None
        self._retime_gripper_driver = 100.0
        self._retime_substep_index = 0
        self._retime_substep_count = 1
        self._pending_substep_reached = False
        self._pending_policy_tick_complete = False
        self._last_advance_completed_policy_tick = False
        self._last_retime_report: dict[str, Any] | None = None
        global _ACTIVE_CONTROLLER
        _ACTIVE_CONTROLLER = self

    def _boundary_metrics(self) -> dict[str, Any] | None:
        if self.buffered_plan is None or self._last_sent_driver is None:
            return None
        if self.active_plan is not None:
            active_end_tick = int(
                self.active_plan.start_tick + len(self.active_plan.physical_actions)
            )
            if self.action_tick < active_end_tick:
                return None
        buffered = self.buffered_plan
        prefix = int(self.action_tick - buffered.start_tick)
        remaining = int(len(buffered.physical_actions) - prefix)
        if prefix < 0 or remaining < self.min_suffix_actions:
            return None
        physical = np.asarray(buffered.physical_actions, dtype=np.float64)
        last_rad = self._driver_to_joint_rad(self._last_sent_driver)
        velocity_rad = self._driver_velocity_to_joint_rad(
            self._last_command_velocity_driver
        )
        raw_first = physical[prefix, :6]
        switch_rms_deg = math.degrees(
            float(np.sqrt(np.mean(np.square(raw_first - last_rad))))
        )
        bridge_count = min(self.blend_actions, remaining)
        low = np.asarray(self.policy._joint_low_rad, dtype=np.float64)
        high = np.asarray(self.policy._joint_high_rad, dtype=np.float64)
        max_correction_rad = 0.0
        for bridge_index in range(bridge_count):
            alpha = self._quintic_take_new_weight(bridge_index, bridge_count)
            continuation = np.clip(
                last_rad + float(bridge_index + 1) * velocity_rad,
                low,
                high,
            )
            raw_new = physical[prefix + bridge_index, :6]
            bridged = (1.0 - alpha) * continuation + alpha * raw_new
            max_correction_rad = max(
                max_correction_rad,
                float(np.max(np.abs(bridged - raw_new))),
            )
        return {
            "prefix": prefix,
            "remaining": remaining,
            "bridge_count": bridge_count,
            "switch_joint_rms_deg": switch_rms_deg,
            "max_switch_joint_delta_deg": float(
                np.max(np.abs(np.rad2deg(raw_first - last_rad)))
            ),
            "max_bridge_correction_deg": math.degrees(max_correction_rad),
        }

    def install(
        self,
        *,
        plan_id: int,
        start_tick: int,
        actions: np.ndarray,
        prediction_report: dict[str, Any] | None,
        result_age_ms: float,
        max_result_age_ms: float,
        include_diagnostic: bool = False,
    ) -> dict[str, Any]:
        """Gate a fresh immediate replacement after an exhausted old plan."""
        active_end_tick = (
            None
            if self.active_plan is None
            else int(
                self.active_plan.start_tick
                + len(self.active_plan.physical_actions)
            )
        )
        immediate_boundary = bool(
            self.active_plan is not None
            and active_end_tick is not None
            and self.action_tick >= active_end_tick
            and self._last_sent_driver is not None
        )
        if immediate_boundary and result_age_ms <= max_result_age_ms:
            physical = base.physical_chunk_from_report(
                prediction_report, len(np.asarray(actions))
            )
            prefix = int(self.action_tick - int(start_tick))
            remaining = int(len(physical) - prefix)
            if prefix >= 0 and remaining >= self.min_suffix_actions:
                last_rad = self._driver_to_joint_rad(self._last_sent_driver)
                first = physical[prefix, :6].astype(np.float64)
                switch_rms_deg = math.degrees(
                    float(np.sqrt(np.mean(np.square(first - last_rad))))
                )
                max_correction_deg = float(
                    np.max(np.abs(np.rad2deg(first - last_rad)))
                )
                hard_connection_limit_deg = float(
                    self.options.max_joint_step_deg
                    * self.options.max_substeps_per_action
                )
                if max_correction_deg > hard_connection_limit_deg:
                    raise RuntimeError(
                        "fresh RTC plan cannot be safely connected by the "
                        "feedback retimer: "
                        f"max_joint_delta={max_correction_deg:.3f}deg, "
                        f"capacity={hard_connection_limit_deg:.3f}deg"
                    )
                soft_limit_exceeded = bool(
                    switch_rms_deg > self.options.max_switch_joint_rms_deg
                    or max_correction_deg
                    > self.options.max_bridge_correction_deg
                )
            else:
                soft_limit_exceeded = False
                switch_rms_deg = None
                max_correction_deg = None
        else:
            soft_limit_exceeded = False
            switch_rms_deg = None
            max_correction_deg = None
        report = super().install(
            plan_id=plan_id,
            start_tick=start_tick,
            actions=actions,
            prediction_report=prediction_report,
            result_age_ms=result_age_ms,
            max_result_age_ms=max_result_age_ms,
            include_diagnostic=include_diagnostic,
        )
        if immediate_boundary and report.get("accepted"):
            report["boundary_continuity_soft_limit_exceeded"] = bool(
                soft_limit_exceeded
            )
            report["boundary_continuity_action"] = (
                "accepted_with_feedback_retiming"
                if soft_limit_exceeded
                else "accepted_within_soft_limits"
            )
            report["switch_joint_rms_before_deg"] = switch_rms_deg
            report["switch_joint_rms_limit_deg"] = float(
                self.options.max_switch_joint_rms_deg
            )
            report["max_bridge_correction_deg"] = max_correction_deg
            report["max_bridge_correction_limit_deg"] = float(
                self.options.max_bridge_correction_deg
            )
        return report

    def _activate_buffered_plan_if_ready(self) -> dict[str, Any] | None:
        metrics = self._boundary_metrics()
        if metrics is not None:
            soft_limit_exceeded = bool(
                metrics["switch_joint_rms_deg"]
                > self.options.max_switch_joint_rms_deg
                or
                metrics["max_bridge_correction_deg"]
                > self.options.max_bridge_correction_deg
            )
            hard_connection_limit_deg = float(
                self.options.max_joint_step_deg
                * self.options.max_substeps_per_action
            )
            if (
                metrics["max_switch_joint_delta_deg"]
                > hard_connection_limit_deg
            ):
                raise RuntimeError(
                    "buffered RTC plan cannot be safely connected by the "
                    "feedback retimer: "
                    f"max_joint_delta={metrics['max_switch_joint_delta_deg']:.3f}deg, "
                    f"capacity={hard_connection_limit_deg:.3f}deg"
                )
        else:
            soft_limit_exceeded = False
        event = super()._activate_buffered_plan_if_ready()
        if event is not None and event.get("activated") and metrics is not None:
            event["boundary_continuity_soft_limit_exceeded"] = bool(
                soft_limit_exceeded
            )
            event["boundary_continuity_action"] = (
                "accepted_with_feedback_retiming"
                if soft_limit_exceeded
                else "accepted_within_soft_limits"
            )
            event["switch_joint_rms_limit_deg"] = float(
                self.options.max_switch_joint_rms_deg
            )
            event["max_bridge_correction_limit_deg"] = float(
                self.options.max_bridge_correction_deg
            )
            self._last_boundary_activation = event
        return event

    def _initialize_segment(
        self,
        *,
        key: tuple[int, int, int],
        target_rad: np.ndarray,
        gripper_driver: float,
        current_driver: np.ndarray,
    ) -> None:
        anchor_driver = (
            np.asarray(current_driver, dtype=np.float64).reshape(6)
            if self._last_sent_driver is None
            else self._last_sent_driver.copy()
        )
        anchor_rad = self._driver_to_joint_rad(anchor_driver)
        target_rad = np.asarray(target_rad, dtype=np.float64).reshape(6)
        max_delta_deg = float(
            np.max(np.abs(np.rad2deg(target_rad - anchor_rad)))
        )
        substeps = max(
            1,
            int(math.ceil(max_delta_deg / self.options.max_joint_step_deg)),
        )
        if substeps > self.options.max_substeps_per_action:
            raise RuntimeError(
                "retimed waypoint requires too many substeps: "
                f"required={substeps}, limit={self.options.max_substeps_per_action}, "
                f"max_delta_deg={max_delta_deg:.3f}"
            )
        self._retime_key = key
        self._retime_anchor_rad = anchor_rad
        self._retime_target_rad = target_rad.copy()
        self._retime_gripper_driver = float(gripper_driver)
        self._retime_substep_index = 0
        self._retime_substep_count = substeps

    def peek(self, driver_state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        self._last_advance_completed_policy_tick = False
        self._activate_buffered_plan_if_ready()
        if self.active_plan is None:
            raise RuntimeError("RTC has no active action plan")
        requested_offset = int(self.action_tick - self.active_plan.start_tick)
        if requested_offset < 0:
            raise RuntimeError(
                "RTC active plan starts in the future: "
                f"tick={self.action_tick}, plan={self.active_plan.plan_id}, "
                f"offset={requested_offset}"
            )
        plan_exhausted_hold = requested_offset >= len(
            self.active_plan.physical_actions
        )
        offset = min(requested_offset, len(self.active_plan.physical_actions) - 1)
        physical = np.asarray(
            self.active_plan.physical_actions[offset], dtype=np.float64
        )
        key = (int(self.active_plan.plan_id), int(self.action_tick), int(offset))
        target_changed = (
            self._retime_target_rad is None
            or not np.allclose(
                self._retime_target_rad, physical[:6], atol=1e-10, rtol=0.0
            )
        )
        current_driver = np.asarray(driver_state[:6], dtype=np.float64)
        continuous_mode = self.options.advance_mode == "control-frame"
        if continuous_mode:
            # Track the next absolute model target every control frame.  Carry
            # any unexecuted displacement through ``_last_sent_driver`` rather
            # than holding the model clock until a discrete subgoal is reached.
            # This removes stop-and-go timing while preserving a strict
            # physical-angle step bound on every sent frame.
            anchor_driver = (
                current_driver
                if self._last_sent_driver is None
                else self._last_sent_driver.copy()
            )
            anchor_rad = self._driver_to_joint_rad(anchor_driver)
            target_rad = np.asarray(physical[:6], dtype=np.float64).reshape(6)
            remaining_delta_deg = np.rad2deg(target_rad - anchor_rad)
            bounded_delta_deg = np.clip(
                remaining_delta_deg,
                -self.options.max_joint_step_deg,
                self.options.max_joint_step_deg,
            )
            desired_rad = anchor_rad + np.deg2rad(bounded_delta_deg)
            max_remaining_delta_deg = float(
                np.max(np.abs(remaining_delta_deg))
            )
            self._retime_key = key
            self._retime_anchor_rad = anchor_rad
            self._retime_target_rad = target_rad.copy()
            self._retime_gripper_driver = float(physical[6] >= 0.5) * 100.0
            self._retime_substep_index = 0
            self._retime_substep_count = max(
                1,
                int(
                    math.ceil(
                        max_remaining_delta_deg
                        / self.options.max_joint_step_deg
                    )
                ),
            )
            continuous_catchup_limited = bool(
                self._retime_substep_count
                > self.options.max_substeps_per_action
            )
            alpha = min(
                1.0,
                self.options.max_joint_step_deg
                / max(max_remaining_delta_deg, 1e-12),
            )
        else:
            continuous_catchup_limited = False
            if self._retime_key != key or target_changed:
                self._initialize_segment(
                    key=key,
                    target_rad=physical[:6],
                    gripper_driver=float(physical[6] >= 0.5) * 100.0,
                    current_driver=current_driver,
                )
            assert self._retime_anchor_rad is not None
            assert self._retime_target_rad is not None
            alpha = float(self._retime_substep_index + 1) / float(
                self._retime_substep_count
            )
            desired_rad = (
                (1.0 - alpha) * self._retime_anchor_rad
                + alpha * self._retime_target_rad
            )
        desired_driver = np.asarray(
            self.policy._joint_rad_to_driver(desired_rad), dtype=np.float64
        )

        current_rad = self._driver_to_joint_rad(current_driver)
        feedback_delta_deg = np.rad2deg(desired_rad - current_rad)
        max_feedback_lead_deg = float(np.max(np.abs(feedback_delta_deg)))
        lead_scale = 1.0
        if max_feedback_lead_deg > self.options.max_feedback_lead_deg:
            lead_scale = (
                self.options.max_feedback_lead_deg / max_feedback_lead_deg
            )
            limited_rad = current_rad + np.deg2rad(feedback_delta_deg * lead_scale)
            desired_driver = np.asarray(
                self.policy._joint_rad_to_driver(limited_rad), dtype=np.float64
            )

        raw_driver = np.empty(7, dtype=np.float64)
        raw_driver[:6] = desired_driver
        raw_driver[6] = self._retime_gripper_driver
        safe_action, pending_velocity = self._filter_from_previous_sent_command(
            raw_driver,
            current_driver,
        )
        filter_delta = np.asarray(safe_action[:6], dtype=np.float64) - desired_driver
        filter_limited = not np.allclose(filter_delta, 0.0, atol=1e-5, rtol=0.0)
        if continuous_mode:
            # ``advance`` is called only after the command has been accepted by
            # the send path.  Advance absolute policy time once per successful
            # control frame; command/feedback lag remains bounded and is
            # carried into the following frame instead of inserting a pause.
            self._pending_substep_reached = True
            # If policy time has outrun the bounded physical command by more
            # than the configured catch-up authority, keep sending bounded
            # catch-up commands but temporarily stop policy-time advancement.
            # This is fail-slow instead of aborting or chasing an arbitrarily
            # stale trajectory.
            self._pending_policy_tick_complete = bool(
                not plan_exhausted_hold and not continuous_catchup_limited
            )
        elif self.options.advance_mode == "command-substep":
            # Advance one interpolated substep for each command accepted by the
            # send path.  Do not wait for feedback to land exactly on the
            # subgoal; do not advance policy time until the complete waypoint
            # has been issued.  This preserves path phase without stop-and-go.
            self._pending_substep_reached = True
            self._pending_policy_tick_complete = bool(
                self._retime_substep_index + 1 >= self._retime_substep_count
                and not plan_exhausted_hold
            )
        else:
            self._pending_substep_reached = bool(
                lead_scale >= 1.0 - 1e-12 and not filter_limited
            )
            self._pending_policy_tick_complete = bool(
                self._pending_substep_reached
                and self._retime_substep_index + 1 >= self._retime_substep_count
                and not plan_exhausted_hold
            )
        self._pending_velocity = np.asarray(pending_velocity, dtype=np.float64)
        self._pending_sent_driver = safe_action[:6].astype(np.float64).copy()
        self._last_retime_report = {
            "enabled": True,
            "advance_mode": self.options.advance_mode,
            "policy_tick": int(self.action_tick),
            "plan_id": int(self.active_plan.plan_id),
            "plan_offset": int(offset),
            "substep_index": int(self._retime_substep_index),
            "substeps_total": int(self._retime_substep_count),
            "interpolation_alpha": alpha,
            "max_joint_step_deg": float(self.options.max_joint_step_deg),
            "max_feedback_lead_deg": float(
                self.options.max_feedback_lead_deg
            ),
            "observed_feedback_lead_before_limit_deg": max_feedback_lead_deg,
            "feedback_lead_scale": float(lead_scale),
            "command_filter_limited": bool(filter_limited),
            "continuous_catchup_limited": bool(
                continuous_catchup_limited
            ),
            "pending_substep_reached": bool(self._pending_substep_reached),
            "pending_policy_tick_complete": bool(
                self._pending_policy_tick_complete
            ),
        }
        status = self.status(offset=offset)
        status.update(
            {
                "requested_plan_offset": int(requested_offset),
                "plan_exhausted_hold": bool(plan_exhausted_hold),
                "command_filter_anchor": "previous_sent_command",
                "command_filter": copy.deepcopy(self._pending_filter_report),
                "retimer": copy.deepcopy(self._last_retime_report),
            }
        )
        return safe_action, status

    def advance(self) -> None:
        if self._pending_sent_driver is None:
            raise RuntimeError("RTC cannot advance before peeking a sent command")
        pending = self._pending_sent_driver.copy()
        self._previous_sent_driver = (
            None
            if self._last_sent_driver is None
            else self._last_sent_driver.copy()
        )
        self._last_sent_driver = pending
        if self._previous_sent_driver is None:
            self._last_command_velocity_driver.fill(0.0)
        else:
            self._last_command_velocity_driver = (
                self._last_sent_driver - self._previous_sent_driver
            )
        policy_tick_complete = bool(self._pending_policy_tick_complete)
        if self._pending_substep_reached:
            self._retime_substep_index += 1
        if policy_tick_complete:
            self.action_tick += 1
            self._retime_key = None
            self._retime_anchor_rad = None
            self._retime_target_rad = None
            self._retime_substep_index = 0
            self._retime_substep_count = 1
        self._last_advance_completed_policy_tick = policy_tick_complete
        self._pending_velocity = None
        self._pending_sent_driver = None
        self._pending_filter_report = None
        self._pending_substep_reached = False
        self._pending_policy_tick_complete = False

    def cancel_pending(self) -> None:
        super().cancel_pending()
        self._pending_substep_reached = False
        self._pending_policy_tick_complete = False
        self._last_advance_completed_policy_tick = False

    def status(self, *, offset: int | None = None) -> dict[str, Any]:
        report = super().status(offset=offset)
        report["feedback_retimer"] = {
            "enabled": True,
            "advance_mode": self.options.advance_mode,
            "max_joint_step_deg": float(self.options.max_joint_step_deg),
            "max_feedback_lead_deg": float(
                self.options.max_feedback_lead_deg
            ),
            "max_substeps_per_action": int(
                self.options.max_substeps_per_action
            ),
            "substep_index": int(self._retime_substep_index),
            "substeps_total": int(self._retime_substep_count),
            "last_accepted_command_completed_policy_tick": bool(
                self._last_advance_completed_policy_tick
            ),
        }
        report["boundary_continuity_gate"] = {
            "max_switch_joint_rms_deg": float(
                self.options.max_switch_joint_rms_deg
            ),
            "max_bridge_correction_deg": float(
                self.options.max_bridge_correction_deg
            ),
        }
        return report


class RetimedAbsoluteTimeGripper(base.AbsoluteTimeGripper):
    """Advance gripper prediction time only with a full policy waypoint."""

    def advance_prediction_tick(self) -> None:
        controller = _ACTIVE_CONTROLLER
        if (
            controller is None
            or controller._last_advance_completed_policy_tick
        ):
            super().advance_prediction_tick()


def _add_boolean_default(enabled: str, disabled: str, *, value: bool) -> None:
    if enabled in sys.argv or disabled in sys.argv:
        return
    sys.argv.append(enabled if value else disabled)


def _add_value_default(option: str, value: str) -> None:
    if option not in sys.argv:
        sys.argv.extend([option, value])


def main() -> None:
    global _OPTIONS
    _OPTIONS = _parse_retime_options()
    _add_boolean_default(
        "--rtc-finish-active-plan-before-install",
        "--no-rtc-finish-active-plan-before-install",
        value=True,
    )
    _add_boolean_default(
        "--rtc-replan-after-contact",
        "--no-rtc-replan-after-contact",
        value=False,
    )
    _add_value_default("--query-frequency", "30")
    _add_value_default("--rtc-replan-interval", "28")
    _add_value_default("--rtc-blend-actions", "8")
    _add_value_default("--rtc-min-suffix-actions", "8")
    _add_value_default("--max-relative-target", "1.5")
    _add_value_default("--max-joint-command-step", "1.0")
    _add_value_default("--max-joint-command-accel", "0.25")
    base.DecodedJointRtcController = FeedbackRetimedDecodedJointRtcController
    base.AbsoluteTimeGripper = RetimedAbsoluteTimeGripper
    base.legacy.JointSpaceAdapter = SoftClampJointSpaceAdapter
    # Offsets are now applied by SoftClampJointSpaceAdapter in physical-angle
    # space.  Disable only the later command/Home offset stages to avoid double
    # application.  Proprioception conversion remains untouched and still
    # subtracts the offsets before model inference.
    base.legacy.apply_j2_command_offset_to_hardware_action = (
        _j2_offset_already_applied_to_hardware_action
    )
    base.legacy.apply_j3_command_offset_to_hardware_action = (
        _j3_offset_already_applied_to_hardware_action
    )
    base.legacy.apply_j5_command_offset_to_hardware_action = (
        _j5_offset_already_applied_to_hardware_action
    )
    base.legacy.apply_j2_command_offset_to_initial_pose = (
        _offset_already_applied_to_initial_pose
    )
    base.legacy.apply_j3_command_offset_to_initial_pose = (
        _offset_already_applied_to_initial_pose
    )
    base.legacy.apply_j5_command_offset_to_initial_pose = (
        _offset_already_applied_to_initial_pose
    )
    base.main()


if __name__ == "__main__":
    main()
