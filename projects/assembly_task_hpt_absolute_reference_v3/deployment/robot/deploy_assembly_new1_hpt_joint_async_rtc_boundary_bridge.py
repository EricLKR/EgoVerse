#!/usr/bin/env python
"""Assembly New1 HPT asynchronous RTC with a committed-tail boundary bridge.

This is an additive deployment entrypoint.  It keeps the current
``finish-active`` scheduler (the active plan always reaches its final action),
but removes the raw jump into the next independently predicted chunk:

* the decoded 6D joint suffix is blended for ``--rtc-blend-actions`` actions
  against a constant-velocity continuation of the last command;
* command velocity and acceleration are defined from the previous *sent
  command*, not reset or re-anchored at a chunk boundary;
* hardware tracking remains enforced by the existing outer command-safety
  layer, which freezes the action tick whenever it must alter a command.

Gripper values are never blended.  The existing binary gripper state machine,
contact replan, command mapping, hardware safety checks, and homing behavior
remain owned by the audited base deployment.
"""

from __future__ import annotations

import copy
import math
import sys
from typing import Any

import numpy as np

import deploy_assembly_new1_hpt_joint_async_rtc as base


class BoundaryBridgedDecodedJointRtcController(base.DecodedJointRtcController):
    """Finish active chunks, then install a C2-weighted decoded-joint bridge."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_sent_driver: np.ndarray | None = None
        self._previous_sent_driver: np.ndarray | None = None
        self._pending_sent_driver: np.ndarray | None = None
        self._last_command_velocity_driver = np.zeros(6, dtype=np.float64)
        self._pending_filter_report: dict[str, Any] | None = None

    @staticmethod
    def _quintic_take_new_weight(index: int, count: int) -> float:
        if count <= 0:
            return 1.0
        s = float(index + 1) / float(count)
        return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5

    def _driver_to_joint_rad(self, driver: np.ndarray) -> np.ndarray:
        driver = np.asarray(driver, dtype=np.float64).reshape(6)
        low = np.asarray(self.policy._joint_low_rad, dtype=np.float64).reshape(6)
        high = np.asarray(self.policy._joint_high_rad, dtype=np.float64).reshape(6)
        return low + (driver + 100.0) / 200.0 * (high - low)

    def _driver_velocity_to_joint_rad(self, velocity: np.ndarray) -> np.ndarray:
        velocity = np.asarray(velocity, dtype=np.float64).reshape(6)
        low = np.asarray(self.policy._joint_low_rad, dtype=np.float64).reshape(6)
        high = np.asarray(self.policy._joint_high_rad, dtype=np.float64).reshape(6)
        return velocity / 200.0 * (high - low)

    def _activate_buffered_plan_if_ready(self) -> dict[str, Any] | None:
        if self.buffered_plan is None:
            return None
        if self.active_plan is not None:
            active_end_tick = int(
                self.active_plan.start_tick + len(self.active_plan.physical_actions)
            )
            if self.action_tick < active_end_tick:
                return None

        buffered = self.buffered_plan
        self.buffered_plan = None
        prefix = int(self.action_tick - buffered.start_tick)
        remaining = int(len(buffered.physical_actions) - prefix)
        if prefix < 0 or remaining < self.min_suffix_actions:
            event = {
                "activated": False,
                "reason": (
                    "buffered_plan_starts_in_the_future"
                    if prefix < 0
                    else "buffered_plan_has_too_few_suffix_actions"
                ),
                "plan_id": int(buffered.plan_id),
                "activation_tick": int(self.action_tick),
                "skipped_prefix_actions": int(max(0, prefix)),
                "remaining_suffix_actions": int(max(0, remaining)),
            }
            self._last_boundary_activation = event
            return event

        physical = np.asarray(buffered.physical_actions, dtype=np.float32).copy()
        bridge_count = min(self.blend_actions, remaining)
        bridge_weights: list[float] = []
        max_correction_rad = 0.0
        switch_rms_before_rad = None
        switch_rms_after_rad = None

        if self._last_sent_driver is not None and bridge_count > 0:
            last_rad = self._driver_to_joint_rad(self._last_sent_driver)
            velocity_rad = self._driver_velocity_to_joint_rad(
                self._last_command_velocity_driver
            )
            low = np.asarray(self.policy._joint_low_rad, dtype=np.float64)
            high = np.asarray(self.policy._joint_high_rad, dtype=np.float64)
            raw_first = physical[prefix, :6].astype(np.float64)
            switch_rms_before_rad = float(
                np.sqrt(np.mean(np.square(raw_first - last_rad)))
            )
            for bridge_index in range(bridge_count):
                action_index = prefix + bridge_index
                alpha = self._quintic_take_new_weight(
                    bridge_index, bridge_count
                )
                bridge_weights.append(float(alpha))
                continuation = np.clip(
                    last_rad + float(bridge_index + 1) * velocity_rad,
                    low,
                    high,
                )
                raw_new = physical[action_index, :6].astype(np.float64)
                bridged = (1.0 - alpha) * continuation + alpha * raw_new
                max_correction_rad = max(
                    max_correction_rad,
                    float(np.max(np.abs(bridged - raw_new))),
                )
                physical[action_index, :6] = bridged.astype(np.float32)
            switch_rms_after_rad = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            physical[prefix, :6].astype(np.float64) - last_rad
                        )
                    )
                )
            )

        prediction_report = copy.deepcopy(buffered.prediction_report)
        prediction_report["selected_physical_actions"] = physical.tolist()
        prediction_report["finish_active_boundary_bridge"] = {
            "enabled": True,
            "activation_tick": int(self.action_tick),
            "start_offset": int(prefix),
            "bridge_actions": int(bridge_count),
            "quintic_take_new_weights": bridge_weights,
            "reference": "last_sent_command_constant_velocity_continuation",
            "gripper_blended": False,
        }
        self.active_plan = base.ActivePlan(
            plan_id=int(buffered.plan_id),
            start_tick=int(buffered.start_tick),
            physical_actions=physical,
            prediction_report=prediction_report,
        )
        event = {
            "activated": True,
            "reason": None,
            "plan_id": int(buffered.plan_id),
            "activation_tick": int(self.action_tick),
            "buffered_tick": int(buffered.buffered_tick),
            "buffer_wait_action_ticks": int(
                self.action_tick - buffered.buffered_tick
            ),
            "skipped_prefix_actions": int(prefix),
            "remaining_suffix_actions": int(remaining),
            "rtc_overlap_blend_actions": int(bridge_count),
            "boundary_bridge": "decoded_joint_quintic_constant_velocity",
            "boundary_bridge_weights": bridge_weights,
            "switch_joint_rms_before_deg": (
                None
                if switch_rms_before_rad is None
                else math.degrees(switch_rms_before_rad)
            ),
            "switch_joint_rms_after_deg": (
                None
                if switch_rms_after_rad is None
                else math.degrees(switch_rms_after_rad)
            ),
            "max_bridge_correction_deg": math.degrees(max_correction_rad),
            "continuity_authority": (
                "decoded_joint_boundary_bridge_plus_previous_sent_command_filter"
            ),
        }
        self._last_boundary_activation = event
        return event

    def _filter_from_previous_sent_command(
        self,
        raw_driver_action: np.ndarray,
        current_driver: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        action = np.asarray(raw_driver_action, dtype=np.float64).reshape(7).copy()
        current = np.asarray(current_driver, dtype=np.float64).reshape(6)
        if self._last_sent_driver is None:
            command_anchor = current.copy()
            previous_velocity = np.zeros(6, dtype=np.float64)
        else:
            command_anchor = self._last_sent_driver.copy()
            previous_velocity = self._last_command_velocity_driver.copy()

        max_step = float(self.policy.config.max_joint_command_step)
        max_accel = float(self.policy.config.max_joint_command_accel)
        # Keep command-space velocity and acceleration continuous.  Do not
        # re-anchor this trajectory to noisy feedback here: the existing outer
        # hardware command guard compares the proposed command with fresh
        # feedback and freezes the action tick if it has to alter it.  That
        # separation preserves both command continuity and tracking safety.
        velocity_low = np.maximum(-max_step, previous_velocity - max_accel)
        velocity_high = np.minimum(max_step, previous_velocity + max_accel)
        command_low = np.maximum.reduce(
            [
                command_anchor + velocity_low,
                np.full(6, -100.0),
            ]
        )
        command_high = np.minimum.reduce(
            [
                command_anchor + velocity_high,
                np.full(6, 100.0),
            ]
        )
        next_command = np.clip(action[:6], command_low, command_high)
        next_velocity = next_command - command_anchor
        self._pending_filter_report = {
            "feasible": True,
            "infeasible_joint_indices": [],
            "max_abs_command_velocity_driver": float(
                np.max(np.abs(next_velocity))
            ),
            "max_abs_command_acceleration_driver": float(
                    np.max(np.abs(next_velocity - previous_velocity))
                ),
            "max_abs_tracking_lead_driver": float(
                np.max(np.abs(next_command - current))
            ),
            "tracking_safety_authority": "outer_hardware_command_guard",
        }
        action[:6] = next_command
        return action.astype(np.float32), next_velocity.astype(np.float64)

    def peek(self, driver_state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        self._activate_buffered_plan_if_ready()
        if self.active_plan is None:
            raise RuntimeError("RTC has no active action plan")
        requested_offset = self.action_tick - self.active_plan.start_tick
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
        physical = self.active_plan.physical_actions[offset]
        raw_driver = np.empty(7, dtype=np.float32)
        raw_driver[:6] = self.policy._joint_rad_to_driver(physical[:6])
        raw_driver[6] = float(physical[6] >= 0.5) * 100.0
        safe_action, pending_velocity = self._filter_from_previous_sent_command(
            raw_driver,
            np.asarray(driver_state[:6], dtype=np.float64),
        )
        self._pending_velocity = np.asarray(pending_velocity, dtype=np.float64)
        self._pending_sent_driver = safe_action[:6].astype(np.float64).copy()
        status = self.status(offset=offset)
        status.update(
            {
                "requested_plan_offset": int(requested_offset),
                "plan_exhausted_hold": bool(plan_exhausted_hold),
                "command_filter_anchor": "previous_sent_command",
                "command_filter": copy.deepcopy(self._pending_filter_report),
            }
        )
        return safe_action, status

    def cancel_pending(self) -> None:
        super().cancel_pending()
        self._pending_sent_driver = None
        self._pending_filter_report = None

    def advance(self) -> None:
        if self._pending_sent_driver is None:
            raise RuntimeError("RTC cannot advance before peeking a sent command")
        pending = self._pending_sent_driver.copy()
        super().advance()
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
        self._pending_sent_driver = None
        self._pending_filter_report = None


def _add_boolean_default(enabled: str, disabled: str) -> None:
    if enabled not in sys.argv and disabled not in sys.argv:
        sys.argv.append(enabled)


def _add_value_default(option: str, value: str) -> None:
    if option not in sys.argv:
        sys.argv.extend([option, value])


def main() -> None:
    _add_boolean_default(
        "--rtc-finish-active-plan-before-install",
        "--no-rtc-finish-active-plan-before-install",
    )
    _add_boolean_default(
        "--rtc-replan-after-contact",
        "--no-rtc-replan-after-contact",
    )
    _add_value_default("--query-frequency", "30")
    _add_value_default("--rtc-replan-interval", "28")
    _add_value_default("--rtc-blend-actions", "8")
    _add_value_default("--rtc-min-suffix-actions", "8")
    _add_value_default("--max-relative-target", "1.5")
    _add_value_default("--max-joint-command-step", "1.0")
    _add_value_default("--max-joint-command-accel", "0.25")
    base.DecodedJointRtcController = BoundaryBridgedDecodedJointRtcController
    base.main()


if __name__ == "__main__":
    main()
