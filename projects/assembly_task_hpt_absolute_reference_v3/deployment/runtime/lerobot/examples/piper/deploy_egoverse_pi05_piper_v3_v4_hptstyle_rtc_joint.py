#!/usr/bin/env python
"""HPT-style decoded-joint RTC for EgoVerse pi0.5 PIPER V3/V4.

This additive entrypoint leaves the audited synchronous and generative-RTC
pi0.5 deployment files unchanged.  It keeps the complete training
``ModelWrapper`` and native 30-step Gemma action-expert decoder, but applies
continuity only after decoding, following the single-frame Mixed Clean V2 RTC
contract:

* D405 colour gains are estimated from startup frames and then frozen;
* inference runs in one background worker while the old plan is executed;
* every forecast is anchored to the successful action tick at capture time;
* actions that expired during inference are discarded;
* the decoded 6-D arm trajectory is smoothed inside each forecast while the
  binary gripper sequence is preserved bit-for-bit;
* old/new forecasts are blended with a cosine ramp in safe physical 6-D joint
  radians and then converted back to PIPER driver coordinates;
* the committed arm command passes through a causal velocity, acceleration,
  and jerk limiter after RTC plan switching;
* the binary gripper is never numerically blended and instead uses an
  absolute-time, latched state machine with object-contact detection;
* policy time advances only after a command tick is successfully committed.

There is deliberately no Flow-sampler RTC guidance in this file.  It provides
one continuity mechanism, not two stacked mechanisms.  ``checkpoint`` and
``offline`` never construct a robot. ``live-dry-run`` is observation-only.
``execute`` still requires ``--allow-motion``.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import deploy_egoverse_pi05_piper_v3_v4_joint as base
from lerobot.utils.robot_utils import precise_sleep


legacy = base.legacy
logger = logging.getLogger(
    "deploy_egoverse_pi05_piper_v3_v4_hptstyle_rtc_joint"
)

CONTROL_HZ = 30.0
MODEL_HORIZON = 30


def _selected_physical_chunk(report: dict[str, Any]) -> np.ndarray:
    """Return the joint-limited physical chunk actually eligible to execute."""
    physical = np.asarray(report.get("selected_physical_actions"), dtype=np.float32)
    if physical.shape != (MODEL_HORIZON, 7) or not np.isfinite(physical).all():
        raise RuntimeError(
            "HPT-style RTC requires 30 aligned selected physical actions, "
            f"got {physical.shape}"
        )
    if not np.isin(physical[:, 6], (0.0, 1.0)).all():
        raise RuntimeError("selected physical gripper actions are not binary")
    return physical.copy()


def _driver_chunk_from_physical(policy, physical: np.ndarray) -> np.ndarray:
    values = np.asarray(physical, dtype=np.float32)
    if values.shape != (MODEL_HORIZON, 7):
        raise RuntimeError(f"invalid physical action chunk {values.shape}")
    driver = np.empty_like(values, dtype=np.float32)
    driver[:, :6] = np.stack(
        [policy._joint_rad_to_driver(row[:6]) for row in values], axis=0
    )
    driver[:, 6] = values[:, 6] * 100.0
    if not np.isfinite(driver).all():
        raise RuntimeError("physical-to-driver conversion produced NaN/Inf")
    return driver


@dataclass
class ActivePlan:
    plan_id: int
    start_tick: int
    physical: np.ndarray
    driver: np.ndarray
    prediction_report: dict[str, Any]


class HPTStyleDecodedJointController:
    """Absolute-time plan switcher with a fully audited smoothness pipeline."""

    def __init__(
        self,
        policy,
        *,
        min_suffix_actions: int,
        blend_actions: int,
        replan_interval: int,
        intrachunk_smoothing: bool,
        intrachunk_window: int,
        intrachunk_sigma: float,
        max_command_step: float,
        max_command_accel: float,
        max_command_jerk: float,
        command_deadband: float,
    ) -> None:
        self.policy = policy
        self.min_suffix_actions = int(min_suffix_actions)
        self.blend_actions = int(blend_actions)
        self.replan_interval = int(replan_interval)
        self.intrachunk_smoothing = bool(intrachunk_smoothing)
        self.intrachunk_window = int(intrachunk_window)
        self.intrachunk_sigma = float(intrachunk_sigma)
        self.max_command_step = float(max_command_step)
        self.max_command_accel = float(max_command_accel)
        self.max_command_jerk = float(max_command_jerk)
        self.command_deadband = float(command_deadband)
        self.action_tick = 0
        self.active_plan: ActivePlan | None = None
        self.last_request_tick = 0
        self.last_command: np.ndarray | None = None
        self.last_velocity = np.zeros(6, dtype=np.float64)
        self.last_acceleration = np.zeros(6, dtype=np.float64)
        self.pending_command: np.ndarray | None = None
        self.pending_velocity: np.ndarray | None = None
        self.pending_acceleration: np.ndarray | None = None
        self.last_raw_action: np.ndarray | None = None

    @staticmethod
    def _cosine_take_new(index: int, count: int) -> float:
        phase = float(index + 1) / float(count + 1)
        return 0.5 - 0.5 * math.cos(math.pi * phase)

    def note_request(self, start_tick: int) -> None:
        if int(start_tick) != self.action_tick:
            raise RuntimeError(
                "replan request must be anchored to the current successful tick: "
                f"request={start_tick}, current={self.action_tick}"
            )
        self.last_request_tick = int(start_tick)

    def install(
        self,
        *,
        plan_id: int,
        start_tick: int,
        prediction_report: dict[str, Any],
        result_age_ms: float,
        max_result_age_ms: float,
    ) -> dict[str, Any]:
        raw_physical = _selected_physical_chunk(prediction_report)
        raw_smoothness = base.adjacent_joint_rms(raw_physical)
        physical = (
            base.smooth_joint_action_chunk(
                raw_physical,
                self.intrachunk_window,
                self.intrachunk_sigma,
            )
            if self.intrachunk_smoothing
            else raw_physical.copy()
        )
        # Smoothing is strictly arm-only.  Preserve the audited binary output
        # even if the shared helper changes in a future checkout.
        physical[:, 6] = raw_physical[:, 6]
        if not np.isin(physical[:, 6], (0.0, 1.0)).all():
            raise RuntimeError("intrachunk smoothing changed binary gripper values")
        smoothed_smoothness = base.adjacent_joint_rms(physical)
        driver = _driver_chunk_from_physical(self.policy, physical)
        prefix = self.action_tick - int(start_tick)
        remaining = MODEL_HORIZON - prefix
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
        blend_weights: list[float] = []
        before_rad = None
        after_rad = None
        before_driver = None
        after_driver = None
        if accepted and self.active_plan is not None:
            old_offset = self.action_tick - self.active_plan.start_tick
            if (
                0 <= old_offset < MODEL_HORIZON
                and 0 <= prefix < MODEL_HORIZON
            ):
                reference = "existing_absolute_time_joint_plan"
                old_physical = self.active_plan.physical[old_offset, :6].copy()
                old_driver = self.active_plan.driver[old_offset, :6].copy()
                before_rad = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                physical[prefix, :6].astype(np.float64)
                                - old_physical.astype(np.float64)
                            )
                        )
                    )
                )
                before_driver = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                driver[prefix, :6].astype(np.float64)
                                - old_driver.astype(np.float64)
                            )
                        )
                    )
                )
                blend_count = min(
                    self.blend_actions,
                    MODEL_HORIZON - prefix,
                    MODEL_HORIZON - old_offset,
                )
                for blend_index in range(blend_count):
                    alpha = self._cosine_take_new(blend_index, blend_count)
                    blend_weights.append(alpha)
                    old_index = old_offset + blend_index
                    new_index = prefix + blend_index
                    blended = (
                        (1.0 - alpha)
                        * self.active_plan.physical[old_index, :6]
                        + alpha * physical[new_index, :6]
                    )
                    physical[new_index, :6] = blended
                    driver[new_index, :6] = self.policy._joint_rad_to_driver(
                        blended
                    )
                after_rad = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                physical[prefix, :6].astype(np.float64)
                                - old_physical.astype(np.float64)
                            )
                        )
                    )
                )
                after_driver = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                driver[prefix, :6].astype(np.float64)
                                - old_driver.astype(np.float64)
                            )
                        )
                    )
                )

        if accepted:
            self.active_plan = ActivePlan(
                plan_id=int(plan_id),
                start_tick=int(start_tick),
                physical=physical.copy(),
                driver=driver.copy(),
                prediction_report=prediction_report,
            )
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
            "rtc_mode": "post_decode_absolute_time_cosine_overlap",
            "rtc_space": "safe_physical_6d_joint_radians",
            "rtc_gripper_guidance": False,
            "rtc_overlap_reference": reference,
            "rtc_old_source_offset": old_offset,
            "rtc_blend_actions_requested": int(self.blend_actions),
            "rtc_blended_actions": int(blend_count),
            "rtc_cosine_take_new_weights": blend_weights,
            "rtc_intrachunk_smoothing": bool(self.intrachunk_smoothing),
            "rtc_intrachunk_window": int(self.intrachunk_window),
            "rtc_intrachunk_sigma": float(self.intrachunk_sigma),
            "intrachunk_raw_adjacent_joint_rms_rad_mean": float(
                raw_smoothness[0]
            ),
            "intrachunk_raw_adjacent_joint_rms_rad_max": float(
                raw_smoothness[1]
            ),
            "intrachunk_smoothed_adjacent_joint_rms_rad_mean": float(
                smoothed_smoothness[0]
            ),
            "intrachunk_smoothed_adjacent_joint_rms_rad_max": float(
                smoothed_smoothness[1]
            ),
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

    def remaining_actions(self) -> int:
        if self.active_plan is None:
            return 0
        return max(
            0,
            MODEL_HORIZON
            - (self.action_tick - self.active_plan.start_tick),
        )

    def should_request(self, worker_busy: bool) -> bool:
        if worker_busy:
            return False
        if self.active_plan is None:
            return True
        elapsed_since_request = self.action_tick - self.last_request_tick
        return (
            elapsed_since_request >= self.replan_interval
            or self.remaining_actions() <= self.min_suffix_actions
        )

    def status(self) -> dict[str, Any]:
        offset = (
            None
            if self.active_plan is None
            else self.action_tick - self.active_plan.start_tick
        )
        return {
            "action_tick": int(self.action_tick),
            "active_plan_id": (
                None if self.active_plan is None else self.active_plan.plan_id
            ),
            "active_plan_start_tick": (
                None if self.active_plan is None else self.active_plan.start_tick
            ),
            "source_offset": offset,
            "remaining_actions": int(self.remaining_actions()),
            "last_request_tick": int(self.last_request_tick),
            "rtc_space": "safe_physical_6d_joint_radians",
        }

    def _causal_filter(
        self,
        raw_action: np.ndarray,
        current_driver: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        """Apply a causal V/A/J limiter in PIPER driver coordinates.

        Position, velocity, acceleration, and jerk are defined relative to the
        last successfully committed command.  Measured feedback initializes
        that trajectory but never rewrites its derivatives; normal actuator
        lag therefore cannot corrupt the limiter state.  Pending dynamics are
        committed only by :meth:`advance`, so a gripper wait or RTC gap cannot
        advance hidden filter state.  If the robot driver projects a command
        to its relative safety envelope, :meth:`advance` commits the value that
        was actually sent instead of freezing policy time on the same action.
        """
        action = np.asarray(raw_action, dtype=np.float64).reshape(7).copy()
        current = np.asarray(current_driver, dtype=np.float64).reshape(7)
        if not (
            np.isfinite(action).all()
            and np.isfinite(current).all()
            and np.isfinite(self.last_velocity).all()
            and np.isfinite(self.last_acceleration).all()
        ):
            raise RuntimeError("non-finite input to RTC causal trajectory filter")

        if self.last_command is None:
            # Feedback can be marginally outside the normalized command range
            # at a calibrated hard stop (for example -101.0 while the nearest
            # legal command is -100).  Project that one-time initialization
            # without treating it as trajectory velocity.
            self.last_command = np.clip(current[:6], -100.0, 100.0)
        command_anchor = self.last_command
        error = action[:6] - command_anchor
        desired_velocity = np.clip(
            error,
            -self.max_command_step,
            self.max_command_step,
        )
        desired_velocity[np.abs(error) <= self.command_deadband] = 0.0
        desired_acceleration = np.clip(
            desired_velocity - self.last_velocity,
            -self.max_command_accel,
            self.max_command_accel,
        )
        next_acceleration = np.clip(
            desired_acceleration,
            self.last_acceleration - self.max_command_jerk,
            self.last_acceleration + self.max_command_jerk,
        )
        next_acceleration = np.clip(
            next_acceleration,
            -self.max_command_accel,
            self.max_command_accel,
        )
        # Begin removing acceleration early enough that the discrete jerk
        # ramp reaches zero exactly at the velocity limit.  A naive velocity
        # clip would create a hidden acceleration/jerk discontinuity at the
        # saturation point and reintroduce the vibration this filter is meant
        # to remove.
        for index in range(6):
            velocity = float(self.last_velocity[index])
            acceleration = float(next_acceleration[index])
            if velocity >= 0.0 and acceleration > 0.0:
                margin = max(0.0, self.max_command_step - velocity)
                acceleration = min(
                    acceleration,
                    self._max_viable_acceleration(
                        margin, self.max_command_jerk
                    ),
                )
            elif velocity <= 0.0 and acceleration < 0.0:
                margin = max(0.0, self.max_command_step + velocity)
                acceleration = max(
                    acceleration,
                    -self._max_viable_acceleration(
                        margin, self.max_command_jerk
                    ),
                )
            next_acceleration[index] = acceleration
        next_velocity = self.last_velocity + next_acceleration
        if np.any(np.abs(next_velocity) > self.max_command_step + 1e-9):
            raise RuntimeError(
                "causal filter viability failure exceeded velocity bound: "
                f"previous_velocity={self.last_velocity.tolist()} "
                f"previous_acceleration={self.last_acceleration.tolist()} "
                f"next_acceleration={next_acceleration.tolist()} "
                f"next_velocity={next_velocity.tolist()} "
                f"limit={self.max_command_step}"
            )

        # Never deliberately step farther than the current target when the
        # limited velocity is already directed toward it.  This removes the
        # small limit-cycle otherwise created around a changing target.
        toward_target = next_velocity * error > 0.0
        would_overshoot = toward_target & (np.abs(next_velocity) > np.abs(error))
        stop_velocity = error
        stop_acceleration = stop_velocity - self.last_velocity
        stop_jerk = stop_acceleration - self.last_acceleration
        safe_to_stop = would_overshoot & (
            np.abs(stop_acceleration) <= self.max_command_accel + 1e-12
        ) & (np.abs(stop_jerk) <= self.max_command_jerk + 1e-12)
        next_velocity[safe_to_stop] = stop_velocity[safe_to_stop]
        next_acceleration = next_velocity - self.last_velocity

        filtered = action.copy()
        unconstrained_command = command_anchor + next_velocity
        filtered[:6] = np.clip(unconstrained_command, -100.0, 100.0)
        hard_limit_projection = (
            np.abs(filtered[:6] - unconstrained_command) > 1e-12
        )
        next_velocity = filtered[:6] - command_anchor
        next_acceleration = next_velocity - self.last_velocity
        # The normalized command boundary is a physical safety constraint and
        # has priority over differentiability.  Reset acceleration state only
        # on projected axes so the following tick restarts from a feasible
        # boundary state instead of propagating a fictitious large impulse.
        next_acceleration[hard_limit_projection] = 0.0
        jerk = next_acceleration - self.last_acceleration
        report = {
            "space": "piper_driver_coordinates",
            "raw_target": action[:6].tolist(),
            "measured_feedback": current[:6].tolist(),
            "command_anchor": command_anchor.tolist(),
            "command_feedback_error": (
                command_anchor - current[:6]
            ).tolist(),
            "target_error": error.tolist(),
            "desired_velocity": desired_velocity.tolist(),
            "committed_candidate_velocity": next_velocity.tolist(),
            "committed_candidate_acceleration": next_acceleration.tolist(),
            "committed_candidate_jerk": jerk.tolist(),
            "max_command_step": self.max_command_step,
            "max_command_accel": self.max_command_accel,
            "max_command_jerk": self.max_command_jerk,
            "command_deadband": self.command_deadband,
            "velocity_limited_axes": int(
                np.count_nonzero(np.abs(error) > self.max_command_step + 1e-12)
            ),
            "acceleration_limited_axes": int(
                np.count_nonzero(
                    np.abs(desired_velocity - self.last_velocity)
                    > self.max_command_accel + 1e-12
                )
            ),
            "jerk_limited_axes": int(
                np.count_nonzero(
                    np.abs(desired_acceleration - self.last_acceleration)
                    > self.max_command_jerk + 1e-12
                )
            ),
            "overshoot_detected_axes": int(np.count_nonzero(would_overshoot)),
            "overshoot_clamped_axes": int(np.count_nonzero(safe_to_stop)),
            "hard_limit_projection_axes": int(
                np.count_nonzero(hard_limit_projection)
            ),
            "hard_limit_projection_mask": hard_limit_projection.tolist(),
            "hard_limit_projection_resets_acceleration_state": True,
        }
        return (
            filtered.astype(np.float32),
            filtered[:6].copy(),
            next_velocity,
            next_acceleration,
            report,
        )

    @staticmethod
    def _max_viable_acceleration(velocity_margin: float, max_jerk: float) -> float:
        """Largest acceleration whose jerk ramp-to-zero fits in a margin."""
        margin = max(0.0, float(velocity_margin))
        jerk = float(max_jerk)
        if margin <= 0.0:
            return 0.0

        def total_increment(acceleration: float) -> float:
            value = max(0.0, float(acceleration))
            total = 0.0
            while value > 1e-12:
                total += value
                value = max(0.0, value - jerk)
            return total

        low = 0.0
        high = margin
        for _ in range(48):
            midpoint = 0.5 * (low + high)
            if total_increment(midpoint) <= margin:
                low = midpoint
            else:
                high = midpoint
        return low

    def peek(self, current_driver: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        current = np.asarray(current_driver, dtype=np.float64).reshape(7)
        offset = None
        gap_hold = False
        if self.active_plan is not None:
            offset = self.action_tick - self.active_plan.start_tick
        if self.active_plan is not None and offset is not None and 0 <= offset < MODEL_HORIZON:
            raw = self.active_plan.driver[offset].copy()
            self.last_raw_action = raw.copy()
        else:
            gap_hold = True
            raw = np.empty(7, dtype=np.float32)
            raw[:6] = current[:6]
            raw[6] = (
                float(self.last_raw_action[6])
                if self.last_raw_action is not None
                else (100.0 if current[6] >= 50.0 else 0.0)
            )
        (
            safe,
            pending_command,
            pending_velocity,
            pending_acceleration,
            filter_report,
        ) = (
            self._causal_filter(raw, current)
        )
        self.pending_command = pending_command
        self.pending_velocity = pending_velocity
        self.pending_acceleration = pending_acceleration
        report = self.status()
        report.update(
            {
                "source_offset": offset,
                "gap_hold": bool(gap_hold),
                "raw_driver_action": raw.tolist(),
                "causal_filter": filter_report,
            }
        )
        return safe, report

    def advance(
        self,
        committed_arm_action: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if (
            self.pending_command is None
            or self.pending_velocity is None
            or self.pending_acceleration is None
        ):
            raise RuntimeError("cannot advance before peeking an action")
        planned_command = self.pending_command.copy()
        if committed_arm_action is None:
            committed_command = planned_command
        else:
            committed_command = np.asarray(
                committed_arm_action, dtype=np.float64
            ).reshape(6)
            if not np.isfinite(committed_command).all():
                raise RuntimeError("non-finite driver command cannot be committed")
            committed_command = np.clip(committed_command, -100.0, 100.0)
        driver_projection = bool(
            np.any(np.abs(committed_command - planned_command) > 1e-9)
        )
        self.last_command = committed_command.copy()
        if driver_projection:
            # A downstream driver projection is a discontinuous safety event,
            # not a rejected command.  Synchronize position to what was sent
            # and restart derivatives from rest.  This prevents the filter
            # from integrating an unsent target while still allowing the
            # absolute policy clock to leave the clipped action behind.
            self.last_velocity = np.zeros(6, dtype=np.float64)
            self.last_acceleration = np.zeros(6, dtype=np.float64)
        else:
            self.last_velocity = self.pending_velocity.copy()
            self.last_acceleration = self.pending_acceleration.copy()
        self.pending_command = None
        self.pending_velocity = None
        self.pending_acceleration = None
        self.action_tick += 1
        return {
            "committed_source": (
                "driver_sent_action" if committed_arm_action is not None
                else "causal_filter_output"
            ),
            "driver_projection_committed": driver_projection,
            "planned_arm_action": planned_command.tolist(),
            "committed_arm_action": committed_command.tolist(),
            "derivatives_reset_after_driver_projection": driver_projection,
        }

    def cancel_pending(self) -> None:
        self.pending_command = None
        self.pending_velocity = None
        self.pending_acceleration = None


class AbsoluteTimeContactGripper(legacy.IndependentGripperStateMachine):
    """Binary absolute-time gripper with stalled-on-object completion."""

    def __init__(
        self,
        *,
        contact_detection: bool,
        contact_max_feedback: float,
        contact_stall_frames: int,
        contact_motion_tolerance: float,
        contact_min_close_frames: int,
        contact_min_drop: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
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
        close_offsets = np.flatnonzero(commands == 0.0)
        open_offsets = np.flatnonzero(commands == 100.0)
        return {
            "chunk_id": chunk,
            "absolute_start_tick": int(start_tick),
            "install_tick": int(self.prediction_tick),
            "skipped_prefix_actions": int(
                max(0, self.prediction_tick - int(start_tick))
            ),
            "horizon_steps": int(commands.size),
            "close_steps": int(close_offsets.size),
            "open_steps": int(open_offsets.size),
            "first_close_offset": (
                int(close_offsets[0]) if close_offsets.size else None
            ),
            "first_open_offset": (
                int(open_offsets[0]) if open_offsets.size else None
            ),
            "commands_pct": commands.tolist(),
        }

    def _reset_contact_tracking(self) -> None:
        self._contact_feedback_window.clear()
        self._close_start_feedback = None
        self._contact_latched = False

    def update(self, model_command_pct: float, feedback_pct: float) -> dict[str, Any]:
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
                "contact_close_start_feedback": self._close_start_feedback,
                "contact_max_feedback": self.contact_max_feedback,
                "contact_stall_frames": self.contact_stall_frames,
                "contact_motion_tolerance": self.contact_motion_tolerance,
                "contact_min_close_frames": self.contact_min_close_frames,
                "contact_min_drop": self.contact_min_drop,
            }
        )
        return report


@dataclass(frozen=True)
class InferenceRequest:
    plan_id: int
    start_tick: int
    observation: dict[str, Any]
    submitted_monotonic: float


def _copy_model_observation(
    observation: dict[str, Any],
    wrist_color_normalizer,
) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in observation.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
    copied["435if"] = np.ascontiguousarray(
        observation["435if"], dtype=np.uint8
    ).copy()
    wrist = np.asarray(observation["405"], dtype=np.uint8)
    if wrist_color_normalizer is not None:
        wrist = wrist_color_normalizer.apply(wrist)
    copied["405"] = np.ascontiguousarray(wrist, dtype=np.uint8).copy()
    return copied


class BackgroundPredictor:
    """Single producer; robot I/O and plan installation stay on the main thread."""

    def __init__(self, policy, device: torch.device) -> None:
        self.policy = policy
        self.device = device
        self._jobs: queue.Queue[InferenceRequest | None] = queue.Queue(maxsize=1)
        self._completions: queue.Queue[dict[str, Any]] = queue.Queue()
        self._errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)
        self._busy_lock = threading.Lock()
        self._busy = False
        self._thread = threading.Thread(
            target=self._run,
            name="piper-pi05-hptstyle-rtc-inference",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _set_busy(self, value: bool) -> None:
        with self._busy_lock:
            self._busy = bool(value)

    def busy(self) -> bool:
        with self._busy_lock:
            return bool(self._busy)

    def submit(self, request: InferenceRequest) -> None:
        if self.busy():
            raise RuntimeError("cannot submit while inference worker is busy")
        self._set_busy(True)
        try:
            self._jobs.put_nowait(request)
        except Exception:
            self._set_busy(False)
            raise

    def _run(self) -> None:
        while True:
            request = self._jobs.get()
            if request is None:
                return
            started = time.perf_counter()
            try:
                batch = legacy.live_observation_to_batch(
                    request.observation, self.device
                )
                self.policy.predict_action_chunk(batch)
                report = copy.deepcopy(self.policy.last_prediction_report)
                if report is None:
                    raise RuntimeError("background inference produced no report")
                latency_s = time.perf_counter() - started
                self._completions.put(
                    {
                        "event": "hptstyle_rtc_replan_completed",
                        "plan_id": int(request.plan_id),
                        "start_tick": int(request.start_tick),
                        "prediction_report": report,
                        "inference_latency_ms": latency_s * 1e3,
                        "result_age_ms": (
                            time.monotonic() - request.submitted_monotonic
                        )
                        * 1e3,
                        "command_sent": False,
                    }
                )
            except BaseException as error:
                if self._errors.empty():
                    self._errors.put(error)
                self._completions.put(
                    {
                        "event": "hptstyle_rtc_replan_failed",
                        "plan_id": int(request.plan_id),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "command_sent": False,
                    }
                )
            finally:
                self._set_busy(False)

    def drain(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        while True:
            try:
                records.append(self._completions.get_nowait())
            except queue.Empty:
                return records

    def raise_if_failed(self) -> None:
        try:
            error = self._errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("background HPT-style RTC inference failed") from error

    def stop(self) -> None:
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            return
        self._thread.join(timeout=2.0)


def _make_gripper(args: argparse.Namespace) -> AbsoluteTimeContactGripper:
    return AbsoluteTimeContactGripper(
        mode=args.gripper_control_mode,
        close_confirm_frames=args.gripper_close_confirm_frames,
        open_confirm_frames=args.gripper_open_confirm_frames,
        min_close_hold_frames=args.gripper_min_close_hold_frames,
        min_open_hold_frames=args.gripper_min_open_hold_frames,
        close_feedback_threshold=args.gripper_close_feedback_threshold,
        open_feedback_threshold=args.gripper_open_feedback_threshold,
        terminal_open_after_release=args.gripper_terminal_open_after_release,
        contact_detection=args.gripper_contact_detection,
        contact_max_feedback=args.gripper_contact_max_feedback,
        contact_stall_frames=args.gripper_contact_stall_frames,
        contact_motion_tolerance=args.gripper_contact_motion_tolerance,
        contact_min_close_frames=args.gripper_contact_min_close_frames,
        contact_min_drop=args.gripper_contact_min_drop,
    )


def _should_advance_action_tick(
    *,
    execute: bool,
    gap_hold: bool,
    arm_clipped: bool | None,
    gripper_target_reached: bool,
) -> bool:
    if gap_hold:
        return False
    if not execute:
        return True
    # ``arm_clipped`` means the robot driver projected the requested command
    # into its relative safety envelope and sent that projected command.  It
    # is not a send failure.  Holding policy time on such a projection repeats
    # the same out-of-envelope action forever (especially at calibrated hard
    # stops such as joint_1=-100 / joint_2=+100).  Keep the physical clamp, but
    # advance after the successful send and synchronize the controller to the
    # value the driver actually accepted.
    _ = arm_clipped
    return bool(gripper_target_reached)


def run_live(
    args: argparse.Namespace,
    policy,
    log: legacy.JsonlLogger,
    device: torch.device,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
    run_timestamp: str,
) -> None:
    execute = args.mode == "execute"
    if execute and not args.allow_motion:
        raise RuntimeError("execute mode requires --allow-motion")

    robot = legacy.make_robot(args, joint_limits, gripper_stroke_mm)
    robot.connect(calibrate=False)
    normalizer = legacy.make_wrist_color_normalizer(args)
    gripper = _make_gripper(args)
    controller = HPTStyleDecodedJointController(
        policy,
        min_suffix_actions=args.rtc_min_suffix_actions,
        blend_actions=args.rtc_blend_actions,
        replan_interval=args.rtc_replan_interval,
        intrachunk_smoothing=args.rtc_intrachunk_smoothing,
        intrachunk_window=args.rtc_intrachunk_window,
        intrachunk_sigma=args.rtc_intrachunk_sigma,
        max_command_step=args.max_joint_command_step,
        max_command_accel=args.max_joint_command_accel,
        max_command_jerk=args.rtc_max_joint_command_jerk,
        command_deadband=args.rtc_command_deadband,
    )
    worker: BackgroundPredictor | None = None
    next_plan_id = 1
    last_safe_action: np.ndarray | None = None
    starvation_frames = 0
    try:
        camera_reports = legacy.configure_camera_auto_controls(robot)
        for camera_name, camera_report in camera_reports.items():
            log.write(
                {
                    "event": "camera_config",
                    "camera": camera_name,
                    **camera_report,
                    "command_sent": False,
                }
            )
        color_report, warmup_observation = legacy.calibrate_wrist_color_normalizer(
            robot, normalizer
        )
        if color_report is not None:
            log.write(
                {
                    "event": "wrist_color_normalization",
                    **color_report,
                    "command_sent": False,
                }
            )

        warmup_model_observation = _copy_model_observation(
            warmup_observation, normalizer
        )
        warmup_started = time.perf_counter()
        policy.predict_action_chunk(
            legacy.live_observation_to_batch(
                warmup_model_observation, device
            )
        )
        initial_report = copy.deepcopy(policy.last_prediction_report)
        if initial_report is None:
            raise RuntimeError("initial inference produced no report")
        initial_install = controller.install(
            plan_id=0,
            start_tick=0,
            prediction_report=initial_report,
            result_age_ms=0.0,
            max_result_age_ms=args.rtc_max_result_age_ms,
        )
        if not initial_install["accepted"]:
            raise RuntimeError(f"initial RTC plan was rejected: {initial_install}")
        initial_gripper = gripper.observe_prediction_horizon_at(
            0,
            0,
            initial_report["full_control_gripper_commands_pct"],
        )
        current_driver = legacy.observation_to_driver_state(
            warmup_observation, joint_limits, gripper_stroke_mm
        )
        first_safe, first_status = controller.peek(current_driver)
        continuity = legacy.initial_continuity_report(
            warmup_observation,
            torch.from_numpy(first_safe),
            joint_limits,
            args.max_initial_joint_delta_deg,
        )
        controller.cancel_pending()
        log.write(
            {
                "event": "warmup",
                "mode": args.mode,
                "total_step_ms": (time.perf_counter() - warmup_started) * 1e3,
                "initial_continuity": continuity,
                "initial_install": initial_install,
                "initial_controller": first_status,
                "initial_gripper_horizon": initial_gripper,
                "chunk_report": initial_report,
                "command_sent": False,
            }
        )
        if execute:
            held = robot.enable_motion_hold_current()
            log.write(
                {
                    "event": "motion_enable",
                    "held_position": held,
                    "command_sent": True,
                }
            )

        worker = BackgroundPredictor(policy, device)
        worker.start()
        started_run = time.perf_counter()
        for frame_index in range(args.max_steps):
            loop_started = time.perf_counter()
            for completion in worker.drain():
                if completion["event"] == "hptstyle_rtc_replan_completed":
                    install = controller.install(
                        plan_id=completion["plan_id"],
                        start_tick=completion["start_tick"],
                        prediction_report=completion["prediction_report"],
                        result_age_ms=completion["result_age_ms"],
                        max_result_age_ms=args.rtc_max_result_age_ms,
                    )
                    completion["install"] = install
                    if install["accepted"]:
                        completion["gripper_horizon_update"] = (
                            gripper.observe_prediction_horizon_at(
                                completion["plan_id"],
                                completion["start_tick"],
                                completion["prediction_report"][
                                    "full_control_gripper_commands_pct"
                                ],
                            )
                        )
                log.write(completion)
            worker.raise_if_failed()

            observation = robot.get_observation()
            submitted = None
            if controller.should_request(worker.busy()):
                request_tick = controller.action_tick
                controller.note_request(request_tick)
                request = InferenceRequest(
                    plan_id=next_plan_id,
                    start_tick=request_tick,
                    observation=_copy_model_observation(observation, normalizer),
                    submitted_monotonic=time.monotonic(),
                )
                worker.submit(request)
                submitted = {
                    "event": "hptstyle_rtc_replan_submitted",
                    "plan_id": next_plan_id,
                    "start_tick": request_tick,
                    "controller": controller.status(),
                    "command_sent": False,
                }
                log.write(submitted)
                next_plan_id += 1

            driver_state = legacy.observation_to_driver_state(
                observation, joint_limits, gripper_stroke_mm
            )
            safe_action, controller_before = controller.peek(driver_state)
            gap_hold = bool(controller_before["gap_hold"])
            if gap_hold:
                starvation_frames += 1
                if starvation_frames > args.rtc_max_starvation_hold_frames:
                    raise RuntimeError(
                        "HPT-style RTC exceeded bounded queue starvation hold: "
                        f"{starvation_frames} control frames"
                    )
            else:
                starvation_frames = 0
                last_safe_action = safe_action.copy()

            model_action = legacy.action_to_robot_dict(
                torch.from_numpy(safe_action)
            )
            gripper_state = gripper.update_from_prediction_horizons(
                model_action["joint_6.pos"], driver_state[6]
            )
            robot_action = dict(model_action)
            robot_action["joint_6.pos"] = gripper_state["command_pct"]
            sent_action = robot.send_action(robot_action) if execute else None
            arm_clipped = None
            gripper_clipped = None
            if sent_action is not None:
                arm_clipped = any(
                    abs(
                        robot_action[f"joint_{index}.pos"]
                        - sent_action[f"joint_{index}.pos"]
                    )
                    > args.driver_clip_tolerance
                    for index in range(6)
                )
                gripper_clipped = (
                    abs(
                        robot_action["joint_6.pos"]
                        - sent_action["joint_6.pos"]
                    )
                    > args.driver_clip_tolerance
                )
            # Match the Mixed Clean V2 runtime contract: in execute mode the
            # absolute action clock must not advance while a binary gripper
            # transition is still physically in flight.  A close can finish
            # either at the calibrated closed threshold or through the
            # stalled-on-object contact detector above.  An open must reach
            # the calibrated open threshold.  live-dry-run has no commanded
            # actuator feedback, so it advances deterministically.
            action_advanced = _should_advance_action_tick(
                execute=execute,
                gap_hold=gap_hold,
                arm_clipped=arm_clipped,
                gripper_target_reached=bool(gripper_state["target_reached"]),
            )
            advance_report = None
            if action_advanced:
                committed_arm_action = None
                if sent_action is not None:
                    committed_arm_action = np.asarray(
                        [
                            sent_action[f"joint_{index}.pos"]
                            for index in range(6)
                        ],
                        dtype=np.float64,
                    )
                advance_report = controller.advance(committed_arm_action)
                gripper.advance_prediction_tick()
            else:
                controller.cancel_pending()

            image_paths = None
            if args.save_inference_images and submitted is not None:
                corrected_wrist = (
                    normalizer.apply(observation["405"])
                    if normalizer is not None
                    else observation["405"]
                )
                image_paths = legacy.save_inference_images(
                    args.log_dir,
                    run_timestamp,
                    int(submitted["plan_id"]),
                    observation["435if"],
                    corrected_wrist,
                    observation["405"],
                )
            log.write(
                {
                    "mode": args.mode,
                    "frame": int(frame_index),
                    "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": time.perf_counter() - started_run,
                    "total_step_ms": (time.perf_counter() - loop_started) * 1e3,
                    "raw_state": [
                        float(observation[f"joint_{index}.pos"])
                        for index in range(7)
                    ],
                    "driver_state": driver_state.tolist(),
                    "safe_action": safe_action.tolist(),
                    "model_action": model_action,
                    "proposed_action": robot_action,
                    "sent_action": sent_action,
                    "arm_command_was_clipped": arm_clipped,
                    "gripper_command_was_clipped": gripper_clipped,
                    "action_advanced": action_advanced,
                    "action_advance_commit": advance_report,
                    "action_waiting_for_gripper": bool(
                        execute
                        and not gripper_state["target_reached"]
                    ),
                    "rtc_starvation_hold": gap_hold,
                    "rtc_starvation_hold_frames": int(starvation_frames),
                    "rtc_controller_before": controller_before,
                    "rtc_controller_after": controller.status(),
                    "rtc_worker_busy": worker.busy(),
                    "gripper_state_machine": gripper_state,
                    "inference_images": image_paths,
                    "last_safe_action": (
                        None if last_safe_action is None else last_safe_action.tolist()
                    ),
                    "command_sent": execute,
                }
            )
            precise_sleep(
                1.0 / float(args.rtc_control_hz)
                - (time.perf_counter() - loop_started)
            )
    finally:
        if worker is not None:
            worker.stop()
            for completion in worker.drain():
                completion["event"] = (
                    "hptstyle_rtc_replan_completed_after_control_loop"
                    if completion.get("event")
                    == "hptstyle_rtc_replan_completed"
                    else completion.get("event")
                )
                log.write(completion)
        robot.disconnect()


def run_offline_pair(
    args: argparse.Namespace,
    policy,
    log: legacy.JsonlLogger,
    device: torch.device,
) -> None:
    if args.offline_npz is None:
        raise ValueError("offline mode requires --offline-npz")
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
            f"offline NPZ needs frame {request_tick}, contains {count}"
        )

    first_batch = legacy.make_policy_batch(
        arrays[keys[0]][0], arrays[keys[1]][0], arrays[keys[2]][0], device
    )
    policy.predict_action_chunk(first_batch)
    first_report = copy.deepcopy(policy.last_prediction_report)
    second_batch = legacy.make_policy_batch(
        arrays[keys[0]][request_tick],
        arrays[keys[1]][request_tick],
        arrays[keys[2]][request_tick],
        device,
    )
    policy.predict_action_chunk(second_batch)
    second_report = copy.deepcopy(policy.last_prediction_report)
    if first_report is None or second_report is None:
        raise RuntimeError("offline prediction produced no report")

    controller = HPTStyleDecodedJointController(
        policy,
        min_suffix_actions=args.rtc_min_suffix_actions,
        blend_actions=args.rtc_blend_actions,
        replan_interval=args.rtc_replan_interval,
        intrachunk_smoothing=args.rtc_intrachunk_smoothing,
        intrachunk_window=args.rtc_intrachunk_window,
        intrachunk_sigma=args.rtc_intrachunk_sigma,
        max_command_step=args.max_joint_command_step,
        max_command_accel=args.max_joint_command_accel,
        max_command_jerk=args.rtc_max_joint_command_jerk,
        command_deadband=args.rtc_command_deadband,
    )
    initial = controller.install(
        plan_id=0,
        start_tick=0,
        prediction_report=first_report,
        result_age_ms=0.0,
        max_result_age_ms=args.rtc_max_result_age_ms,
    )
    controller.action_tick = install_tick
    replacement = controller.install(
        plan_id=1,
        start_tick=request_tick,
        prediction_report=second_report,
        result_age_ms=0.0,
        max_result_age_ms=args.rtc_max_result_age_ms,
    )
    if not initial["accepted"] or not replacement["accepted"]:
        raise RuntimeError(
            f"offline RTC plan rejected: initial={initial}, replacement={replacement}"
        )
    log.write(
        {
            "event": "pi05_hptstyle_decoded_joint_rtc_offline_pair",
            "request_tick": request_tick,
            "install_tick": install_tick,
            "initial_install": initial,
            "replacement_install": replacement,
            "first_chunk_report": first_report,
            "second_chunk_report": second_report,
            "command_sent": False,
        }
    )


def parse_args() -> argparse.Namespace:
    rtc_parser = argparse.ArgumentParser(add_help=False)
    rtc_parser.add_argument("--rtc-control-hz", type=float, default=CONTROL_HZ)
    rtc_parser.add_argument("--rtc-replan-interval", type=int, default=10)
    rtc_parser.add_argument("--rtc-blend-actions", type=int, default=8)
    rtc_parser.add_argument("--rtc-min-suffix-actions", type=int, default=8)
    rtc_parser.add_argument(
        "--rtc-intrachunk-smoothing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "smooth only the decoded 6-D arm trajectory before RTC overlap; "
            "binary gripper predictions are preserved exactly"
        ),
    )
    rtc_parser.add_argument("--rtc-intrachunk-window", type=int, default=7)
    rtc_parser.add_argument("--rtc-intrachunk-sigma", type=float, default=1.5)
    rtc_parser.add_argument(
        "--rtc-max-joint-command-jerk",
        type=float,
        default=0.025,
        help=(
            "maximum change of command acceleration per successful tick in "
            "PIPER driver-coordinate units"
        ),
    )
    rtc_parser.add_argument(
        "--rtc-command-deadband",
        type=float,
        default=0.02,
        help="driver-coordinate target error treated as zero by the causal limiter",
    )
    rtc_parser.add_argument("--rtc-max-result-age-ms", type=float, default=1500.0)
    rtc_parser.add_argument(
        "--rtc-max-starvation-hold-frames", type=int, default=90
    )
    rtc_parser.add_argument("--rtc-offline-request-tick", type=int, default=3)
    rtc_parser.add_argument(
        "--rtc-offline-inference-delay-ticks", type=int, default=6
    )
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
    # Override the legacy defaults for this entrypoint while retaining the
    # same public option names.
    rtc_parser.add_argument(
        "--gripper-min-open-hold-frames", type=int, default=60
    )
    rtc_parser.add_argument(
        "--wrist-color-normalization",
        choices=("off", "startup"),
        default="startup",
    )
    rtc_args, remaining = rtc_parser.parse_known_args()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = base.parse_args()
    finally:
        sys.argv = old_argv
    for key, value in vars(rtc_args).items():
        setattr(args, key, value)
    if args.list_checkpoint_profiles:
        return args
    if int(args.query_frequency) != MODEL_HORIZON:
        raise SystemExit("HPT-style decoded-joint RTC requires --query-frequency 30")
    if args.temporal_ensemble:
        raise SystemExit("do not combine HPT-style RTC with --temporal-ensemble")
    if args.chunk_smoothing:
        raise SystemExit(
            "use --rtc-intrachunk-smoothing, not legacy --chunk-smoothing, "
            "with HPT-style RTC"
        )
    if args.gripper_prediction_source != "full-horizon":
        raise SystemExit("HPT-style RTC requires --gripper-prediction-source full-horizon")
    if not 1 <= args.rtc_replan_interval < MODEL_HORIZON:
        raise SystemExit("--rtc-replan-interval must be in [1,29]")
    if not 1 <= args.rtc_blend_actions < MODEL_HORIZON:
        raise SystemExit("--rtc-blend-actions must be in [1,29]")
    if not 1 <= args.rtc_min_suffix_actions <= MODEL_HORIZON:
        raise SystemExit("--rtc-min-suffix-actions must be in [1,30]")
    if (
        args.rtc_intrachunk_window <= 0
        or args.rtc_intrachunk_window > MODEL_HORIZON
        or args.rtc_intrachunk_window % 2 == 0
    ):
        raise SystemExit("--rtc-intrachunk-window must be odd and in [1,29]")
    if args.rtc_intrachunk_sigma <= 0:
        raise SystemExit("--rtc-intrachunk-sigma must be positive")
    if not 0 < args.rtc_max_joint_command_jerk <= args.max_joint_command_accel:
        raise SystemExit(
            "--rtc-max-joint-command-jerk must be in "
            "(0,--max-joint-command-accel]"
        )
    if args.rtc_command_deadband < 0:
        raise SystemExit("--rtc-command-deadband must be non-negative")
    if args.rtc_max_result_age_ms <= 0 or args.rtc_control_hz <= 0:
        raise SystemExit("RTC time limits must be positive")
    if args.rtc_max_starvation_hold_frames <= 0:
        raise SystemExit("--rtc-max-starvation-hold-frames must be positive")
    if args.driver_clip_tolerance < 0:
        raise SystemExit("--driver-clip-tolerance must be non-negative")
    if args.gripper_min_open_hold_frames < 0:
        raise SystemExit("--gripper-min-open-hold-frames must be non-negative")
    if not 0 <= args.gripper_contact_max_feedback <= 100:
        raise SystemExit("--gripper-contact-max-feedback must be in [0,100]")
    if args.gripper_contact_stall_frames <= 1:
        raise SystemExit("--gripper-contact-stall-frames must exceed one")
    if args.gripper_contact_motion_tolerance < 0:
        raise SystemExit("--gripper-contact-motion-tolerance must be non-negative")
    if args.gripper_contact_min_close_frames < 0:
        raise SystemExit("--gripper-contact-min-close-frames must be non-negative")
    if not 0 <= args.gripper_contact_min_drop <= 100:
        raise SystemExit("--gripper-contact-min-drop must be in [0,100]")
    if not 0 <= args.rtc_offline_request_tick < MODEL_HORIZON:
        raise SystemExit("--rtc-offline-request-tick must be in [0,29]")
    if not 0 <= args.rtc_offline_inference_delay_ticks < MODEL_HORIZON:
        raise SystemExit("--rtc-offline-inference-delay-ticks must be in [0,29]")
    if (
        args.rtc_offline_request_tick
        + args.rtc_offline_inference_delay_ticks
        >= MODEL_HORIZON
    ):
        raise SystemExit("offline RTC install tick must remain inside 30 actions")
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
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

    policy = base.EgoVersePI05PiperV3V4JointPolicy(
        args, joint_limits, gripper_stroke_mm
    )
    if args.mode == "checkpoint":
        print(
            "PI05_PIPER_V3_V4_HPTSTYLE_RTC_CHECKPOINT_OK "
            f"profile={args.checkpoint_profile} sha256={actual_sha256} "
            "robot_constructed=false camera_opened=false can_opened=false"
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = legacy.JsonlLogger(
        args.log_dir
        / f"pi05_piper_v3_v4_hptstyle_rtc_{args.mode}_{timestamp}.jsonl"
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
                "rtc_mode": "post_decode_absolute_time_cosine_overlap",
                "rtc_space": "safe_physical_6d_joint_radians",
                "flow_sampler_rtc_guidance": False,
                "rtc_replan_interval": args.rtc_replan_interval,
                "rtc_blend_actions": args.rtc_blend_actions,
                "rtc_min_suffix_actions": args.rtc_min_suffix_actions,
                "rtc_max_result_age_ms": args.rtc_max_result_age_ms,
                "rtc_intrachunk_smoothing": args.rtc_intrachunk_smoothing,
                "rtc_intrachunk_window": args.rtc_intrachunk_window,
                "rtc_intrachunk_sigma": args.rtc_intrachunk_sigma,
                "rtc_causal_filter_space": "piper_driver_coordinates",
                "rtc_max_joint_command_step": args.max_joint_command_step,
                "rtc_max_joint_command_accel": args.max_joint_command_accel,
                "rtc_max_joint_command_jerk": (
                    args.rtc_max_joint_command_jerk
                ),
                "rtc_command_deadband": args.rtc_command_deadband,
                "successful_action_tick_clock": True,
                "inference_thread": True,
                "wrist_color_normalization": args.wrist_color_normalization,
                "gripper_absolute_time_binary": True,
                "gripper_contact_detection": args.gripper_contact_detection,
                "gripper_min_close_hold_frames": (
                    args.gripper_min_close_hold_frames
                ),
                "gripper_min_open_hold_frames": (
                    args.gripper_min_open_hold_frames
                ),
                "command_sent": False,
            }
        )
        if args.mode == "offline":
            run_offline_pair(args, policy, log, device)
        else:
            run_live(
                args,
                policy,
                log,
                device,
                joint_limits,
                gripper_stroke_mm,
                timestamp,
            )
    finally:
        log.close()
        logger.info("wrote HPT-style pi0.5 RTC audit log to %s", log.path)


if __name__ == "__main__":
    main()
