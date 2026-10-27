#!/usr/bin/env python
"""CPU structural tests for the additive pi0.5 HPT-style RTC entrypoint."""

from __future__ import annotations

import numpy as np

from deploy_egoverse_pi05_piper_v3_v4_hptstyle_rtc_joint import (
    AbsoluteTimeContactGripper,
    HPTStyleDecodedJointController,
    _should_advance_action_tick,
)


class DummyPolicy:
    def __init__(self) -> None:
        self.low = np.full(6, -2.0, dtype=np.float64)
        self.high = np.full(6, 2.0, dtype=np.float64)

    def _joint_rad_to_driver(self, joint_rad: np.ndarray) -> np.ndarray:
        values = np.asarray(joint_rad, dtype=np.float64)
        return ((values - self.low) / (self.high - self.low) * 200.0 - 100.0)

    def _filter_driver_action(self, action, current, velocity):
        del current, velocity
        return np.asarray(action, dtype=np.float32).copy(), np.zeros(6)


def make_report(offset: float, gripper_open: float) -> dict:
    ticks = np.arange(30, dtype=np.float32)[:, None]
    joints = np.repeat(offset + 0.01 * ticks, 6, axis=1)
    physical = np.concatenate(
        (
            joints,
            np.full((30, 1), gripper_open, dtype=np.float32),
        ),
        axis=1,
    )
    return {
        "selected_physical_actions": physical.tolist(),
        "full_control_gripper_commands_pct": (
            physical[:, 6] * 100.0
        ).tolist(),
    }


def make_controller(
    *,
    smoothing: bool = True,
    max_step: float = 0.5,
    max_accel: float = 0.1,
    max_jerk: float = 0.025,
) -> HPTStyleDecodedJointController:
    return HPTStyleDecodedJointController(
        DummyPolicy(),
        min_suffix_actions=8,
        blend_actions=8,
        replan_interval=10,
        intrachunk_smoothing=smoothing,
        intrachunk_window=7,
        intrachunk_sigma=1.5,
        max_command_step=max_step,
        max_command_accel=max_accel,
        max_command_jerk=max_jerk,
        command_deadband=0.02,
    )


def test_decoded_joint_overlap() -> None:
    controller = make_controller()
    initial = controller.install(
        plan_id=0,
        start_tick=0,
        prediction_report=make_report(0.0, 1.0),
        result_age_ms=0.0,
        max_result_age_ms=1500.0,
    )
    assert initial["accepted"]
    controller.action_tick = 13
    replacement = controller.install(
        plan_id=1,
        start_tick=10,
        prediction_report=make_report(0.30, 0.0),
        result_age_ms=400.0,
        max_result_age_ms=1500.0,
    )
    assert replacement["accepted"]
    assert replacement["skipped_prefix_actions"] == 3
    assert replacement["rtc_blended_actions"] == 8
    assert replacement["rtc_intrachunk_smoothing"]
    assert (
        replacement["intrachunk_smoothed_adjacent_joint_rms_rad_mean"]
        < replacement["intrachunk_raw_adjacent_joint_rms_rad_mean"]
    )
    assert (
        replacement["switch_joint_rms_after_rad"]
        < replacement["switch_joint_rms_before_rad"]
    )
    assert (
        replacement["switch_driver_rms_after"]
        < replacement["switch_driver_rms_before"]
    )
    # Joint RTC must never interpolate the new chunk's binary gripper.
    assert np.all(controller.active_plan.physical[:, 6] == 0.0)
    assert np.all(controller.active_plan.driver[:, 6] == 0.0)


def test_stale_forecast_rejected() -> None:
    controller = make_controller()
    controller.action_tick = 25
    rejected = controller.install(
        plan_id=1,
        start_tick=0,
        prediction_report=make_report(0.0, 1.0),
        result_age_ms=300.0,
        max_result_age_ms=1500.0,
    )
    assert not rejected["accepted"]
    assert rejected["reason"] == "result_has_too_few_time_aligned_suffix_actions"


def test_binary_gripper_survives_intrachunk_smoothing() -> None:
    report = make_report(0.0, 1.0)
    commands = np.asarray(report["selected_physical_actions"], dtype=np.float32)
    commands[10:20, 6] = 0.0
    report["selected_physical_actions"] = commands.tolist()
    controller = make_controller()
    installed = controller.install(
        plan_id=0,
        start_tick=0,
        prediction_report=report,
        result_age_ms=0.0,
        max_result_age_ms=1500.0,
    )
    assert installed["accepted"]
    assert np.array_equal(controller.active_plan.physical[:, 6], commands[:, 6])
    assert set(controller.active_plan.driver[:, 6].tolist()) == {0.0, 100.0}


def test_causal_velocity_acceleration_and_jerk_limits() -> None:
    controller = make_controller(max_step=0.5, max_accel=0.1, max_jerk=0.025)
    current = np.zeros(7, dtype=np.float64)
    outputs = []
    velocities = []
    accelerations = []
    jerks = []
    for tick in range(30):
        raw = np.zeros(7, dtype=np.float64)
        raw[:6] = 20.0 if tick < 15 else -20.0
        raw[6] = 100.0 if tick % 2 == 0 else 0.0
        safe, command, velocity, acceleration, report = (
            controller._causal_filter(raw, current)
        )
        outputs.append(safe.copy())
        velocities.append(velocity.copy())
        accelerations.append(acceleration.copy())
        jerks.append(np.asarray(report["committed_candidate_jerk"]))
        # The discrete gripper must bypass the numerical arm filter.
        assert safe[6] == raw[6]
        controller.pending_command = command
        controller.pending_velocity = velocity
        controller.pending_acceleration = acceleration
        controller.advance()
        current[:6] = safe[:6]
    assert np.max(np.abs(velocities)) <= 0.5 + 1e-9
    assert np.max(np.abs(accelerations)) <= 0.1 + 1e-9
    assert np.max(np.abs(jerks)) <= 0.025 + 1e-9
    # Reversal is gradual: a discontinuous target cannot flip the command
    # velocity sign on the same control tick.
    assert velocities[14][0] > 0.0
    assert velocities[15][0] > 0.0
    assert outputs[15][6] == 0.0


def test_cancel_does_not_commit_filter_state() -> None:
    controller = make_controller()
    raw = np.asarray([10.0] * 6 + [100.0], dtype=np.float64)
    current = np.zeros(7, dtype=np.float64)
    first = controller._causal_filter(raw, current)
    controller.pending_command = first[1]
    controller.pending_velocity = first[2]
    controller.pending_acceleration = first[3]
    controller.cancel_pending()
    second = controller._causal_filter(raw, current)
    assert np.allclose(first[0], second[0])
    assert np.allclose(controller.last_velocity, 0.0)
    assert np.allclose(controller.last_acceleration, 0.0)


def test_driver_projection_commits_sent_action_and_resets_derivatives() -> None:
    controller = make_controller(max_step=1.5, max_accel=1.5, max_jerk=0.5)
    raw = np.asarray([20.0] * 6 + [100.0], dtype=np.float64)
    current = np.zeros(7, dtype=np.float64)
    result = controller._causal_filter(raw, current)
    controller.pending_command = result[1]
    controller.pending_velocity = result[2]
    controller.pending_acceleration = result[3]
    sent = result[1].copy()
    sent[0] -= 0.25
    report = controller.advance(sent)
    assert controller.action_tick == 1
    assert report["driver_projection_committed"]
    assert np.allclose(controller.last_command, sent)
    assert np.allclose(controller.last_velocity, 0.0)
    assert np.allclose(controller.last_acceleration, 0.0)


def make_gripper(*, min_open_hold_frames: int = 0) -> AbsoluteTimeContactGripper:
    return AbsoluteTimeContactGripper(
        mode="latched",
        close_confirm_frames=1,
        open_confirm_frames=1,
        min_close_hold_frames=0,
        min_open_hold_frames=min_open_hold_frames,
        close_feedback_threshold=10.0,
        open_feedback_threshold=90.0,
        terminal_open_after_release=False,
        contact_detection=True,
        contact_max_feedback=70.0,
        contact_stall_frames=5,
        contact_motion_tolerance=0.5,
        contact_min_close_frames=3,
        contact_min_drop=10.0,
    )


def test_contact_completion() -> None:
    gripper = make_gripper()
    reports = [
        gripper.update(0.0, feedback)
        for feedback in (97.0, 90.0, 75.0, 60.0, 45.0, 40.0, 40.0, 40.0, 40.0, 40.0)
    ]
    assert reports[-1]["target_reached"]
    assert reports[-1]["contact_reached"]
    assert reports[-1]["contact_completion_source"] == "stalled_on_object"


def test_minimum_open_hold() -> None:
    gripper = make_gripper(min_open_hold_frames=3)
    opened = gripper.update(100.0, 100.0)
    assert opened["state"] == "open"
    held = [gripper.update(0.0, 100.0) for _ in range(3)]
    assert all(report["state"] == "open" for report in held)
    assert all(
        report["close_request_ignored_during_min_open_hold"]
        for report in held
    )
    closed = gripper.update(0.0, 100.0)
    assert closed["state"] == "close"


def test_action_tick_waits_for_physical_gripper() -> None:
    assert _should_advance_action_tick(
        execute=False,
        gap_hold=False,
        arm_clipped=None,
        gripper_target_reached=False,
    )
    assert not _should_advance_action_tick(
        execute=True,
        gap_hold=False,
        arm_clipped=False,
        gripper_target_reached=False,
    )
    assert _should_advance_action_tick(
        execute=True,
        gap_hold=False,
        arm_clipped=False,
        gripper_target_reached=True,
    )
    assert _should_advance_action_tick(
        execute=True,
        gap_hold=False,
        arm_clipped=True,
        gripper_target_reached=True,
    )


def main() -> None:
    test_decoded_joint_overlap()
    test_stale_forecast_rejected()
    test_binary_gripper_survives_intrachunk_smoothing()
    test_causal_velocity_acceleration_and_jerk_limits()
    test_cancel_does_not_commit_filter_state()
    test_driver_projection_commits_sent_action_and_resets_derivatives()
    test_contact_completion()
    test_minimum_open_hold()
    test_action_tick_waits_for_physical_gripper()
    print("PI05_PIPER_V3_V4_HPTSTYLE_RTC_UNIT_TESTS_OK")


if __name__ == "__main__":
    main()
