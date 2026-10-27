#!/usr/bin/env python
"""CPU structural tests for single-frame HPT Mixed decoded-joint RTC."""

from __future__ import annotations

import numpy as np

from deploy_egoverse_hpt_joint_rtc import (
    HPTAlignedGripperCompatibility,
    HPTDecodedJointRTCController,
)


class DummyPolicy:
    def __init__(self) -> None:
        self._joint_low_rad = np.full(6, -2.0, dtype=np.float64)
        self._joint_high_rad = np.full(6, 2.0, dtype=np.float64)

    def _joint_rad_to_driver(self, joint_rad: np.ndarray) -> np.ndarray:
        span = self._joint_high_rad - self._joint_low_rad
        return (np.asarray(joint_rad) - self._joint_low_rad) / span * 200.0 - 100.0

    def _filter_driver_action(self, action, current, velocity):
        return np.asarray(action, dtype=np.float32).copy(), np.zeros(6)


def make_plan(offset: float, gripper: float) -> tuple[np.ndarray, dict]:
    ticks = np.arange(30, dtype=np.float32)[:, None]
    joints = np.repeat(0.01 * ticks + offset, 6, axis=1)
    physical = np.concatenate(
        (joints, np.full((30, 1), gripper, dtype=np.float32)), axis=1
    )
    policy = DummyPolicy()
    driver = np.empty_like(physical)
    driver[:, :6] = np.stack(
        [policy._joint_rad_to_driver(row) for row in joints], axis=0
    )
    driver[:, 6] = gripper * 100.0
    return driver, {"selected_physical_actions": physical.tolist()}


def test_decoded_joint_rtc_blend() -> None:
    policy = DummyPolicy()
    controller = HPTDecodedJointRTCController(
        policy, min_suffix_actions=5, transition_blend_actions=8
    )
    old_actions, old_report = make_plan(0.0, 1.0)
    initial = controller.install(
        plan_id=0,
        start_tick=0,
        actions=old_actions,
        prediction_report=old_report,
        result_age_ms=0.0,
        max_result_age_ms=500.0,
    )
    assert initial["accepted"]
    assert not controller.should_request(10)
    controller.action_tick = 10
    assert controller.should_request(10)

    new_actions, new_report = make_plan(0.30, 0.0)
    controller.action_tick = 13
    replacement = controller.install(
        plan_id=1,
        start_tick=10,
        actions=new_actions,
        prediction_report=new_report,
        result_age_ms=100.0,
        max_result_age_ms=500.0,
    )
    assert replacement["accepted"]
    assert replacement["skipped_prefix_actions"] == 3
    assert replacement["rtc_blended_actions"] == 8
    assert replacement["switch_joint_rms_after_rad"] < replacement[
        "switch_joint_rms_before_rad"
    ]
    assert replacement["switch_driver_rms_after"] < replacement[
        "switch_driver_rms_before"
    ]
    # RTC never blends the binary gripper channel.
    assert np.all(controller.active_plan.actions[:, 6] == 0.0)


def test_rejects_expired_forecast() -> None:
    policy = DummyPolicy()
    controller = HPTDecodedJointRTCController(
        policy, min_suffix_actions=8, transition_blend_actions=4
    )
    actions, report = make_plan(0.0, 1.0)
    controller.action_tick = 25
    rejected = controller.install(
        plan_id=1,
        start_tick=0,
        actions=actions,
        prediction_report=report,
        result_age_ms=10.0,
        max_result_age_ms=500.0,
    )
    assert not rejected["accepted"]
    assert rejected["reason"] == "result_has_too_few_time_aligned_suffix_actions"


def make_gripper() -> HPTAlignedGripperCompatibility:
    return HPTAlignedGripperCompatibility(
        mode="latched",
        close_confirm_frames=1,
        open_confirm_frames=1,
        min_close_hold_frames=0,
        min_open_hold_frames=0,
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


def test_gripper_contact_completion() -> None:
    gripper = make_gripper()
    reports = []
    for feedback in (97.0, 90.0, 75.0, 60.0, 45.0, 40.0, 40.0, 40.0, 40.0, 40.0):
        reports.append(gripper.update(0.0, feedback))
    assert not reports[4]["target_reached"]
    assert reports[-1]["target_reached"]
    assert reports[-1]["contact_reached"]
    assert reports[-1]["contact_completion_source"] == "stalled_on_object"
    # Contact remains latched during this close episode even if compression
    # causes a small feedback rebound.
    rebound = gripper.update(0.0, 41.0)
    assert rebound["target_reached"]
    assert rebound["contact_reached"]
    opened = gripper.update(100.0, 95.0)
    assert opened["state"] == "open"
    assert not opened["contact_reached"]


def test_open_plateau_is_not_false_contact() -> None:
    gripper = make_gripper()
    reports = [gripper.update(0.0, 97.0) for _ in range(20)]
    assert not any(report["contact_reached"] for report in reports)
    assert not reports[-1]["target_reached"]


def test_minimum_open_hold_blocks_early_reclose() -> None:
    gripper = make_gripper()
    gripper.min_open_hold_frames = 3
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


def main() -> None:
    test_decoded_joint_rtc_blend()
    test_rejects_expired_forecast()
    test_gripper_contact_completion()
    test_open_plateau_is_not_false_contact()
    test_minimum_open_hold_blocks_early_reclose()
    print("HPT_MIXED_DECODED_JOINT_RTC_UNIT_TESTS_OK")


if __name__ == "__main__":
    main()
