from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from lerobot.policies.egoverse_hpt.modeling_egoverse_hpt_joint import (
    EgoVerseHPTJointPolicy,
    absolute_time_joint_ensemble,
    model_indices_for_control_ticks,
    prepare_rgb_chw_image,
    smooth_joint_action_chunk,
)

from deploy_egoverse_hpt_joint import (
    CHECKPOINT_PROFILES,
    IndependentGripperStateMachine,
    action_to_robot_dict,
    full_gripper_horizon_from_report,
    initial_continuity_report,
    observation_to_driver_state,
    split_relative_target_limits,
)


def make_gripper_controller(**overrides):
    settings = {
        "mode": "latched",
        "close_confirm_frames": 2,
        "open_confirm_frames": 3,
        "min_close_hold_frames": 2,
        "close_feedback_threshold": 10.0,
        "open_feedback_threshold": 90.0,
    }
    settings.update(overrides)
    return IndependentGripperStateMachine(**settings)


def test_gripper_state_machine_debounces_and_latches_close():
    controller = make_gripper_controller()
    first = controller.update(0.0, 75.0)
    assert first["state"] == "hold"
    assert first["command_pct"] == 75.0
    second = controller.update(0.0, 75.0)
    assert second["state"] == "close"
    assert second["transition"] == "hold->close"
    assert second["command_pct"] == 0.0


def test_gripper_close_latch_ignores_short_open_prediction():
    controller = make_gripper_controller()
    controller.update(0.0, 100.0)
    controller.update(0.0, 100.0)
    assert controller.update(100.0, 40.0)["state"] == "close"
    assert controller.update(100.0, 25.0)["state"] == "close"
    released = controller.update(100.0, 5.0)
    assert released["state"] == "open"
    assert released["transition"] == "close->open"
    assert released["command_pct"] == 100.0


def test_gripper_feedback_reports_completion_without_blocking_transition():
    controller = make_gripper_controller(mode="direct")
    closing = controller.update(0.0, 35.0)
    assert closing["state"] == "close"
    assert not closing["target_reached"]
    closed = controller.update(0.0, 8.0)
    assert closed["target_reached"]
    opening = controller.update(100.0, 8.0)
    assert opening["state"] == "open"
    assert not opening["target_reached"]
    opened = controller.update(100.0, 95.0)
    assert opened["target_reached"]


def test_full_gripper_horizon_preserves_late_close_across_arm_replan():
    controller = make_gripper_controller(open_confirm_frames=1)
    first_horizon = np.full(30, 100.0)
    first_horizon[12:18] = 0.0
    controller.observe_prediction_horizon(1, first_horizon)
    reports = []
    for tick in range(14):
        if tick == 10:
            # The new arm chunk restarts with an open prefix.  It must not erase
            # the old chunk's correctly time-aligned close at absolute tick 12.
            controller.observe_prediction_horizon(2, np.full(30, 100.0))
        reports.append(
            controller.update_from_prediction_horizons(100.0, feedback_pct=100.0)
        )
        controller.advance_prediction_tick()
    assert reports[11]["state"] == "open"
    assert reports[12]["model_request"] == "close"
    assert reports[12]["state"] == "open"
    assert reports[13]["state"] == "close"
    assert reports[13]["command_pct"] == 0.0
    assert [vote["chunk_id"] for vote in reports[13]["horizon_votes"]] == [1, 2]
    assert reports[13]["selected_prediction_chunk_id"] == 1


def test_oldest_active_gripper_horizon_allows_scheduled_reopen():
    controller = make_gripper_controller(
        close_confirm_frames=1,
        open_confirm_frames=2,
        min_close_hold_frames=0,
    )
    committed = np.full(30, 0.0)
    committed[4:] = 100.0
    controller.observe_prediction_horizon(1, committed)
    reports = []
    for tick in range(7):
        if tick == 2:
            # A newer plan restarts with a close prefix.  The committed older
            # plan must still reopen at its absolute offsets 4 and 5.
            controller.observe_prediction_horizon(2, np.zeros(30))
        reports.append(
            controller.update_from_prediction_horizons(0.0, feedback_pct=20.0)
        )
        controller.advance_prediction_tick()
    assert reports[0]["state"] == "close"
    assert reports[4]["model_request"] == "open"
    assert reports[4]["state"] == "close"
    assert reports[5]["state"] == "open"
    assert reports[5]["command_pct"] == 100.0
    assert reports[5]["selected_prediction_chunk_id"] == 1
    assert [vote["chunk_id"] for vote in reports[5]["horizon_votes"]] == [1, 2]


def test_initial_open_does_not_prevent_later_grasp_close():
    controller = make_gripper_controller(
        close_confirm_frames=1,
        open_confirm_frames=1,
        min_close_hold_frames=0,
    )
    initial_open = controller.update(100.0, 100.0)
    assert initial_open["state"] == "open"
    assert not initial_open["has_closed_once"]
    assert not initial_open["release_latched"]

    grasp = controller.update(0.0, 100.0)
    assert grasp["state"] == "close"
    assert grasp["has_closed_once"]
    assert not grasp["release_latched"]


def test_release_terminal_open_latch_ignores_later_close_predictions():
    controller = make_gripper_controller(
        close_confirm_frames=1,
        open_confirm_frames=1,
        min_close_hold_frames=0,
    )
    controller.update(100.0, 100.0)
    controller.update(0.0, 100.0)
    released = controller.update(100.0, 20.0)
    assert released["transition"] == "close->open"
    assert released["release_latched"]
    assert released["command_pct"] == 100.0

    erroneous_reclose = controller.update(0.0, 80.0)
    assert erroneous_reclose["model_request"] == "close"
    assert erroneous_reclose["effective_request"] == "open"
    assert erroneous_reclose["state"] == "open"
    assert erroneous_reclose["command_pct"] == 100.0
    assert erroneous_reclose["release_latched"]
    assert erroneous_reclose["close_request_ignored_after_release"]


def test_full_gripper_horizon_report_requires_exactly_30_binary_steps():
    commands = [100.0] * 30
    output = full_gripper_horizon_from_report(
        {"full_control_gripper_commands_pct": commands}
    )
    np.testing.assert_array_equal(output, commands)
    with pytest.raises(RuntimeError, match="exactly 30"):
        full_gripper_horizon_from_report(
            {"full_control_gripper_commands_pct": commands[:-1]}
        )
    bad = commands.copy()
    bad[7] = 50.0
    with pytest.raises(RuntimeError, match="binary"):
        full_gripper_horizon_from_report(
            {"full_control_gripper_commands_pct": bad}
        )


def test_qf10_maps_raw_ticks_to_resampled_horizon():
    assert model_indices_for_control_ticks(100, 30, 10) == [
        0,
        3,
        7,
        10,
        14,
        17,
        20,
        24,
        27,
        31,
    ]


def test_qf10_preserves_native_30_step_v3_indices():
    assert model_indices_for_control_ticks(30, 30, 10) == list(range(10))


def test_v3_profile_binds_native_horizon_and_audited_hash():
    profile = CHECKPOINT_PROFILES["hpt-shared-v3-15k"]
    assert profile["model_action_horizon"] == 30
    assert profile["sha256"] == (
        "04acf3339fe53b22dd59a491e4c6ab5f5e325689c217a0de646c651db7fb9de2"
    )


def test_rgb_preparation_accepts_uint8_hwc():
    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    output = prepare_rgb_chw_image(
        image,
        expected_height=480,
        expected_width=640,
        resize_mismatch=False,
        name="front",
    )
    assert output.shape == (3, 480, 640)
    assert output.dtype == torch.float32
    assert float(output.min()) == 1.0


def test_action_contract_rejects_fractional_gripper():
    with pytest.raises(RuntimeError, match="not binary"):
        action_to_robot_dict(torch.tensor([0, 0, 0, 0, 0, 0, 50], dtype=torch.float32))


def test_feedback_conversion_matches_calibrated_driver_space():
    limits = [(-100.0, 100.0)] * 6
    observation = {f"joint_{index}.pos": 0.0 for index in range(6)}
    observation["joint_6.pos"] = 50_000.0
    state = observation_to_driver_state(observation, limits, 100.0)
    np.testing.assert_allclose(state[:6], 0.0)
    assert state[6] == 50.0


def test_relative_target_limits_keep_arm_and_gripper_independent():
    limits = split_relative_target_limits(1.5, 100.0)
    assert limits == {
        "joint_0": 1.5,
        "joint_1": 1.5,
        "joint_2": 1.5,
        "joint_3": 1.5,
        "joint_4": 1.5,
        "joint_5": 1.5,
        "joint_6": 100.0,
    }


def test_initial_continuity_is_measured_in_physical_degrees():
    observation = {f"joint_{index}.pos": 0.0 for index in range(6)}
    limits = [(-100.0, 100.0)] * 6
    report = initial_continuity_report(
        observation,
        torch.tensor([0.5, 0, 0, 0, 0, 0, 100], dtype=torch.float32),
        limits,
        max_delta_deg=1.0,
    )
    assert report["max_abs_joint_delta_deg"] == pytest.approx(0.5)
    with pytest.raises(RuntimeError, match="continuity check failed"):
        initial_continuity_report(
            observation,
            torch.tensor([2, 0, 0, 0, 0, 0, 100], dtype=torch.float32),
            limits,
            max_delta_deg=1.0,
        )


def test_joint_trajectory_filter_limits_velocity_and_acceleration():
    policy = object.__new__(EgoVerseHPTJointPolicy)
    policy.config = SimpleNamespace(
        max_joint_command_step=0.5,
        max_joint_command_accel=0.1,
    )
    raw = np.zeros((10, 7), dtype=np.float32)
    raw[:, :6] = 50.0
    raw[:, 6] = 100.0
    current = np.zeros(6, dtype=np.float64)
    velocity = np.zeros(6, dtype=np.float64)
    output = []
    for action in raw:
        safe, velocity = policy._filter_driver_action(action, current, velocity)
        output.append(safe)
        current = safe[:6].astype(np.float64)
    output = np.asarray(output)
    increments = np.diff(
        np.concatenate([np.zeros((1, 6)), output[:, :6]], axis=0), axis=0
    )
    accelerations = np.diff(
        np.concatenate([np.zeros((1, 6)), increments], axis=0), axis=0
    )
    assert np.max(np.abs(increments)) <= 0.5 + 1e-6
    assert np.max(np.abs(accelerations)) <= 0.1 + 1e-6
    np.testing.assert_allclose(output[:5, 0], [0.1, 0.3, 0.6, 1.0, 1.5])


def test_joint_trajectory_filter_carries_velocity_across_replans():
    policy = object.__new__(EgoVerseHPTJointPolicy)
    policy.config = SimpleNamespace(
        max_joint_command_step=0.5,
        max_joint_command_accel=0.1,
    )
    raw = np.zeros((3, 7), dtype=np.float32)
    raw[:, :6] = -50.0
    current = np.zeros(6, dtype=np.float64)
    velocity = np.full(6, 0.5, dtype=np.float64)
    output = []
    for action in raw:
        safe, velocity = policy._filter_driver_action(action, current, velocity)
        output.append(safe)
        current = safe[:6].astype(np.float64)
    output = np.asarray(output)
    # Acceleration limiting prevents an instantaneous +0.5 -> -0.5 reversal.
    np.testing.assert_allclose(output[:, 0], [0.4, 0.7, 0.9])


def test_absolute_time_joint_ensemble_aligns_different_chunk_offsets():
    old = np.zeros((4, 7), dtype=np.float32)
    new = np.zeros((4, 7), dtype=np.float32)
    old[:, :6] = np.asarray([0.0, 10.0, 20.0, 30.0])[:, None]
    new[:, :6] = np.asarray([100.0, 110.0, 120.0, 130.0])[:, None]
    old[:, 6] = [100.0, 100.0, 0.0, 0.0]
    new[:, 6] = [100.0, 0.0, 100.0, 100.0]
    action, report = absolute_time_joint_ensemble(
        [(1, 0, old), (2, 2, new)], policy_tick=2, decay=0.0
    )
    # Absolute tick 2 means old offset 2 and new offset 0, not offset 0 for both.
    np.testing.assert_allclose(action[:6], 60.0)
    assert report["source_offsets"] == [2, 0]
    assert report["vote_count"] == 2
    assert action[6] == 100.0  # a 50/50 vote deterministically resolves open


def test_absolute_time_joint_ensemble_decay_favors_recent_plan():
    old = np.zeros((6, 7), dtype=np.float32)
    new = np.zeros((6, 7), dtype=np.float32)
    old[:, :6] = -10.0
    new[:, :6] = 10.0
    action, report = absolute_time_joint_ensemble(
        [(1, 0, old), (2, 4, new)], policy_tick=4, decay=0.5
    )
    assert action[0] > 0.0
    assert report["weights"][1] > report["weights"][0]
    assert action[6] in (0.0, 100.0)


def test_chunk_smoothing_reduces_joint_jitter_without_changing_gripper():
    actions = np.zeros((21, 7), dtype=np.float32)
    actions[:, :6] = np.where(np.arange(21)[:, None] % 2 == 0, -1.0, 1.0)
    actions[:, 6] = np.where(np.arange(21) % 3 == 0, 1.0, 0.0)
    smoothed = smooth_joint_action_chunk(actions, window=11, sigma=2.0)
    raw_rms = np.sqrt(np.mean(np.square(np.diff(actions[:, :6], axis=0)), axis=1))
    smooth_rms = np.sqrt(np.mean(np.square(np.diff(smoothed[:, :6], axis=0)), axis=1))
    assert smooth_rms.mean() < raw_rms.mean() * 0.25
    np.testing.assert_array_equal(smoothed[:, 6], actions[:, 6])


def test_chunk_smoothing_preserves_constant_joint_trajectory():
    actions = np.full((21, 7), 0.25, dtype=np.float32)
    actions[:, 6] = np.arange(21) % 2
    smoothed = smooth_joint_action_chunk(actions, window=11, sigma=2.0)
    np.testing.assert_allclose(smoothed[:, :6], actions[:, :6], atol=1e-7)
    np.testing.assert_array_equal(smoothed[:, 6], actions[:, 6])


def _contract_policy(head, *, require_residual=True):
    policy = object.__new__(EgoVerseHPTJointPolicy)
    policy._embodiment_id = 1
    policy.config = SimpleNamespace(
        require_residual_joint_head=require_residual,
        chunk_size=100,
        action_horizon_frames=30,
    )
    policy._algo = SimpleNamespace(
        ac_keys={1: "actions_joints"},
        shared_ac_key="actions_joints",
        auxiliary_ac_keys={},
        proprio_keys={1: ["observations.state.joints"]},
        norm_stats=SimpleNamespace(
            norm_stats={
                1: {
                    "observations.state.joints": {},
                    "actions_joints": {},
                }
            }
        ),
        nets={"policy": SimpleNamespace(heads={"shared": head})},
    )
    return policy


def test_checkpoint_contract_accepts_residual_head():
    ResidualJointPolicy = type("ResidualJointPolicy", (), {})
    head = ResidualJointPolicy()
    head.requires_data_for_inference = True
    head.action_horizon = 100
    head.control_horizon = 30
    head.joint_dims = 6
    policy = _contract_policy(head)
    policy._validate_checkpoint_contract()
    assert policy.checkpoint_report["is_residual_joint_head"] is True
    assert policy.checkpoint_report["command_generation"].endswith("no_ik")


def test_checkpoint_contract_rejects_legacy_head_by_default():
    LegacyJointFM = type("JointFMPolicy", (), {})
    head = LegacyJointFM()
    with pytest.raises(RuntimeError, match="not the deterministic residual-joint"):
        _contract_policy(head)._validate_checkpoint_contract()


def test_checkpoint_contract_can_inspect_legacy_head_explicitly():
    LegacyJointFM = type("JointFMPolicy", (), {})
    policy = _contract_policy(LegacyJointFM(), require_residual=False)
    policy._validate_checkpoint_contract()
    assert policy.checkpoint_report["is_residual_joint_head"] is False


def _v3_contract_policy(head):
    policy = object.__new__(EgoVerseHPTJointPolicy)
    policy._embodiment_id = 1
    policy.config = SimpleNamespace(
        require_residual_joint_head=True,
        chunk_size=30,
        action_horizon_frames=30,
    )
    policy._algo = SimpleNamespace(
        ac_keys={1: "actions_cartesian_pose"},
        shared_ac_key="actions_cartesian_pose",
        auxiliary_ac_keys={"piper_left_arm": ["actions_joints"]},
        proprio_keys={1: ["observations.state.joints"]},
        norm_stats=SimpleNamespace(
            norm_stats={
                1: {
                    "observations.state.joints": {},
                    "actions_joints": {},
                }
            }
        ),
        nets={"policy": SimpleNamespace(heads={"shared": head})},
    )
    return policy


def test_checkpoint_contract_accepts_shared_action_v3_head():
    head = SimpleNamespace(
        contract_name="hpt_mixed_shared_action_v3",
        contract_version=3,
        requires_data_for_inference=True,
        action_horizon=30,
        joint_dims=6,
    )
    policy = _v3_contract_policy(head)
    policy._validate_checkpoint_contract()
    assert policy.checkpoint_report["action_head_name"] == "shared:actions_joints"
    assert policy.checkpoint_report["model_action_horizon"] == 30
    assert policy.checkpoint_report["native_control_horizon"] == 30


def test_shared_action_v3_inference_uses_predict_all_joint_projection():
    expected = torch.zeros((1, 30, 7), dtype=torch.float32)

    class SharedV3Head:
        contract_name = "hpt_mixed_shared_action_v3"

        def predict_all(self, features, data):
            assert features == "features"
            assert data == {"contract": "data"}
            return {
                "actions_cartesian_pose": torch.ones((1, 30, 9)),
                "actions_joints": expected,
            }

    policy = _v3_contract_policy(SharedV3Head())
    prediction, output_name = policy._predict_normalized_joint_actions(
        "features", {"contract": "data"}
    )
    assert prediction is expected
    assert output_name == "shared:actions_joints"
