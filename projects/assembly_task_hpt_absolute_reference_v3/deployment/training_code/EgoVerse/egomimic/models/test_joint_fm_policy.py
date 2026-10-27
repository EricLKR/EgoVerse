import torch
from torch import nn

from egomimic.models.joint_fm_policy import JointFMPolicy


class _ZeroVelocity(nn.Module):
    def forward(self, x_t, time, global_cond):
        return torch.zeros_like(x_t)


def _policy(**kwargs):
    return JointFMPolicy(
        model=_ZeroVelocity(),
        action_horizon=4,
        infer_ac_dims={"piper_left_arm": 7},
        num_inference_steps=2,
        joint_dims=6,
        gripper_index=6,
        **kwargs,
    )


def test_inference_hard_binarizes_only_gripper():
    policy = _policy()
    noise = torch.tensor(
        [[[0.2, -0.3, 0.4, 0.5, -0.6, 0.7, -0.1]] * 4], dtype=torch.float32
    )
    output = policy.inference(noise, torch.zeros(1, 2, 3))
    assert torch.allclose(output[..., :6], noise[..., :6])
    assert torch.equal(output[..., 6], -torch.ones_like(output[..., 6]))


def test_loss_is_finite_for_binary_labels():
    policy = _policy()
    actions = torch.zeros(2, 4, 7)
    actions[0, :, 6] = -1.0
    actions[1, :, 6] = 1.0
    loss = policy.compute_loss(
        torch.zeros(2, 2, 3),
        {"action": actions},
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    components = policy.last_loss_components
    assert set(components) == {
        "joint_flow_loss",
        "joint_position_loss",
        "gripper_bce",
        "joint_delta_loss",
        "joint_acceleration_loss",
        "joint_total_loss",
    }
    assert torch.isclose(loss.detach(), components["joint_total_loss"])


def test_gripper_transition_neighborhood_is_upweighted():
    policy = _policy(
        gripper_closed_weight=1.0,
        gripper_open_weight=1.0,
        gripper_transition_weight=2.0,
        gripper_transition_radius=1,
    )
    target_open = torch.tensor([[False, False, True, True, True]])
    weights = policy._gripper_weights(target_open, torch.ones(1, 5))
    assert torch.equal(weights, torch.tensor([[1.0, 2.0, 2.0, 2.0, 1.0]]))


def test_joint_delta_loss_penalizes_jitter_not_matching_motion():
    policy = _policy(joint_delta_loss_weight=0.05, joint_delta_beta=0.1)
    target = torch.zeros(1, 4, 7)
    temporal_weights = torch.ones(1, 4)
    matching = policy._joint_delta_loss(target.clone(), target, temporal_weights)
    jittered = target.clone()
    jittered[0, :, 0] = torch.tensor([0.0, 0.5, -0.5, 0.5])
    jitter = policy._joint_delta_loss(jittered, target, temporal_weights)
    assert matching.item() == 0.0
    assert jitter.item() > 0.0


def test_joint_position_loss_anchors_clean_trajectory():
    policy = _policy(joint_dimension_weights=[1, 1, 1, 2, 1, 1])
    target = torch.zeros(1, 4, 7)
    temporal_weights = torch.ones(1, 4)
    matching = policy._joint_position_loss(target.clone(), target, temporal_weights)
    shifted = target.clone()
    shifted[..., 3] = 0.5
    mismatching = policy._joint_position_loss(shifted, target, temporal_weights)
    assert matching.item() == 0.0
    assert mismatching.item() > 0.0


def test_joint_dimension_weights_emphasize_selected_joint():
    policy = _policy(joint_dimension_weights=[1, 1, 1, 2, 1, 1])
    target = torch.zeros(1, 4, 7)
    temporal_weights = torch.ones(1, 4)
    joint_2_error = target.clone()
    joint_2_error[..., 1] = 0.5
    joint_4_error = target.clone()
    joint_4_error[..., 3] = 0.5
    loss_2 = policy._joint_position_loss(joint_2_error, target, temporal_weights)
    loss_4 = policy._joint_position_loss(joint_4_error, target, temporal_weights)
    assert torch.isclose(loss_4, 2.0 * loss_2)


def test_joint_acceleration_loss_penalizes_unmatched_second_difference():
    policy = _policy()
    target = torch.zeros(1, 4, 7)
    temporal_weights = torch.ones(1, 4)
    matching = policy._joint_acceleration_loss(
        target.clone(), target, temporal_weights
    )
    jittered = target.clone()
    jittered[0, :, 0] = torch.tensor([0.0, 0.5, -0.5, 0.5])
    jitter = policy._joint_acceleration_loss(jittered, target, temporal_weights)
    assert matching.item() == 0.0
    assert jitter.item() > 0.0
