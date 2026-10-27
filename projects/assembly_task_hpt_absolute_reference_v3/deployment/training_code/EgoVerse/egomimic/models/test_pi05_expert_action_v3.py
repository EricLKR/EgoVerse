import math

import torch

from egomimic.models.pi05_expert_action_v3 import (
    JOINT_SLICE,
    POSE_SLICE,
    RESERVED_SLICE,
    Pi05ExpertActionAuxiliaryV3,
    build_expert_action_target,
    clean_action_from_velocity,
    joint_residual_from_flow,
)
from egomimic.rldb.embodiment.pi05_expert_action_v3 import PiperExpertActionV3


def _pose(batch=2):
    action = torch.randn(batch, 30, 32)
    action[..., 9:] = 0
    weight = torch.zeros_like(action)
    weight[..., :9] = 1
    return action, weight


def test_v3_uses_native_30_step_data_contract():
    assert PiperExpertActionV3.ACTION_HORIZON == 30
    assert PiperExpertActionV3.CHUNK_LENGTH == 30


def test_human_target_has_exact_zero_robot_gradient_mask():
    pose, pose_weight = _pose()
    target, mask = build_expert_action_target(
        pose, pose_weight, pose_weight=1.0
    )
    assert target.shape == mask.shape == (2, 30, 32)
    assert torch.all(mask[..., POSE_SLICE] == 1)
    assert torch.all(mask[..., JOINT_SLICE] == 0)
    assert torch.all(mask[..., RESERVED_SLICE] == 0)

    prediction = torch.randn_like(target, requires_grad=True)
    loss = ((prediction - target).square() * mask).sum() / mask.sum()
    loss.backward()
    assert prediction.grad[..., POSE_SLICE].abs().sum() > 0
    assert prediction.grad[..., JOINT_SLICE].abs().sum() == 0
    assert prediction.grad[..., RESERVED_SLICE].abs().sum() == 0


def test_piper_joint_residual_roundtrip_and_mask():
    pose, pose_weight = _pose(batch=1)
    current = torch.zeros(1, 7)
    target_joint = torch.zeros(1, 30, 7)
    target_joint[..., :6] = torch.linspace(0, 0.3, 30)[None, :, None]
    target_joint[..., 6] = (torch.arange(30) < 15).float()
    scale = math.pi / 2
    target, mask = build_expert_action_target(
        pose,
        pose_weight,
        pose_weight=1.0,
        target_joint_raw=target_joint,
        current_raw=current,
        joint_residual_scale_rad=scale,
    )
    decoded = joint_residual_from_flow(
        target[..., JOINT_SLICE], current[..., :6], scale
    )
    torch.testing.assert_close(decoded, target_joint[..., :6])
    assert torch.all(mask[..., JOINT_SLICE] == 1)
    assert torch.all(mask[..., RESERVED_SLICE] == 0)


def test_rectified_flow_clean_action_identity():
    action = torch.randn(3, 30, 32)
    noise = torch.randn_like(action)
    time = torch.tensor([0.1, 0.5, 0.9])
    noisy = time[:, None, None] * noise + (1 - time[:, None, None]) * action
    velocity = noise - action
    recovered = clean_action_from_velocity(noisy, time, velocity)
    torch.testing.assert_close(recovered, action)


def test_auxiliary_loss_and_gripper_are_differentiable_and_binary():
    module = Pi05ExpertActionAuxiliaryV3(
        expert_dim=64,
        gripper_hidden_dim=16,
        enable_camera_consistency=False,
    )
    hidden = torch.randn(2, 30, 64, requires_grad=True)
    clean = torch.zeros(2, 30, 32, requires_grad=True)
    current = torch.zeros(2, 7)
    target = torch.zeros(2, 30, 7)
    target[..., :6] = torch.linspace(0, 0.1, 30)[None, :, None]
    target[:, :15, 6] = 1
    loss = module.compute_loss(
        suffix_hidden=hidden,
        predicted_clean_action32=clean,
        current_raw=current,
        target_joint_raw=target,
        predicted_pose_camera_raw=None,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert clean.grad is not None and clean.grad[..., JOINT_SLICE].abs().sum() > 0

    actions, logits = module.decode_actions(
        sampled_action32=clean.detach(),
        suffix_hidden=hidden.detach(),
        current_raw=current,
    )
    assert actions.shape == (2, 30, 7)
    assert logits.shape == (2, 30)
    assert set(torch.unique(actions[..., 6]).tolist()) <= {0.0, 1.0}
