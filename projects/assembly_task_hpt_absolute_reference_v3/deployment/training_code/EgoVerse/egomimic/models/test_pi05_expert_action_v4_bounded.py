import math

import pytest
import torch

from egomimic.models.pi05_expert_action_v3 import JOINT_SLICE, POSE_SLICE
from egomimic.models.pi05_expert_action_v4_bounded import (
    Pi05ExpertActionAuxiliaryV4Bounded,
    bounded_joint_from_latent,
    bounded_joint_to_latent,
    build_expert_action_target_v4,
)


SCALE_DEG = [45.0, 35.0, 40.0, 55.0, 50.0, 60.0]
SCALE_RAD = torch.deg2rad(torch.tensor(SCALE_DEG))


def test_bounded_joint_round_trip_and_limit():
    current = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]])
    fraction = torch.linspace(-0.8, 0.8, 30)[None, :, None]
    target = current[:, None] + fraction * SCALE_RAD[None, None]
    latent = bounded_joint_to_latent(target, current, SCALE_RAD)
    decoded = bounded_joint_from_latent(latent, current, SCALE_RAD)
    torch.testing.assert_close(decoded, target, atol=2e-6, rtol=2e-6)

    extreme = bounded_joint_from_latent(
        torch.full((1, 30, 6), 1e6), current, SCALE_RAD
    )
    torch.testing.assert_close(
        extreme - current[:, None],
        SCALE_RAD[None, None].expand_as(extreme),
        atol=2e-6,
        rtol=2e-6,
    )


def test_bounded_encoder_rejects_unrepresentable_target():
    current = torch.zeros(1, 6)
    target = torch.zeros(1, 30, 6)
    target[:, :, 2] = SCALE_RAD[2]
    with pytest.raises(ValueError, match="reaches/exceeds"):
        bounded_joint_to_latent(target, current, SCALE_RAD)


def test_v4_target_masks_reserved_channels_and_round_trips():
    pose = torch.randn(2, 30, 32)
    weights = torch.ones_like(pose)
    current = torch.zeros(2, 7)
    joint = torch.zeros(2, 30, 7)
    joint[..., :6] = 0.4 * SCALE_RAD
    joint[..., 6] = 1.0
    target, mask = build_expert_action_target_v4(
        pose,
        weights,
        pose_weight=0.5,
        target_joint_raw=joint,
        current_raw=current,
        joint_residual_scale_rad=SCALE_RAD,
    )
    assert torch.all(mask[..., POSE_SLICE] == 0.5)
    assert torch.all(mask[..., JOINT_SLICE] == 1.0)
    assert torch.count_nonzero(target[..., 15:]) == 0
    assert torch.count_nonzero(mask[..., 15:]) == 0
    decoded = bounded_joint_from_latent(
        target[..., JOINT_SLICE], current[..., :6], SCALE_RAD
    )
    torch.testing.assert_close(decoded, joint[..., :6])


def test_v4_auxiliary_checkpoint_buffers_and_finite_loss():
    auxiliary = Pi05ExpertActionAuxiliaryV4Bounded(
        expert_dim=32,
        gripper_hidden_dim=8,
        action_horizon=30,
        joint_residual_scale_deg=SCALE_DEG,
        enable_camera_consistency=False,
    )
    assert "joint_residual_scale_rad_vector" in auxiliary.state_dict()
    predicted = torch.zeros(2, 30, 32, requires_grad=True)
    hidden = torch.randn(2, 30, 32, requires_grad=True)
    current = torch.zeros(2, 7)
    target = torch.zeros(2, 30, 7)
    target[..., 6] = (torch.arange(30)[None] < 15).float()
    loss = auxiliary.compute_loss(
        suffix_hidden=hidden,
        predicted_clean_action32=predicted,
        current_raw=current,
        target_joint_raw=target,
        predicted_pose_camera_raw=None,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for key in ("first_action", "latent_saturation", "max_predicted_utilization"):
        assert key in auxiliary.last_loss_components


def test_scale_is_per_joint_not_scalar():
    current = torch.zeros(1, 6)
    target = 0.5 * SCALE_RAD[None, None].expand(1, 30, 6)
    latent = bounded_joint_to_latent(target, current, SCALE_RAD)
    expected = math.atanh(0.5)
    torch.testing.assert_close(latent, torch.full_like(latent, expected))
