import torch

from egomimic.models.pi05_egoverse_multihead_v2 import (
    Pi05PrefixJointHeadV2,
    ensure_btchw,
    rotation_6d_to_matrix,
)


def test_single_frame_image_contract_accepts_channel_first_and_last():
    channel_first = torch.zeros(2, 3, 16, 20)
    channel_last = torch.zeros(2, 16, 20, 3)
    assert ensure_btchw(channel_first).shape == (2, 1, 3, 16, 20)
    assert ensure_btchw(channel_last).shape == (2, 1, 3, 16, 20)


def test_rotation_6d_is_orthonormal():
    matrix = rotation_6d_to_matrix(torch.randn(2, 30, 6))
    identity = matrix.transpose(-1, -2) @ matrix
    torch.testing.assert_close(identity, torch.eye(3).expand_as(identity), atol=1e-5, rtol=1e-5)


def test_joint_head_outputs_100_by_7_and_binary_gripper():
    head = Pi05PrefixJointHeadV2(
        prefix_dim=64,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
    )
    output = head.predict(
        prefix_features=torch.randn(2, 12, 64),
        prefix_pad_mask=torch.ones(2, 12, dtype=torch.bool),
        normalized_state_history=torch.randn(2, 1, 7),
        current_raw=torch.randn(2, 7),
    )
    assert output.shape == (2, 100, 7)
    assert set(torch.unique(output[..., 6]).tolist()) <= {0.0, 1.0}


def test_single_frame_joint_physical_loss_is_finite_and_differentiable():
    head = Pi05PrefixJointHeadV2(
        prefix_dim=64,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    current = torch.zeros(1, 7)
    phase = torch.linspace(0.0, 1.0, 100)[None, :, None]
    target = current[:, None].expand(-1, 100, -1).clone()
    target[..., :6] += phase * torch.tensor(
        [0.05, -0.04, 0.03, -0.02, 0.01, -0.03]
    )
    target[..., 6] = (torch.arange(100) < 50).float()
    with torch.no_grad():
        fk = head._fk(target[..., :6])
        camera_pose = head.camera_from_world[None, None] @ fk
        rotation6d = torch.cat(
            (camera_pose[..., :3, 0], camera_pose[..., :3, 1]), dim=-1
        )
        target_pose = torch.cat((camera_pose[..., :3, 3], rotation6d), dim=-1)
    loss = head.compute_loss(
        prefix_features=torch.randn(1, 12, 64),
        prefix_pad_mask=torch.ones(1, 12, dtype=torch.bool),
        normalized_state_history=current[:, None],
        current_raw=current,
        target_joint_raw=target,
        target_pose_camera_raw=target_pose,
    )
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in head.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
