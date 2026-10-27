#!/usr/bin/env python
"""CPU-only structural tests for the MultiHead V2 deployment head."""

from __future__ import annotations

import torch

from lerobot.policies.pi05.modeling_pi05_multihead_v2_joint import (
    Pi05PrefixJointHeadV2Deployment,
)


def test_joint_head_shape_binary_gripper_and_residual_anchor() -> None:
    torch.manual_seed(7)
    head = Pi05PrefixJointHeadV2Deployment(
        prefix_dim=16,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()
    batch = 2
    prefix = torch.randn(batch, 12, 16)
    prefix_mask = torch.ones(batch, 12, dtype=torch.bool)
    normalized_state = torch.randn(batch, 1, 7).clamp(-1.0, 1.0)
    current = torch.randn(batch, 7)
    prediction, logits = head.predict_with_details(
        prefix_features=prefix,
        prefix_pad_mask=prefix_mask,
        normalized_state_history=normalized_state,
        current_raw=current,
    )
    assert prediction.shape == (batch, 100, 7)
    assert logits.shape == (batch, 30)
    assert torch.isfinite(prediction).all()
    assert set(torch.unique(prediction[..., 6]).tolist()).issubset({0.0, 1.0})
    max_residual = (prediction[..., :6] - current[:, None, :6]).abs().max()
    assert max_residual <= head.max_residual_rad + 1e-6


def test_training_persistent_buffers_are_part_of_strict_contract() -> None:
    head = Pi05PrefixJointHeadV2Deployment()
    state = head.state_dict()
    assert state["joint_dimension_weights"].shape == (6,)
    assert state["camera_from_world"].shape == (4, 4)


if __name__ == "__main__":
    test_joint_head_shape_binary_gripper_and_residual_anchor()
    test_training_persistent_buffers_are_part_of_strict_contract()
    print("PI05_MULTIHEAD_V2_DEPLOY_STATIC_TEST_OK")

