#!/usr/bin/env python
"""CPU-only structural tests for HPT Shared-Action V3 deployment plumbing."""

from __future__ import annotations

import numpy as np
import torch

from deploy_egoverse_hpt_shared_action_v3_joint import (
    HPT_SHARED_V3_CHECKPOINT_PROFILES,
    validate_deploy_result,
)
from lerobot.policies.egoverse_hpt.modeling_egoverse_hpt_joint import (
    model_indices_for_control_ticks,
)


def valid_result() -> dict[str, torch.Tensor]:
    actions = torch.zeros(1, 30, 7)
    actions[..., 6] = torch.tensor(([1.0] * 10) + ([0.0] * 10) + ([1.0] * 10))
    return {
        "actions": actions,
        "actions_normalized": torch.zeros(1, 30, 7),
        "shared_pose": torch.zeros(1, 30, 9),
        "gripper_logits": torch.linspace(2.0, -2.0, 30).unsqueeze(0),
        "normalized_state": torch.zeros(1, 7),
    }


def test_native_30_step_contract() -> None:
    actions, normalized, pose, logits, state = validate_deploy_result(valid_result())
    assert actions.shape == (30, 7)
    assert normalized.shape == (30, 7)
    assert pose.shape == (30, 9)
    assert logits.shape == (30,)
    assert state.shape == (7,)
    assert np.array_equal(
        model_indices_for_control_ticks(30, 30, 30), list(range(30))
    )
    assert np.array_equal(
        model_indices_for_control_ticks(30, 30, 10), list(range(10))
    )


def test_nonbinary_gripper_is_rejected() -> None:
    result = valid_result()
    result["actions"][..., 6] = 0.5
    try:
        validate_deploy_result(result)
    except RuntimeError as exc:
        assert "not binary" in str(exc)
    else:
        raise AssertionError("non-binary gripper output was accepted")


def test_profiles_are_distinct_and_audited() -> None:
    five = HPT_SHARED_V3_CHECKPOINT_PROFILES["hpt-v3-trim-val-best-5k"]
    fifteen = HPT_SHARED_V3_CHECKPOINT_PROFILES[
        "hpt-v3-trim-joint-candidate-15k"
    ]
    assert len(five["sha256"]) == 64
    assert len(fifteen["sha256"]) == 64
    assert five["sha256"] != fifteen["sha256"]
    assert five["checkpoint"] != fifteen["checkpoint"]
    assert five["model_action_horizon"] == 30
    assert fifteen["model_action_horizon"] == 30


if __name__ == "__main__":
    test_native_30_step_contract()
    test_nonbinary_gripper_is_rejected()
    test_profiles_are_distinct_and_audited()
    print("HPT_SHARED_ACTION_V3_DEPLOY_STATIC_TEST_OK")
