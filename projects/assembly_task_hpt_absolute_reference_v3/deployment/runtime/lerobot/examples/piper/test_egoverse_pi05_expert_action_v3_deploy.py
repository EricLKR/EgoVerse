#!/usr/bin/env python
"""CPU-only structural tests for Expert-Action V3 deployment plumbing."""

from __future__ import annotations

import numpy as np
import torch

from deploy_egoverse_pi05_expert_action_v3_joint import (
    EXPERT_ACTION_V3_CHECKPOINT_PROFILES,
    validate_v3_deploy_result,
)
from lerobot.policies.egoverse_hpt.modeling_egoverse_hpt_joint import (
    model_indices_for_control_ticks,
)


def test_native_30_step_result_contract() -> None:
    actions = torch.zeros(1, 30, 7)
    actions[..., 6] = torch.tensor(([1.0] * 10) + ([0.0] * 10) + ([1.0] * 10))
    result = {
        "actions": actions,
        "action32": torch.randn(1, 30, 32),
        "gripper_logits": torch.linspace(2.0, -2.0, 30).unsqueeze(0),
        "normalized_state": torch.zeros(1, 7),
    }
    decoded, action32, logits, state = validate_v3_deploy_result(result)
    assert decoded.shape == (30, 7)
    assert action32.shape == (30, 32)
    assert logits.shape == (30,)
    assert state.shape == (7,)
    assert np.array_equal(model_indices_for_control_ticks(30, 30, 30), list(range(30)))


def test_nonbinary_gripper_is_rejected() -> None:
    result = {
        "actions": torch.zeros(1, 30, 7),
        "action32": torch.zeros(1, 30, 32),
        "gripper_logits": torch.zeros(1, 30),
        "normalized_state": torch.zeros(1, 7),
    }
    result["actions"][..., 6] = 0.5
    try:
        validate_v3_deploy_result(result)
    except RuntimeError as exc:
        assert "not binary" in str(exc)
    else:
        raise AssertionError("non-binary V3 gripper output was accepted")


def test_profile_uses_audited_hash() -> None:
    profile = EXPERT_ACTION_V3_CHECKPOINT_PROFILES[
        "pi05-expert-action-v3-20k"
    ]
    assert len(profile["sha256"]) == 64
    assert profile["checkpoint"].endswith("pi05_expert_action_v3_step20000.ckpt")


if __name__ == "__main__":
    test_native_30_step_result_contract()
    test_nonbinary_gripper_is_rejected()
    test_profile_uses_audited_hash()
    print("PI05_EXPERT_ACTION_V3_DEPLOY_STATIC_TEST_OK")
