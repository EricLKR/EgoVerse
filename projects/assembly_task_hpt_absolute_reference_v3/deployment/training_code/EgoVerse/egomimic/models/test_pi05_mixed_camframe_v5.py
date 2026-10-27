import torch

from egomimic.models.pi05_expert_action_v3 import JOINT_SLICE
from egomimic.models.pi05_expert_action_v4_bounded import (
    build_expert_action_target_v4,
)
from egomimic.models.pi05_mixed_camframe_v5 import (
    Pi05MixedDomainConditioningV5,
    human_ramp_weight,
)


def test_domain_conditioning_has_strict_state_contracts():
    module = Pi05MixedDomainConditioningV5(prefix_dim=16, hidden_dim=8)
    piper_token, piper_embedding = module(torch.zeros(2, 7), 0)
    human_token, human_embedding = module(torch.zeros(2, 9), 1)
    assert piper_token.shape == human_token.shape == (2, 16)
    assert piper_embedding.shape == human_embedding.shape == (16,)
    assert not torch.equal(piper_embedding, human_embedding)


def test_human_target_has_no_joint_or_reserved_gradient_mask():
    pose = torch.randn(2, 30, 32)
    pose_weight = torch.ones_like(pose)
    _, mask = build_expert_action_target_v4(
        pose,
        pose_weight,
        pose_weight=1.0,
        joint_residual_scale_rad=torch.ones(6),
    )
    assert torch.count_nonzero(mask[..., JOINT_SLICE]) == 0
    assert torch.count_nonzero(mask[..., 15:]) == 0


def test_human_weight_ramp():
    assert human_ramp_weight(0, final_weight=0.3, ramp_steps=2500) == 0.0
    assert abs(human_ramp_weight(1250, final_weight=0.3, ramp_steps=2500) - 0.15) < 1e-8
    assert human_ramp_weight(3000, final_weight=0.3, ramp_steps=2500) == 0.3


