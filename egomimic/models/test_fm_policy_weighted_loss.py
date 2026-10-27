import torch

from egomimic.models.fm_policy import FMPolicy


class _ZeroNet(torch.nn.Module):
    def forward(self, x_t, time, global_cond):
        return torch.zeros_like(x_t)


def _policy(**kwargs):
    return FMPolicy(
        model=_ZeroNet(),
        action_horizon=4,
        infer_ac_dims={"piper_left_arm": 10},
        num_inference_steps=10,
        **kwargs,
    )


def test_temporal_weights_emphasize_only_configured_prefix():
    policy = _policy(early_horizon=2, early_weight=3.0)
    actions = torch.zeros(1, 4, 10)
    expected = torch.tensor([3.0, 3.0, 1.0, 1.0])
    torch.testing.assert_close(policy._temporal_weights(actions)[0, :, 0], expected)


def test_rotation_geodesic_identity_is_zero():
    policy = _policy(rotation_geodesic_weight=0.1)
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    matrix = policy._rotation_6d_to_matrix(identity_6d)
    torch.testing.assert_close(matrix, torch.eye(3), atol=1e-6, rtol=0.0)


def test_rotation_geodesic_detects_quarter_turn():
    policy = _policy(rotation_geodesic_weight=0.1)
    target = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    quarter_turn = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
    target_matrix = policy._rotation_6d_to_matrix(target)
    pred_matrix = policy._rotation_6d_to_matrix(quarter_turn)
    relative = pred_matrix.transpose(-1, -2) @ target_matrix
    cosine = (torch.trace(relative) - 1.0) * 0.5
    angle = torch.acos(cosine.clamp(-1.0, 1.0))
    torch.testing.assert_close(angle, torch.tensor(torch.pi / 2), atol=1e-6, rtol=0.0)
