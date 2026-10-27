import types

import torch

from egomimic.algo.hpt_absolute_reference_v3 import HPTAbsoluteReferenceV3
from egomimic.models.absolute_reference_joint_policy_v3 import (
    AbsoluteReferenceJointPolicyV3,
)


CENTER = [22.805, 83.5205, -55.842, -14.6397, 25.1686, 27.9814]
REFERENCE_RANGE = [39.23, 57.43, 66.94, 47.82, 71.01, 38.94]
CORRECTION = [4.0, 5.0, 5.0, 4.0, 5.0, 4.0]


def _head_and_data():
    torch.manual_seed(7)
    head = AbsoluteReferenceJointPolicyV3(
        action_horizon=8,
        control_horizon=4,
        infer_ac_dims={"piper_left_arm": 7},
        input_dim=16,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        joint_reference_center_deg=CENTER,
        joint_reference_range_deg=REFERENCE_RANGE,
        joint_correction_scale_deg=CORRECTION,
        fk_position_loss_weight=0.0,
        fk_rotation_loss_weight=0.0,
    )
    head._fk_losses = types.MethodType(
        lambda self, prediction, target: (
            prediction.new_zeros(()),
            prediction.new_zeros(()),
        ),
        head,
    )
    batch = 2
    action_low = torch.full((8, 7), -2.5)
    action_high = torch.full((8, 7), 2.5)
    action_low[:, 6] = 0.0
    action_high[:, 6] = 1.0
    raw_action = torch.zeros(batch, 8, 7)
    raw_action[..., :6] = torch.deg2rad(torch.tensor(CENTER))[None, None]
    raw_action[:, :4, 6] = 1.0
    action = 2.0 * (raw_action - action_low) / (action_high - action_low) - 1.0

    proprio_low = torch.tensor([-3.0] * 6 + [0.0])
    proprio_high = torch.tensor([3.0] * 6 + [1.0])
    raw_proprio = torch.zeros(batch, 7)
    raw_proprio[:, 6] = 1.0
    proprio = 2.0 * (raw_proprio - proprio_low) / (proprio_high - proprio_low) - 1.0
    data = {
        "action": action,
        "state_joints": proprio[:, None],
        "_actions_joints_norm_mode": "quantile",
        "_actions_joints_norm_quantile_1": action_low,
        "_actions_joints_norm_quantile_99": action_high,
        "_proprio_norm_mode": "quantile",
        "_proprio_norm_quantile_1": proprio_low,
        "_proprio_norm_quantile_99": proprio_high,
    }
    return head, data, torch.randn(batch, 5, 16)


def test_zero_logits_decode_to_absolute_reference_not_current_state():
    head, data, features = _head_and_data()
    prediction_a, _ = head._raw_control_prediction(features, data)
    data_b = dict(data)
    data_b["state_joints"] = data["state_joints"] + 0.7
    prediction_b, _ = head._raw_control_prediction(features, data_b)
    expected = torch.deg2rad(torch.tensor(CENTER))[None, None]
    assert torch.allclose(prediction_a, expected.expand_as(prediction_a), atol=1e-6)
    assert torch.allclose(prediction_a, prediction_b, atol=1e-7)


def test_reference_and_correction_are_bounded_per_joint():
    head, _, features = _head_and_data()
    with torch.no_grad():
        head.joint_reference_head.bias.fill_(100.0)
        head.joint_correction_head.bias.fill_(-100.0)
    reference, correction, _ = head._decode_components(features)
    upper = head.joint_reference_center_rad + head.joint_reference_range_rad
    assert torch.allclose(reference[0, 0], upper, atol=1e-5)
    assert torch.allclose(
        correction[0, 0], -head.joint_correction_scale_rad, atol=1e-5
    )


def test_nominal_and_recovery_losses_are_finite_and_differentiable():
    head, data, features = _head_and_data()
    nominal = head.compute_loss(features, data)
    clean = head._last_clean_prediction
    assert clean is not None
    recovery_features = features + 0.05 * torch.randn_like(features)
    weighted, recovery, consistency = head.compute_recovery_losses(
        recovery_features, data, clean
    )
    total = nominal + weighted
    assert all(torch.isfinite(value) for value in (total, recovery, consistency))
    total.backward()
    assert head.joint_reference_head.weight.grad is not None
    assert head.joint_correction_head.weight.grad is not None


def test_train_split_extrema_are_representable_with_margin():
    train_min = torch.tensor([-14.425, 28.097, -120.777, -60.459, -43.840, -8.681])
    train_max = torch.tensor([57.343, 125.858, -1.000, 13.943, 87.700, 64.919])
    center = torch.tensor(CENTER)
    reference_range = torch.tensor(REFERENCE_RANGE)
    assert torch.all(center - reference_range <= train_min - 1.9)
    assert torch.all(center + reference_range >= train_max + 1.9)


def test_algo_resolves_joint_head_from_moduledict():
    head, _, _ = _head_and_data()

    class _Policy(torch.nn.Module):
        def __init__(self, joint_head):
            super().__init__()
            self.heads = torch.nn.ModuleDict(
                {"piper_left_arm_actions_joints": joint_head}
            )

    algo = HPTAbsoluteReferenceV3.__new__(HPTAbsoluteReferenceV3)
    # This unit test exercises only the head resolver.  Do not partially
    # initialize the inherited Lightning/robomimic module: newer PyTorch
    # versions inspect class state such as ``call_super_init`` during
    # ``Module.__init__``, while the real algorithm is always constructed by
    # its normal full initializer.  A direct test fixture keeps the test
    # independent of that framework initialization contract.
    algo.__dict__["nets"] = {"policy": _Policy(head)}
    assert algo._joint_head() is head
