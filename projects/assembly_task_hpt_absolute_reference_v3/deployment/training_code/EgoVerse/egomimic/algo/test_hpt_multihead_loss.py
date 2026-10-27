import types

import torch
from torch import nn

from egomimic.algo.hpt import HPTModel


class _ConstantLossHead(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def compute_loss(self, features, data):
        return data["action"].new_tensor(self.value)


def test_shared_and_robot_auxiliary_losses_are_weighted_and_action_is_restored():
    model = HPTModel(
        embed_dim=4,
        num_blocks=0,
        num_heads=1,
        observation_horizon=1,
        action_horizon=1,
    )
    model.device = torch.device("cpu")
    model.shared_action = True
    model.auxiliary_ac_keys = {"piper_left_arm": ["actions_joints"]}
    model.loss_weights = {
        "piper_left_arm": {"shared": 1.0, "actions_joints": 2.0}
    }
    model.heads = {
        "shared": _ConstantLossHead(2.0),
        "piper_left_arm_actions_joints": _ConstantLossHead(3.0),
    }

    def _forward_features(self, domain, data):
        return torch.zeros(1, 1, 4), []

    model.forward_features = types.MethodType(_forward_features, model)
    shared_action = torch.zeros(1, 2, 9)
    data = {
        "action": shared_action,
        "actions_joints": torch.zeros(1, 2, 7),
    }
    loss = model.compute_loss({"domain": "piper_left_arm", "data": data})

    assert torch.isclose(loss, torch.tensor(8.0))
    assert data["action"] is shared_action


def test_unspecified_head_weight_defaults_to_one():
    model = HPTModel(
        embed_dim=4,
        num_blocks=0,
        num_heads=1,
        observation_horizon=1,
        action_horizon=1,
    )
    assert model._head_loss_weight("human_left_arm", "shared") == 1.0
