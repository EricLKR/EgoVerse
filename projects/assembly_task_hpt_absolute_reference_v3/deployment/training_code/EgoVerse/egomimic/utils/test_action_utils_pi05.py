import pytest
import torch

from egomimic.utils.action_utils import (
    HumanLeftXYZ6DActionConverter,
    PiperJointActionConverter,
)


def test_piper_joint_converter_round_trip_and_mask():
    actions = torch.randn(2, 100, 7)
    actions[..., 6] = (actions[..., 6] > 0).float()
    converter = PiperJointActionConverter(gripper_weight=3.0)

    packed = converter.to32(actions)
    restored = converter.from32(packed)
    weights = converter.loss_weights32(actions)

    assert packed.shape == (2, 100, 32)
    assert torch.equal(restored, actions)
    assert torch.all(weights[..., :6] == 1)
    assert torch.all(weights[..., 6] == 3)
    assert torch.all(weights[..., 7:] == 0)


def test_human_xyz6d_converter_round_trip_and_mask():
    actions = torch.randn(3, 100, 9)
    converter = HumanLeftXYZ6DActionConverter()

    packed = converter.to32(actions)
    restored = converter.from32(packed)
    mask = converter.valid_mask32(actions)

    assert packed.shape == (3, 100, 32)
    assert torch.equal(restored, actions)
    assert torch.all(mask[..., :9])
    assert not torch.any(mask[..., 9:])


@pytest.mark.parametrize("bad_dim", [6, 8, 10])
def test_new_converters_reject_wrong_dimensions(bad_dim):
    actions = torch.zeros(1, 4, bad_dim)
    with pytest.raises(ValueError):
        PiperJointActionConverter().to32(actions)
    with pytest.raises(ValueError):
        HumanLeftXYZ6DActionConverter().to32(actions)
