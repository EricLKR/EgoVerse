from types import SimpleNamespace

import torch
from lightning import LightningModule

from egomimic.utils.trainable_checkpoint import TrainableStateCheckpoint


class _TinyModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(3, 2)
        self.trainable = torch.nn.Linear(2, 1)
        for parameter in self.frozen.parameters():
            parameter.requires_grad_(False)
        # Exercise the duplicate-module alias used by ModelWrapper.
        self.trainable_alias = self.trainable


def test_trainable_checkpoint_is_compact_and_partially_loadable(tmp_path):
    source = _TinyModule()
    callback = TrainableStateCheckpoint(
        dirpath=str(tmp_path), every_n_train_steps=5
    )
    callback._save(SimpleNamespace(is_global_zero=True), source, step=5)

    checkpoint_path = tmp_path / "step_00000005_trainable.ckpt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["checkpoint_format"] == "egomimic_trainable_state_v1"
    assert set(payload["state_dict"]) == {
        "trainable.bias",
        "trainable.weight",
    }
    assert payload["trainable_parameter_count"] == 3

    target = _TinyModule()
    frozen_before = target.frozen.weight.detach().clone()
    incompatible = target.load_state_dict(payload["state_dict"], strict=False)
    assert "frozen.weight" in incompatible.missing_keys
    assert torch.equal(target.frozen.weight, frozen_before)
    assert torch.equal(target.trainable.weight, source.trainable.weight)
    assert torch.equal(target.trainable.bias, source.trainable.bias)
