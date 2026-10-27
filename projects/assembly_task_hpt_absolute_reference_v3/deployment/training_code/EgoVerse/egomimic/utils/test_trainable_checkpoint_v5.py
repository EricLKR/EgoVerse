from __future__ import annotations

import torch
from lightning import LightningModule

from egomimic.utils.trainable_checkpoint_v5 import (
    TrainableStateAndBuffersCheckpoint,
)


class _TinyV5Module(LightningModule):
    def __init__(self):
        super().__init__()
        self.trainable = torch.nn.Linear(2, 2)
        self.frozen = torch.nn.Linear(2, 2)
        self.frozen.requires_grad_(False)
        self.register_buffer("schedule_step", torch.tensor(37, dtype=torch.long))


def test_v5_compact_checkpoint_keeps_buffers_but_not_frozen_parameters():
    state = TrainableStateAndBuffersCheckpoint._trainable_state(_TinyV5Module())
    assert "trainable.weight" in state
    assert "trainable.bias" in state
    assert "schedule_step" in state
    assert state["schedule_step"].item() == 37
    assert "frozen.weight" not in state
    assert "frozen.bias" not in state
