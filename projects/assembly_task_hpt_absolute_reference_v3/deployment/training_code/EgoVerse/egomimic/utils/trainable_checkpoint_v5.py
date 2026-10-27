"""Compact, exact-interface checkpoints for Mixed CamFrame V5.

The legacy compact callback intentionally stores only trainable parameters.
V5 additionally has small persistent buffers that define its deployment and
resume contract (bounded joint scales and the Human-loss schedule step).  This
additive callback stores every trainable parameter plus every persistent model
buffer, while still excluding the frozen pi0.5 backbone parameters.
"""

from __future__ import annotations

import json
import os

import torch
from lightning import LightningModule
from lightning import Trainer

from egomimic.utils.trainable_checkpoint import TrainableStateCheckpoint


class TrainableStateAndBuffersCheckpoint(TrainableStateCheckpoint):
    """Save trainable parameters and all persistent buffers."""

    @staticmethod
    def _trainable_state(pl_module: LightningModule):
        state = pl_module.state_dict()
        required_names = {
            name
            for name, parameter in pl_module.named_parameters()
            if parameter.requires_grad
        }
        # ``named_buffers`` also reports buffers registered with
        # persistent=False; those are intentionally absent from state_dict.
        required_names.update(
            name for name, _ in pl_module.named_buffers() if name in state
        )
        missing = sorted(name for name in required_names if name not in state)
        if missing:
            raise RuntimeError(
                "Checkpoint state is missing required parameter/buffer keys: "
                + ", ".join(missing[:10])
            )
        compact = {
            name: state[name].detach().cpu().clone()
            for name in sorted(required_names)
        }
        if not compact:
            raise RuntimeError("No trainable parameters or buffers were found")
        return compact

    def _save(self, trainer: Trainer, pl_module: LightningModule, step: int) -> None:
        if not trainer.is_global_zero or step == self._last_saved_step:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        state = self._trainable_state(pl_module)
        tensor_count = sum(value.numel() for value in state.values())
        filename = f"step_{step:08d}_trainable.ckpt"
        destination = self.dirpath / filename
        temporary = self.dirpath / f".{filename}.tmp"
        payload = {
            "state_dict": state,
            "global_step": int(step),
            "checkpoint_format": "egomimic_trainable_state_and_buffers_v5",
            "stored_tensor_element_count": int(tensor_count),
            "state_keys": sorted(state),
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        metadata = {
            "checkpoint": str(destination),
            "checkpoint_format": payload["checkpoint_format"],
            "global_step": int(step),
            "state_key_count": len(state),
            "stored_tensor_element_count": int(tensor_count),
            "size_bytes": destination.stat().st_size,
        }
        destination.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._last_saved_step = step
        print(
            "[V5_COMPACT_CHECKPOINT] "
            f"step={step} tensor_elements={tensor_count} path={destination}",
            flush=True,
        )
