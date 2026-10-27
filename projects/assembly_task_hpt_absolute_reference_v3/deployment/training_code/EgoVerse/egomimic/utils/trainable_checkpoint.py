"""Compact checkpoints for parameter-efficient training runs.

The standard Lightning checkpoint contains the frozen pi0.5 backbone as well as
the trainable LoRA adapters and task heads.  That is useful for exact optimizer
resume, but unnecessarily large for milestone evaluation.  This callback saves
only parameters whose ``requires_grad`` flag is true.  ``trainHydra.py`` already
loads evaluation checkpoints with ``strict=False``, after constructing the base
model from its pretrained weights, so these checkpoints remain directly usable
for evaluation and partial-weight initialization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from lightning import Callback, LightningModule, Trainer


class TrainableStateCheckpoint(Callback):
    """Save a compact, model-only checkpoint at fixed optimizer-step intervals."""

    def __init__(
        self,
        dirpath: str,
        every_n_train_steps: int,
        save_on_train_end: bool = True,
    ) -> None:
        super().__init__()
        if every_n_train_steps <= 0:
            raise ValueError("every_n_train_steps must be positive")
        self.dirpath = Path(dirpath)
        self.every_n_train_steps = int(every_n_train_steps)
        self.save_on_train_end = bool(save_on_train_end)
        self._last_saved_step = -1

    @staticmethod
    def _trainable_state(pl_module: LightningModule) -> dict[str, torch.Tensor]:
        state = pl_module.state_dict()
        # ``ModelWrapper`` exposes some modules through more than one attribute.
        # ``named_parameters`` removes such aliases, preventing duplicate tensor
        # storage while retaining a canonical loadable key for every parameter.
        trainable_names = {
            name for name, parameter in pl_module.named_parameters() if parameter.requires_grad
        }
        trainable_state = {
            name: state[name].detach().cpu().clone()
            for name in sorted(trainable_names)
        }
        if not trainable_state:
            raise RuntimeError("No trainable parameters were found for checkpointing")
        return trainable_state

    def _save(self, trainer: Trainer, pl_module: LightningModule, step: int) -> None:
        if not trainer.is_global_zero or step == self._last_saved_step:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        state = self._trainable_state(pl_module)
        parameter_count = sum(value.numel() for value in state.values())
        filename = f"step_{step:08d}_trainable.ckpt"
        destination = self.dirpath / filename
        temporary = self.dirpath / f".{filename}.tmp"
        payload = {
            "state_dict": state,
            "global_step": int(step),
            "checkpoint_format": "egomimic_trainable_state_v1",
            "trainable_parameter_count": int(parameter_count),
            "trainable_state_keys": sorted(state),
        }
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        metadata = {
            "checkpoint": str(destination),
            "checkpoint_format": payload["checkpoint_format"],
            "global_step": int(step),
            "state_key_count": len(state),
            "trainable_parameter_count": int(parameter_count),
            "size_bytes": destination.stat().st_size,
        }
        metadata_path = destination.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._last_saved_step = step
        print(
            "[TRAINABLE_CHECKPOINT] "
            f"step={step} parameters={parameter_count} path={destination}",
            flush=True,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        step = int(trainer.global_step)
        if step > 0 and step % self.every_n_train_steps == 0:
            self._save(trainer, pl_module, step)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        step = int(trainer.global_step)
        if self.save_on_train_end and step > 0:
            self._save(trainer, pl_module, step)
