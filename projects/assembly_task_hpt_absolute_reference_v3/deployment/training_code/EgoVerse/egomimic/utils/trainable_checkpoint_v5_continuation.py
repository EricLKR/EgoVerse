"""Compact checkpoints for a V5 continuation run.

The continuation trainer starts a fresh optimizer and therefore its Lightning
``global_step`` restarts at zero.  The model's persistent V5 schedule buffer,
however, is restored from the 20k compact checkpoint and continues counting.
This callback adds the completed-step offset when naming and annotating compact
checkpoints so 5k/10k local steps are unambiguously saved as 25k/30k total
steps.  It also verifies that the model schedule and total step agree before
writing a checkpoint.
"""

from __future__ import annotations

from lightning import LightningModule, Trainer

from egomimic.utils.trainable_checkpoint_v5 import (
    TrainableStateAndBuffersCheckpoint,
)


class ContinuationTrainableStateAndBuffersCheckpoint(
    TrainableStateAndBuffersCheckpoint
):
    """Save compact V5 checkpoints using total, rather than local, steps."""

    def __init__(
        self,
        dirpath: str,
        every_n_train_steps: int,
        step_offset: int,
        save_on_train_end: bool = True,
    ) -> None:
        if step_offset < 0:
            raise ValueError("step_offset must be non-negative")
        super().__init__(
            dirpath=dirpath,
            every_n_train_steps=every_n_train_steps,
            save_on_train_end=save_on_train_end,
        )
        self.step_offset = int(step_offset)

    @staticmethod
    def _v5_schedule_step(pl_module: LightningModule) -> int:
        state = pl_module.state_dict()
        step_keys = [
            key for key in state if key.endswith("egoverse_v5_training_step")
        ]
        if len(step_keys) != 1:
            raise RuntimeError(
                "Expected exactly one V5 schedule buffer, got "
                f"{step_keys}"
            )
        return int(state[step_keys[0]].item())

    def _save(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        step: int,
    ) -> None:
        total_step = int(step) + self.step_offset
        schedule_step = self._v5_schedule_step(pl_module)
        if schedule_step != total_step:
            raise RuntimeError(
                "Continuation step mismatch: "
                f"local={step}, offset={self.step_offset}, "
                f"total={total_step}, v5_schedule={schedule_step}"
            )
        super()._save(trainer, pl_module, total_step)
