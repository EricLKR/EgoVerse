"""Isolated HPT wrapper for Mixed Clean V2 robot-human cotraining.

The shared action is a 9D camera-frame pose, while the deployable PIPER
auxiliary action is a 7D joint/gripper trajectory.  Base HPT historically
attached only the shared action normalizer under ``_action_norm_*``.  That is
not sufficient for a physical-space residual joint head: it must receive the
normalizer belonging to ``actions_joints`` rather than the 9D pose normalizer.

This subclass preserves the base data path and adds action-key-specific
normalization metadata without changing the existing HPT implementation.
"""

from __future__ import annotations

import torch

from egomimic.algo.hpt import HPT


_NORM_FIELDS = {
    "quantile": ("quantile_1", "quantile_99"),
    "minmax": ("min", "max"),
    "zscore": ("mean", "std"),
}


def attach_keyed_normalization_metadata(
    data: dict,
    *,
    prefix: str,
    mode: str,
    stats: dict,
    reference: torch.Tensor,
) -> None:
    """Attach one action key's normalization statistics to an HPT data dict."""
    if mode not in _NORM_FIELDS:
        raise ValueError(f"unsupported normalization mode {mode!r}")
    missing = [field for field in _NORM_FIELDS[mode] if field not in stats]
    if missing:
        raise ValueError(f"normalization stats for {prefix!r} lack {missing}")
    data[f"_{prefix}_norm_mode"] = mode
    for field in _NORM_FIELDS[mode]:
        data[f"_{prefix}_norm_{field}"] = torch.as_tensor(
            stats[field], device=reference.device, dtype=reference.dtype
        )


class HPTMixedCleanV2(HPT):
    """HPT with separate shared-pose and auxiliary-joint normalizers."""

    def _robomimic_to_hpt_data(
        self, batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys=None
    ):
        aux_ac_keys = list(aux_ac_keys or [])
        data = super()._robomimic_to_hpt_data(
            batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys
        )
        embodiment_id = int(batch["embodiment"].reshape(-1)[0].item())
        embodiment_stats = self.norm_stats.norm_stats.get(embodiment_id, {})
        for action_key in aux_ac_keys:
            if action_key not in batch:
                raise KeyError(
                    f"Mixed Clean V2 auxiliary action {action_key!r} is absent from batch"
                )
            stats = embodiment_stats.get(action_key)
            if stats is None:
                raise ValueError(
                    f"Mixed Clean V2 lacks normalization stats for embodiment "
                    f"{embodiment_id}, key {action_key!r}"
                )
            attach_keyed_normalization_metadata(
                data,
                prefix=action_key,
                mode=self.norm_stats.norm_mode,
                stats=stats,
                reference=batch[action_key],
            )
        return data
