"""HPT wrapper for the masked native-rate Expert-Action V3 head."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F

from egomimic.algo.hpt_mixed_clean_v2 import (
    HPTMixedCleanV2,
    attach_keyed_normalization_metadata,
)
from egomimic.rldb.embodiment.embodiment import get_embodiment


class HPTExpertActionV3(HPTMixedCleanV2):
    """Add joint labels to the shared Flow head without an auxiliary head."""

    def _robomimic_to_hpt_data(
        self, batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys=None
    ):
        # The new 15-D head owns the PIPER joint channels, so no separate HPT
        # auxiliary head is constructed.  Joint labels are attached explicitly.
        data = super()._robomimic_to_hpt_data(
            batch, cam_keys, proprio_keys, lang_keys, ac_key, []
        )
        if "actions_joints" in batch:
            data["actions_joints"] = batch["actions_joints"]
            embodiment_id = int(batch["embodiment"].reshape(-1)[0].item())
            stats = self.norm_stats.norm_stats.get(embodiment_id, {}).get(
                "actions_joints"
            )
            if stats is None:
                raise ValueError(
                    f"Expert-Action V3 lacks actions_joints stats for {embodiment_id}"
                )
            attach_keyed_normalization_metadata(
                data,
                prefix="actions_joints",
                mode=self.norm_stats.norm_mode,
                stats=stats,
                reference=batch["actions_joints"],
            )
        return data

    @staticmethod
    def _upsample(values: torch.Tensor, size: int, *, nearest: bool = False):
        if values.shape[1] == size:
            return values
        mode = "nearest" if nearest else "linear"
        kwargs = {} if nearest else {"align_corners": True}
        return F.interpolate(
            values.transpose(1, 2), size=size, mode=mode, **kwargs
        ).transpose(1, 2)

    @torch.no_grad()
    def forward_eval(self, batch):
        outputs = OrderedDict()
        policy = self.nets["policy"]
        head = policy.heads["shared"]
        for embodiment_id, model_batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys[embodiment_id]
            data = self._robomimic_to_hpt_data(
                model_batch,
                self.camera_keys[embodiment_id],
                self.proprio_keys[embodiment_id],
                self.lang_keys[embodiment_id],
                ac_key,
                [],
            )
            loss_batch = {
                "domain": embodiment_name,
                "data": self._clone_batch(data),
            }
            outputs[f"{embodiment_name}_loss"] = policy.compute_loss(loss_batch)
            for name, value in head.last_loss_components.items():
                outputs[f"{embodiment_name}_{name}"] = value

            inference_data = self._clone_batch(data)
            features, _ = policy.forward_features(embodiment_name, inference_data)
            sampled = head.sample_outputs(features, inference_data)
            target_horizon = int(model_batch[ac_key].shape[1])
            outputs[f"{embodiment_name}_{ac_key}"] = self._upsample(
                sampled["pose_raw"], target_horizon
            )
            if "joint_raw" in sampled:
                joint = sampled["joint_raw"]
                joints = self._upsample(joint[..., :6], target_horizon)
                gripper = self._upsample(
                    joint[..., 6:], target_horizon, nearest=True
                )
                outputs[f"{embodiment_name}_actions_joints"] = torch.cat(
                    (joints, gripper), dim=-1
                )
        return outputs

    @torch.no_grad()
    def forward_joint_deploy(self, embodiment_name: str, data: dict, generator=None):
        """Return raw native-rate [B,30,7] PIPER commands for deployment."""
        policy = self.nets["policy"]
        features, _ = policy.forward_features(embodiment_name, data)
        sampled = policy.heads["shared"].sample_outputs(
            features, data, generator=generator
        )
        if "joint_raw" not in sampled:
            raise ValueError("forward_joint_deploy requires a PIPER observation")
        return sampled["joint_raw"]
