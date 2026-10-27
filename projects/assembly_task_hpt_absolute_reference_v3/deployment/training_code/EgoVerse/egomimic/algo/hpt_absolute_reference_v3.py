"""HPT Mixed V3 with a second proprio-perturbed recovery forward pass."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch

from egomimic.algo.hpt_mixed_clean_v2 import HPTMixedCleanV2
from egomimic.models.absolute_reference_joint_policy_v3 import (
    AbsoluteReferenceJointPolicyV3,
)
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


class HPTAbsoluteReferenceV3(HPTMixedCleanV2):
    """Adds physical proprio perturbations without changing clean targets."""

    def __init__(
        self,
        *args,
        recovery_noise_std_deg: Sequence[float] = (1.5,) * 6,
        recovery_noise_clip_deg: Sequence[float] = (4.0,) * 6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.recovery_noise_std_deg = self._six_positive(
            recovery_noise_std_deg, "recovery_noise_std_deg"
        )
        self.recovery_noise_clip_deg = self._six_positive(
            recovery_noise_clip_deg, "recovery_noise_clip_deg"
        )

    @staticmethod
    def _six_positive(values: Sequence[float], name: str) -> torch.Tensor:
        result = torch.as_tensor(tuple(values), dtype=torch.float32)
        if result.numel() != 6 or not torch.isfinite(result).all() or (result <= 0).any():
            raise ValueError(f"{name} must contain six finite positive values")
        return result

    def _joint_head(self) -> AbsoluteReferenceJointPolicyV3:
        head_key = "piper_left_arm_actions_joints"
        heads = self.nets["policy"].heads
        # ``heads`` is a torch.nn.ModuleDict, which intentionally does not
        # implement ``dict.get``.  Keep the missing-head error explicit while
        # using ModuleDict's supported membership/subscript API.
        if head_key not in heads:
            raise RuntimeError(
                "Absolute-Reference V3 requires a "
                f"{head_key} policy head"
            )
        head = heads[head_key]
        if not isinstance(head, AbsoluteReferenceJointPolicyV3):
            raise RuntimeError(
                "Absolute-Reference V3 requires piper_left_arm_actions_joints "
                "to be AbsoluteReferenceJointPolicyV3"
            )
        return head

    def _perturb_joint_proprio(
        self, data: dict, head: AbsoluteReferenceJointPolicyV3
    ) -> dict:
        perturbed = self._clone_batch(data)
        normalized = perturbed["state_joints"]
        physical = head._unnormalize(normalized, perturbed, "proprio")
        std = torch.deg2rad(self.recovery_noise_std_deg).to(physical)
        limit = torch.deg2rad(self.recovery_noise_clip_deg).to(physical)
        noise = torch.randn_like(physical[..., :6]) * std
        noise = torch.maximum(torch.minimum(noise, limit), -limit)
        physical = physical.clone()
        physical[..., :6] += noise
        perturbed["state_joints"] = head._normalize(
            physical, perturbed, "proprio"
        )
        perturbed["_recovery_joint_noise_rad"] = noise
        return perturbed

    def forward_training(self, batch):
        predictions = OrderedDict()
        hpt_batches = {}
        self.training_step += 1
        for embodiment_id, raw_batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            cam_keys = self.camera_keys[embodiment_id]
            proprio_keys = self.proprio_keys[embodiment_id]
            lang_keys = self.lang_keys[embodiment_id]
            ac_key = self.ac_keys[embodiment_id]
            aux_ac_keys = self.auxiliary_ac_keys.get(embodiment_name, [])
            data = self._robomimic_to_hpt_data(
                raw_batch, cam_keys, proprio_keys, lang_keys, ac_key, aux_ac_keys
            )
            hpt_batch = {"domain": embodiment_name, "data": data}
            hpt_batches[embodiment_id] = self._clone_batch(hpt_batch)

            recovery_source = self._clone_batch(data)
            if self.freeze_repr:
                loss = self.nets["policy"].compute_loss_depth(
                    hpt_batch, depth=self.freeze_depth
                )
            else:
                loss = self.nets["policy"].compute_loss(hpt_batch)

            if embodiment_name == "piper_left_arm":
                if self.freeze_repr:
                    raise RuntimeError(
                        "Absolute-Reference V3 recovery training does not support freeze_repr"
                    )
                head = self._joint_head()
                if head._last_clean_prediction is None:
                    raise RuntimeError("V3 clean prediction was not captured")
                recovery_data = self._perturb_joint_proprio(recovery_source, head)
                recovery_features, _ = self.nets["policy"].forward_features(
                    embodiment_name, recovery_data
                )
                original_action = recovery_data.get("action")
                try:
                    recovery_data["action"] = recovery_data["actions_joints"]
                    recovery_weighted, _, _ = head.compute_recovery_losses(
                        recovery_features,
                        recovery_data,
                        head._last_clean_prediction,
                    )
                finally:
                    if original_action is not None:
                        recovery_data["action"] = original_action
                auxiliary_weight = self.nets["policy"]._head_loss_weight(
                    embodiment_name, "actions_joints"
                )
                loss = loss + auxiliary_weight * recovery_weighted

            predictions[f"{embodiment_name}_{ac_key}"] = raw_batch[ac_key]
            predictions[f"{embodiment_name}_loss"] = loss
            for component_name, component_value in self._current_head_loss_components(
                embodiment_name
            ).items():
                predictions[f"{embodiment_name}_{component_name}"] = component_value

        if self.ot:
            ot_loss, avg_feat_distance = self._forward_ot(
                hpt_batches,
                get_embodiment_id(self.domains[0]),
                get_embodiment_id(self.domains[1]),
            )
            predictions["ot_loss"] = ot_loss
            predictions["avg_feature_distance"] = avg_feat_distance
        return predictions
