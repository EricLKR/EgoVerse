"""Single-frame pi0.5 Expert-Action V4 with bounded PIPER joint residuals."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from egomimic.algo.pi_egoverse_expert_action_v3 import PIEgoVerseExpertActionV3
from egomimic.models.pi05_expert_action_v3 import JOINT_SLICE, POSE_SLICE
from egomimic.models.pi05_expert_action_v4_bounded import (
    Pi05ExpertActionAuxiliaryV4Bounded,
    build_expert_action_target_v4,
)


class PIEgoVerseExpertActionV4Bounded(PIEgoVerseExpertActionV3):
    """Reuse V3 data/Flow plumbing with a checkpointed bounded joint decoder."""

    def __init__(self, *args, auxiliary: Mapping | None = None, **kwargs) -> None:
        auxiliary_cfg = dict(auxiliary or {})
        # Build the unchanged pi0.5 backbone/camera-slot plumbing first.  The
        # temporary V3 head is immediately replaced and is never optimized.
        super().__init__(*args, auxiliary={}, **kwargs)
        expert_dim = int(self.model.action_in_proj.out_features)
        auxiliary_cfg.setdefault("expert_dim", expert_dim)
        auxiliary_cfg.setdefault("action_horizon", 30)
        self.nets["auxiliary"] = Pi05ExpertActionAuxiliaryV4Bounded(
            **auxiliary_cfg
        )
        self.joint_residual_scale_rad = self.nets[
            "auxiliary"
        ].joint_residual_scale_rad_vector

        total = sum(parameter.numel() for parameter in self.nets.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.nets.parameters()
            if parameter.requires_grad
        )
        self.finetune_parameter_stats = {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_percent": 100.0 * trainable / total,
        }

    def _flow_targets(self, model_batch, embodiment_id, pose32, pose_weights):
        current_raw = None
        target_joint_raw = None
        if embodiment_id == self.piper_embodiment_id:
            _, current_raw, target_joint_raw = self._piper_joint_inputs(
                model_batch, embodiment_id
            )
        target32, weights32 = build_expert_action_target_v4(
            pose32,
            pose_weights,
            pose_weight=1.0,
            target_joint_raw=target_joint_raw,
            current_raw=current_raw,
            joint_residual_scale_rad=self.joint_residual_scale_rad,
            joint_weight=1.0,
        )
        pose_mask = torch.zeros_like(weights32)
        pose_mask[..., POSE_SLICE] = weights32[..., POSE_SLICE]
        joint_mask = torch.zeros_like(weights32)
        if target_joint_raw is not None:
            joint_mask[..., JOINT_SLICE] = weights32[..., JOINT_SLICE]
        return target32, pose_mask, joint_mask, current_raw, target_joint_raw
