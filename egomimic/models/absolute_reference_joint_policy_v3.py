"""Absolute-reference plus bounded-correction PIPER joint head.

This V3 head removes the deployment-sensitive ``q_current + residual`` output
contract.  Current proprio is still part of the HPT observation, but the
physical joint trajectory is decoded around a checkpointed, train-only
absolute reference range.  A second, much smaller branch supplies bounded
corrections and is the only branch optimized by recovery-only losses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from egomimic.models.residual_joint_policy_mixed_clean_v2 import (
    ResidualJointPolicyMixedCleanV2,
)


class AbsoluteReferenceJointPolicyV3(ResidualJointPolicyMixedCleanV2):
    contract_name = "piper_absolute_reference_joint_v3"
    contract_version = 3
    requires_data_for_inference = True

    def __init__(
        self,
        *args,
        joint_reference_center_deg: Sequence[float],
        joint_reference_range_deg: Sequence[float],
        joint_correction_scale_deg: Sequence[float],
        reference_position_loss_weight: float = 0.50,
        correction_regularization_weight: float = 0.02,
        recovery_loss_weight: float = 0.50,
        replan_consistency_loss_weight: float = 0.25,
        **kwargs,
    ) -> None:
        # The inherited scalar residual module is constructed only to preserve
        # all audited normalization, FK, smoothness, and gripper utilities. It
        # is removed immediately and is never used by this contract.
        super().__init__(*args, **kwargs)
        del self.joint_residual_head
        hidden_dim = self.context_projection.out_features
        self.joint_reference_head = torch.nn.Linear(hidden_dim, self.joint_dims)
        self.joint_correction_head = torch.nn.Linear(hidden_dim, self.joint_dims)

        center = self._joint_vector(joint_reference_center_deg, "center")
        reference_range = self._joint_vector(
            joint_reference_range_deg, "reference range", strictly_positive=True
        )
        correction_scale = self._joint_vector(
            joint_correction_scale_deg, "correction scale", strictly_positive=True
        )
        self.register_buffer(
            "joint_reference_center_rad", torch.deg2rad(center), persistent=True
        )
        self.register_buffer(
            "joint_reference_range_rad", torch.deg2rad(reference_range), persistent=True
        )
        self.register_buffer(
            "joint_correction_scale_rad", torch.deg2rad(correction_scale), persistent=True
        )

        self.reference_position_loss_weight = float(reference_position_loss_weight)
        self.correction_regularization_weight = float(
            correction_regularization_weight
        )
        self.recovery_loss_weight = float(recovery_loss_weight)
        self.replan_consistency_loss_weight = float(
            replan_consistency_loss_weight
        )
        weights = (
            self.reference_position_loss_weight,
            self.correction_regularization_weight,
            self.recovery_loss_weight,
            self.replan_consistency_loss_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("V3 auxiliary loss weights must be non-negative")

        self._last_clean_prediction: torch.Tensor | None = None
        self.reset_output_prior()

    def _joint_vector(
        self,
        values: Sequence[float],
        name: str,
        *,
        strictly_positive: bool = False,
    ) -> torch.Tensor:
        vector = torch.as_tensor(tuple(values), dtype=torch.float32)
        if vector.numel() != self.joint_dims or not torch.isfinite(vector).all():
            raise ValueError(f"joint reference {name} must contain six finite values")
        if strictly_positive and (vector <= 0).any():
            raise ValueError(f"joint reference {name} must be strictly positive")
        return vector

    def reset_output_prior(self) -> None:
        """Restore deterministic center/correction and binary-open priors."""
        # During the parent constructor only the legacy module exists.
        if hasattr(self, "joint_residual_head"):
            torch.nn.init.zeros_(self.joint_residual_head.weight)
            torch.nn.init.zeros_(self.joint_residual_head.bias)
        if hasattr(self, "joint_reference_head"):
            torch.nn.init.zeros_(self.joint_reference_head.weight)
            torch.nn.init.zeros_(self.joint_reference_head.bias)
        if hasattr(self, "joint_correction_head"):
            torch.nn.init.zeros_(self.joint_correction_head.weight)
            torch.nn.init.zeros_(self.joint_correction_head.bias)
        if hasattr(self, "gripper_head"):
            torch.nn.init.zeros_(self.gripper_head.weight)
            torch.nn.init.constant_(
                self.gripper_head.bias,
                math.log(self.gripper_open_prior / (1.0 - self.gripper_open_prior)),
            )

    def _decode_components(
        self, global_cond
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._unwrap_features(global_cond)
        memory = self.context_projection(features)
        queries = self.control_queries.expand(features.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        center = self.joint_reference_center_rad.to(decoded)[None, None]
        reference_range = self.joint_reference_range_rad.to(decoded)[None, None]
        correction_scale = self.joint_correction_scale_rad.to(decoded)[None, None]
        reference = center + reference_range * torch.tanh(
            self.joint_reference_head(decoded)
        )
        correction = correction_scale * torch.tanh(
            self.joint_correction_head(decoded)
        )
        gripper_logits = self.gripper_head(decoded).squeeze(-1)
        return reference, correction, gripper_logits

    def _raw_control_prediction(
        self, global_cond, data: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference, correction, gripper_logits = self._decode_components(global_cond)
        return reference + correction, gripper_logits

    def compute_loss(self, global_cond, data: dict) -> torch.Tensor:
        self._validate_data_contract(data, require_action=True)
        reference, correction, gripper_logits = self._decode_components(global_cond)
        predicted_joints = reference + correction
        targets = self._raw_control_targets(data)
        target_joints = targets[..., : self.joint_dims]
        target_open = targets[..., self.joint_dims] >= 0.5

        position, delta, acceleration, jerk = self._joint_losses(
            predicted_joints, target_joints
        )
        reference_position = self._weighted_huber(reference - target_joints)
        correction_fraction = correction / self.joint_correction_scale_rad.to(
            correction
        )[None, None]
        correction_regularization = correction_fraction.square().mean()
        gripper = self._gripper_loss(gripper_logits, target_open)
        fk_position, fk_rotation = self._fk_losses(predicted_joints, target_joints)

        total = (
            self.joint_position_loss_weight * position
            + self.joint_delta_loss_weight * delta
            + self.joint_acceleration_loss_weight * acceleration
            + self.joint_jerk_loss_weight * jerk
            + self.fk_position_loss_weight * fk_position
            + self.fk_rotation_loss_weight * fk_rotation
            + self.gripper_loss_weight * gripper
            + self.reference_position_loss_weight * reference_position
            + self.correction_regularization_weight * correction_regularization
        )
        self._last_clean_prediction = predicted_joints
        self.last_loss_components = {
            "absolute_v3_joint_position_loss": position.detach(),
            "absolute_v3_joint_delta_loss": delta.detach(),
            "absolute_v3_joint_acceleration_loss": acceleration.detach(),
            "absolute_v3_joint_jerk_loss": jerk.detach(),
            "absolute_v3_fk_position_loss": fk_position.detach(),
            "absolute_v3_fk_rotation_loss": fk_rotation.detach(),
            "absolute_v3_gripper_bce": gripper.detach(),
            "absolute_v3_reference_position_loss": reference_position.detach(),
            "absolute_v3_correction_regularization": correction_regularization.detach(),
            "absolute_v3_nominal_total_loss": total.detach(),
        }
        return total

    def compute_recovery_losses(
        self,
        recovery_global_cond,
        data: dict,
        clean_prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train the correction path under perturbed proprio observations."""
        self._validate_data_contract(data, require_action=True)
        reference, correction, _ = self._decode_components(recovery_global_cond)
        # Recovery-only gradients must teach the bounded correction rather than
        # move the absolute reference trajectory itself.
        recovered = reference.detach() + correction
        target_joints = self._raw_control_targets(data)[..., : self.joint_dims]
        recovery = self._weighted_huber(recovered - target_joints)
        consistency = self._weighted_huber(
            recovered - clean_prediction.detach()
        )
        weighted = (
            self.recovery_loss_weight * recovery
            + self.replan_consistency_loss_weight * consistency
        )
        self.last_loss_components.update(
            {
                "absolute_v3_recovery_loss": recovery.detach(),
                "absolute_v3_replan_consistency_loss": consistency.detach(),
                "absolute_v3_recovery_weighted_loss": weighted.detach(),
            }
        )
        return weighted, recovery, consistency

