"""Bounded, per-joint residual parameterization for pi0.5 Expert-Action V4.

V4 keeps V3's native 30-step, 32-channel Gemma action-expert contract, but
changes channels 9..14 from an unbounded linear residual into an invertible
latent representation:

    z = atanh((q_target - q_current) / scale)
    q_target = q_current + scale * tanh(z)

The six scales are training-split statistics stored as persistent checkpoint
buffers.  Consequently the exact same decoder is used for training losses,
evaluation, and deployment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from egomimic.models.pi05_expert_action_v3 import (
    ACTION_DIM,
    JOINT_DIMS,
    JOINT_SLICE,
    POSE_SLICE,
    RESERVED_SLICE,
    Pi05ExpertActionAuxiliaryV3,
)


def _scale_tensor(
    scale_rad: torch.Tensor | Sequence[float], reference: torch.Tensor
) -> torch.Tensor:
    scale = torch.as_tensor(scale_rad, dtype=reference.dtype, device=reference.device)
    if scale.ndim != 1 or scale.numel() != JOINT_DIMS:
        raise ValueError("joint residual scale must contain exactly six values")
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("joint residual scales must be finite and positive")
    return scale


def bounded_joint_to_latent(
    target_joint_rad: torch.Tensor,
    current_joint_rad: torch.Tensor,
    scale_rad: torch.Tensor | Sequence[float],
    *,
    strict: bool = True,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Encode physical targets into the V4 atanh residual latent."""
    if target_joint_rad.ndim != 3 or target_joint_rad.shape[-1] != JOINT_DIMS:
        raise ValueError("target_joint_rad must be [B,T,6]")
    if current_joint_rad.shape != (target_joint_rad.shape[0], JOINT_DIMS):
        raise ValueError("current_joint_rad must be [B,6]")
    if not 0.0 < float(epsilon) < 0.1:
        raise ValueError("epsilon must be in (0, 0.1)")
    scale = _scale_tensor(scale_rad, target_joint_rad)
    utilization = (
        target_joint_rad - current_joint_rad[:, None, :]
    ) / scale[None, None, :]
    limit = 1.0 - float(epsilon)
    maximum = utilization.detach().abs().max().item()
    if strict and maximum >= limit:
        raise ValueError(
            "joint residual reaches/exceeds bounded V4 scale: "
            f"max_utilization={maximum:.6f}, limit={limit:.6f}"
        )
    return torch.atanh(utilization.clamp(-limit, limit))


def bounded_joint_from_latent(
    latent: torch.Tensor,
    current_joint_rad: torch.Tensor,
    scale_rad: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Decode V4 joint latents into bounded absolute physical radians."""
    if latent.ndim != 3 or latent.shape[-1] != JOINT_DIMS:
        raise ValueError("latent must be [B,T,6]")
    if current_joint_rad.shape != (latent.shape[0], JOINT_DIMS):
        raise ValueError("current_joint_rad must be [B,6]")
    scale = _scale_tensor(scale_rad, latent)
    return current_joint_rad[:, None, :] + scale[None, None, :] * torch.tanh(latent)


def build_expert_action_target_v4(
    pose_action32: torch.Tensor,
    pose_weights32: torch.Tensor,
    *,
    pose_weight: float,
    target_joint_raw: torch.Tensor | None = None,
    current_raw: torch.Tensor | None = None,
    joint_residual_scale_rad: torch.Tensor | Sequence[float],
    joint_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the V4 action target and exact per-channel Flow loss mask."""
    if pose_action32.ndim != 3 or pose_action32.shape[-1] != ACTION_DIM:
        raise ValueError("pose_action32 must be [B,T,32]")
    if pose_weights32.shape != pose_action32.shape:
        raise ValueError("pose weights must match pose_action32")
    target = torch.zeros_like(pose_action32)
    weights = torch.zeros_like(pose_weights32)
    target[..., POSE_SLICE] = pose_action32[..., POSE_SLICE]
    weights[..., POSE_SLICE] = (
        pose_weights32[..., POSE_SLICE] * float(pose_weight)
    )

    if target_joint_raw is not None:
        if current_raw is None:
            raise ValueError("current_raw is required for PIPER joint targets")
        if target_joint_raw.shape[:2] != target.shape[:2] or target_joint_raw.shape[-1] != 7:
            raise ValueError("target_joint_raw must be [B,T,7] and match pose horizon")
        if current_raw.shape != (target.shape[0], 7):
            raise ValueError("current_raw must be [B,7]")
        target[..., JOINT_SLICE] = bounded_joint_to_latent(
            target_joint_raw[..., :JOINT_DIMS],
            current_raw[..., :JOINT_DIMS],
            joint_residual_scale_rad,
        )
        weights[..., JOINT_SLICE] = float(joint_weight)

    if target[..., RESERVED_SLICE].detach().abs().max().item() != 0.0:
        raise AssertionError("reserved V4 action channels must remain zero")
    if weights[..., RESERVED_SLICE].detach().abs().max().item() != 0.0:
        raise AssertionError("reserved V4 action channels must remain masked")
    return target, weights


class Pi05ExpertActionAuxiliaryV4Bounded(Pi05ExpertActionAuxiliaryV3):
    """V3 auxiliary objectives evaluated through the bounded V4 decoder."""

    contract_name = "egoverse_pi05_expert_action_v4_bounded"
    contract_version = 4

    def __init__(
        self,
        *,
        joint_residual_scale_deg: Sequence[float],
        first_action_loss_weight: float = 1.0,
        latent_saturation_loss_weight: float = 0.01,
        latent_saturation_threshold: float = 0.95,
        **kwargs,
    ) -> None:
        scale_deg = torch.as_tensor(tuple(joint_residual_scale_deg), dtype=torch.float32)
        if scale_deg.ndim != 1 or scale_deg.numel() != JOINT_DIMS:
            raise ValueError("joint_residual_scale_deg must contain J1..J6")
        if not torch.isfinite(scale_deg).all() or (scale_deg <= 0).any():
            raise ValueError("joint residual scales must be finite and positive")
        if float(first_action_loss_weight) < 0:
            raise ValueError("first_action_loss_weight must be non-negative")
        if float(latent_saturation_loss_weight) < 0:
            raise ValueError("latent_saturation_loss_weight must be non-negative")
        if not 0.0 < float(latent_saturation_threshold) < 1.0:
            raise ValueError("latent_saturation_threshold must be in (0,1)")

        # The V3 implementation can be reused after V4 latents are converted
        # to physical delta radians.  A scalar scale of exactly one radian
        # makes its linear decoder an identity on those physical deltas.
        super().__init__(
            joint_residual_scale_deg=math.degrees(1.0),
            **kwargs,
        )
        self.register_buffer(
            "joint_residual_scale_deg_vector", scale_deg, persistent=True
        )
        self.register_buffer(
            "joint_residual_scale_rad_vector", torch.deg2rad(scale_deg), persistent=True
        )
        self.first_action_loss_weight = float(first_action_loss_weight)
        self.latent_saturation_loss_weight = float(latent_saturation_loss_weight)
        self.latent_saturation_threshold = float(latent_saturation_threshold)

    def _physical_delta_action32(
        self, latent_action32: torch.Tensor
    ) -> torch.Tensor:
        converted = latent_action32.clone()
        scale = self.joint_residual_scale_rad_vector.to(converted)
        converted[..., JOINT_SLICE] = (
            scale[None, None, :] * torch.tanh(latent_action32[..., JOINT_SLICE])
        )
        return converted

    def compute_loss(
        self,
        *,
        suffix_hidden: torch.Tensor,
        predicted_clean_action32: torch.Tensor,
        current_raw: torch.Tensor,
        target_joint_raw: torch.Tensor,
        predicted_pose_camera_raw: torch.Tensor | None,
    ) -> torch.Tensor:
        physical_delta_action32 = self._physical_delta_action32(
            predicted_clean_action32
        )
        base = super().compute_loss(
            suffix_hidden=suffix_hidden,
            predicted_clean_action32=physical_delta_action32,
            current_raw=current_raw,
            target_joint_raw=target_joint_raw,
            predicted_pose_camera_raw=predicted_pose_camera_raw,
        )
        predicted_joints = bounded_joint_from_latent(
            predicted_clean_action32[..., JOINT_SLICE],
            current_raw[..., :JOINT_DIMS],
            self.joint_residual_scale_rad_vector,
        )
        target_joints = target_joint_raw[..., :JOINT_DIMS]
        first_action = self._weighted_huber(
            predicted_joints[:, :1] - target_joints[:, :1]
        )
        bounded_fraction = torch.tanh(
            predicted_clean_action32[..., JOINT_SLICE]
        ).abs()
        saturation = F.relu(
            bounded_fraction - self.latent_saturation_threshold
        ).square().mean()
        total = (
            base
            + self.first_action_loss_weight * first_action
            + self.latent_saturation_loss_weight * saturation
        )
        self.last_loss_components.update(
            {
                "first_action": first_action.detach(),
                "latent_saturation": saturation.detach(),
                "max_predicted_utilization": bounded_fraction.detach().max(),
                "auxiliary_total": total.detach(),
            }
        )
        return total

    def decode_actions(
        self,
        *,
        sampled_action32: torch.Tensor,
        suffix_hidden: torch.Tensor,
        current_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        physical_delta_action32 = self._physical_delta_action32(sampled_action32)
        return super().decode_actions(
            sampled_action32=physical_delta_action32,
            suffix_hidden=suffix_hidden,
            current_raw=current_raw,
        )
