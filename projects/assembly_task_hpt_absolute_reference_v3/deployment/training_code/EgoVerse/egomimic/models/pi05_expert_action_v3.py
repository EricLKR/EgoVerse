"""pi0.5 Expert-Action V3 policy components.

This module is deliberately additive.  It does not modify MultiHead V2 or any
of the deployed HPT policies.  V3 gives fixed meanings to OpenPI's 32 action
channels and makes the Gemma action expert itself predict the PIPER trajectory:

* channels 0..8: shared camera-frame xyz + rotation-6D pose;
* channels 9..14: PIPER joint residuals relative to the query observation;
* channels 15..31: reserved and always masked;
* binary gripper: a separate classifier over the action-expert hidden states.

All trajectories use the native 30-step control horizon.  There is no 30<->100
interpolation and no IK in this representation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.pi05_egoverse_multihead_v2 import (
    geodesic_angle,
    rotation_6d_to_matrix,
)


ACTION_DIM = 32
POSE_SLICE = slice(0, 9)
JOINT_SLICE = slice(9, 15)
RESERVED_SLICE = slice(15, 32)
POSE_DIMS = 9
JOINT_DIMS = 6


def clean_action_from_velocity(
    noisy_action: torch.Tensor,
    time: torch.Tensor,
    predicted_velocity: torch.Tensor,
) -> torch.Tensor:
    """Recover the clean-action estimate implied by rectified flow.

    Training constructs ``x_t = action + t * (noise - action)`` and the model
    predicts ``noise - action``.  Therefore ``action_hat = x_t - t * v_hat``.
    """
    if noisy_action.shape != predicted_velocity.shape:
        raise ValueError("noisy_action and predicted_velocity must have equal shapes")
    if time.ndim != 1 or time.shape[0] != noisy_action.shape[0]:
        raise ValueError("time must be [B]")
    return noisy_action - time[:, None, None] * predicted_velocity


def joint_residual_to_flow(
    target_joint_rad: torch.Tensor,
    current_joint_rad: torch.Tensor,
    scale_rad: float,
    *,
    strict: bool = True,
    tolerance: float = 1e-5,
) -> torch.Tensor:
    """Encode physical joint targets as query-relative, dimensionless flow data."""
    if target_joint_rad.ndim != 3 or target_joint_rad.shape[-1] != JOINT_DIMS:
        raise ValueError("target_joint_rad must be [B,T,6]")
    if current_joint_rad.shape != (target_joint_rad.shape[0], JOINT_DIMS):
        raise ValueError("current_joint_rad must be [B,6]")
    if not math.isfinite(scale_rad) or scale_rad <= 0:
        raise ValueError("scale_rad must be finite and positive")
    residual = (target_joint_rad - current_joint_rad[:, None, :]) / scale_rad
    if strict and residual.detach().abs().max().item() > 1.0 + tolerance:
        maximum = residual.detach().abs().max().item()
        raise ValueError(
            f"joint residual exceeds configured flow scale: max={maximum:.6f}"
        )
    return residual


def joint_residual_from_flow(
    normalized_residual: torch.Tensor,
    current_joint_rad: torch.Tensor,
    scale_rad: float,
) -> torch.Tensor:
    """Decode V3 joint channels into absolute physical joint radians."""
    if normalized_residual.ndim != 3 or normalized_residual.shape[-1] != JOINT_DIMS:
        raise ValueError("normalized_residual must be [B,T,6]")
    if current_joint_rad.shape != (normalized_residual.shape[0], JOINT_DIMS):
        raise ValueError("current_joint_rad must be [B,6]")
    return current_joint_rad[:, None, :] + float(scale_rad) * normalized_residual


def build_expert_action_target(
    pose_action32: torch.Tensor,
    pose_weights32: torch.Tensor,
    *,
    pose_weight: float,
    target_joint_raw: torch.Tensor | None = None,
    current_raw: torch.Tensor | None = None,
    joint_residual_scale_rad: float = math.pi / 2,
    joint_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the fixed semantic V3 target and exact per-channel loss mask.

    Human samples omit ``target_joint_raw`` and receive zero loss/gradient on
    channels 9..31.  PIPER targets are expected to contain six radians plus a
    binary gripper channel; the gripper is intentionally not copied into flow.
    """
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
        target[..., JOINT_SLICE] = joint_residual_to_flow(
            target_joint_raw[..., :JOINT_DIMS],
            current_raw[..., :JOINT_DIMS],
            joint_residual_scale_rad,
        )
        weights[..., JOINT_SLICE] = float(joint_weight)

    if target[..., RESERVED_SLICE].detach().abs().max().item() != 0.0:
        raise AssertionError("reserved V3 action channels must remain zero")
    if weights[..., RESERVED_SLICE].detach().abs().max().item() != 0.0:
        raise AssertionError("reserved V3 action channels must remain masked")
    return target, weights


class Pi05ExpertActionAuxiliaryV3(nn.Module):
    """Native-rate trajectory and binary-gripper losses over expert outputs."""

    contract_name = "egoverse_pi05_expert_action_v3"
    contract_version = 3

    def __init__(
        self,
        expert_dim: int = 1024,
        gripper_hidden_dim: int = 256,
        action_horizon: int = 30,
        joint_residual_scale_deg: float = 150.0,
        joint_error_scale_deg: float = 5.0,
        joint_position_loss_weight: float = 1.0,
        joint_delta_loss_weight: float = 0.25,
        joint_acceleration_loss_weight: float = 0.10,
        joint_jerk_loss_weight: float = 0.02,
        camera_position_loss_weight: float = 0.10,
        camera_rotation_loss_weight: float = 0.05,
        camera_position_error_scale_m: float = 0.02,
        rotation_error_scale_deg: float = 10.0,
        gripper_loss_weight: float = 1.5,
        gripper_closed_weight: float = 1.76,
        gripper_open_weight: float = 0.70,
        gripper_transition_weight: float = 3.0,
        gripper_transition_radius: int = 2,
        gripper_open_prior: float = 0.716,
        joint_dimension_weights: tuple[float, ...] | list[float] | None = None,
        enable_camera_consistency: bool = True,
    ) -> None:
        super().__init__()
        self.action_horizon = int(action_horizon)
        if self.action_horizon != 30:
            raise ValueError("Expert-Action V3 requires the native 30-step horizon")
        self.joint_residual_scale_rad = math.radians(float(joint_residual_scale_deg))
        self.joint_error_scale_rad = math.radians(float(joint_error_scale_deg))
        self.joint_position_loss_weight = float(joint_position_loss_weight)
        self.joint_delta_loss_weight = float(joint_delta_loss_weight)
        self.joint_acceleration_loss_weight = float(joint_acceleration_loss_weight)
        self.joint_jerk_loss_weight = float(joint_jerk_loss_weight)
        self.camera_position_loss_weight = float(camera_position_loss_weight)
        self.camera_rotation_loss_weight = float(camera_rotation_loss_weight)
        self.camera_position_error_scale_m = float(camera_position_error_scale_m)
        self.rotation_error_scale_rad = math.radians(float(rotation_error_scale_deg))
        self.gripper_loss_weight = float(gripper_loss_weight)
        self.gripper_closed_weight = float(gripper_closed_weight)
        self.gripper_open_weight = float(gripper_open_weight)
        self.gripper_transition_weight = float(gripper_transition_weight)
        self.gripper_transition_radius = int(gripper_transition_radius)
        self.enable_camera_consistency = bool(enable_camera_consistency)

        self.gripper_head = nn.Sequential(
            nn.LayerNorm(int(expert_dim)),
            nn.Linear(int(expert_dim), int(gripper_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(gripper_hidden_dim), 1),
        )
        nn.init.zeros_(self.gripper_head[-1].weight)
        nn.init.constant_(
            self.gripper_head[-1].bias,
            math.log(gripper_open_prior / (1.0 - gripper_open_prior)),
        )

        if joint_dimension_weights is None:
            joint_dimension_weights = [1.0] * JOINT_DIMS
        if len(joint_dimension_weights) != JOINT_DIMS or any(
            float(value) <= 0 for value in joint_dimension_weights
        ):
            raise ValueError("joint_dimension_weights must contain six positive values")
        dimension_weights = torch.as_tensor(joint_dimension_weights, dtype=torch.float32)
        self.register_buffer(
            "joint_dimension_weights",
            dimension_weights / dimension_weights.mean(),
            persistent=True,
        )

        self._physical_fk_chain = None
        if self.enable_camera_consistency:
            from egomimic.rldb.embodiment.piper import Piper

            if not Piper.EXTRINSICS:
                raise ValueError("Expert-Action V3 requires calibrated Piper.EXTRINSICS")
            world_from_camera = np.asarray(next(iter(Piper.EXTRINSICS.values())))
            if world_from_camera.shape != (4, 4):
                raise ValueError("PIPER front-camera extrinsics must be 4x4")
            camera_from_world = torch.from_numpy(
                np.linalg.inv(world_from_camera)
            ).float()
        else:
            camera_from_world = torch.eye(4, dtype=torch.float32)
        self.register_buffer("camera_from_world", camera_from_world, persistent=True)
        self.last_loss_components: dict[str, torch.Tensor] = {}

    def gripper_logits(self, suffix_hidden: torch.Tensor) -> torch.Tensor:
        if suffix_hidden.ndim != 3 or suffix_hidden.shape[1] != self.action_horizon:
            raise ValueError("suffix_hidden must be [B,30,D]")
        return self.gripper_head(suffix_hidden.float()).squeeze(-1)

    def _weighted_huber(self, error: torch.Tensor) -> torch.Tensor:
        scaled = error / self.joint_error_scale_rad
        values = F.smooth_l1_loss(
            scaled, torch.zeros_like(scaled), reduction="none", beta=1.0
        )
        weights = self.joint_dimension_weights.to(values)[None, None]
        return (values * weights).sum() / (
            values.shape[0] * values.shape[1] * weights.sum()
        )

    def _trajectory_losses(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = []
        for order in range(4):
            predicted_order = prediction
            target_order = target
            for _ in range(order):
                predicted_order = torch.diff(predicted_order, dim=1)
                target_order = torch.diff(target_order, dim=1)
            outputs.append(self._weighted_huber(predicted_order - target_order))
        return tuple(outputs)

    def _gripper_loss(self, logits: torch.Tensor, target_open: torch.Tensor) -> torch.Tensor:
        class_weights = torch.where(
            target_open,
            torch.full_like(logits, self.gripper_open_weight),
            torch.full_like(logits, self.gripper_closed_weight),
        )
        transitions = torch.zeros_like(logits)
        transitions[:, 1:] = (target_open[:, 1:] != target_open[:, :-1]).to(logits)
        if self.gripper_transition_radius:
            transitions = F.max_pool1d(
                transitions[:, None],
                kernel_size=2 * self.gripper_transition_radius + 1,
                stride=1,
                padding=self.gripper_transition_radius,
            ).squeeze(1)
        weights = class_weights * (
            1.0 + (self.gripper_transition_weight - 1.0) * transitions
        )
        values = F.binary_cross_entropy_with_logits(
            logits, target_open.to(logits), reduction="none"
        )
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    def _get_fk_chain(self, reference: torch.Tensor):
        if self._physical_fk_chain is None:
            import pytorch_kinematics as pk

            from egomimic.robot.piper.piper_kinematics import DEFAULT_URDF_PATH

            with open(DEFAULT_URDF_PATH, "rb") as urdf_file:
                self._physical_fk_chain = pk.build_serial_chain_from_urdf(
                    urdf_file.read(),
                    end_link_name="gripper_base",
                    root_link_name="base_link",
                )
        self._physical_fk_chain = self._physical_fk_chain.to(
            device=reference.device, dtype=reference.dtype
        )
        return self._physical_fk_chain

    def _fk(self, joints: torch.Tensor) -> torch.Tensor:
        leading = joints.shape[:-1]
        with torch.autocast(device_type=joints.device.type, enabled=False):
            matrices = self._get_fk_chain(joints.float()).forward_kinematics(
                joints.float().reshape(-1, JOINT_DIMS)
            ).get_matrix()
        return matrices.reshape(*leading, 4, 4)

    def compute_loss(
        self,
        *,
        suffix_hidden: torch.Tensor,
        predicted_clean_action32: torch.Tensor,
        current_raw: torch.Tensor,
        target_joint_raw: torch.Tensor,
        predicted_pose_camera_raw: torch.Tensor | None,
    ) -> torch.Tensor:
        if predicted_clean_action32.shape[1:] != (self.action_horizon, ACTION_DIM):
            raise ValueError("predicted clean action must be [B,30,32]")
        if target_joint_raw.shape[1:] != (self.action_horizon, 7):
            raise ValueError("PIPER joint target must be [B,30,7]")
        if current_raw.shape != (target_joint_raw.shape[0], 7):
            raise ValueError("current_raw must be [B,7]")

        predicted_joints = joint_residual_from_flow(
            predicted_clean_action32[..., JOINT_SLICE],
            current_raw[..., :JOINT_DIMS],
            self.joint_residual_scale_rad,
        )
        target_joints = target_joint_raw[..., :JOINT_DIMS]
        position, delta, acceleration, jerk = self._trajectory_losses(
            predicted_joints, target_joints
        )
        logits = self.gripper_logits(suffix_hidden)
        gripper = self._gripper_loss(logits, target_joint_raw[..., 6] >= 0.5)

        zero = position.new_zeros(())
        camera_position = zero
        camera_rotation = zero
        if self.enable_camera_consistency:
            if predicted_pose_camera_raw is None:
                raise ValueError("camera consistency requires the predicted physical pose")
            if predicted_pose_camera_raw.shape[1:] != (self.action_horizon, POSE_DIMS):
                raise ValueError("predicted physical pose must be [B,30,9]")
            predicted_fk = self._fk(predicted_joints)
            predicted_camera = self.camera_from_world.to(predicted_fk)[None, None] @ predicted_fk
            position_error = torch.linalg.vector_norm(
                predicted_camera[..., :3, 3] - predicted_pose_camera_raw[..., :3],
                dim=-1,
            )
            camera_position = F.smooth_l1_loss(
                position_error / self.camera_position_error_scale_m,
                torch.zeros_like(position_error),
                reduction="mean",
                beta=1.0,
            )
            target_rotation = rotation_6d_to_matrix(
                predicted_pose_camera_raw[..., 3:9]
            )
            rotation_error = geodesic_angle(
                predicted_camera[..., :3, :3], target_rotation
            )
            camera_rotation = F.smooth_l1_loss(
                rotation_error / self.rotation_error_scale_rad,
                torch.zeros_like(rotation_error),
                reduction="mean",
                beta=1.0,
            )

        total = (
            self.joint_position_loss_weight * position
            + self.joint_delta_loss_weight * delta
            + self.joint_acceleration_loss_weight * acceleration
            + self.joint_jerk_loss_weight * jerk
            + self.camera_position_loss_weight * camera_position
            + self.camera_rotation_loss_weight * camera_rotation
            + self.gripper_loss_weight * gripper
        )
        self.last_loss_components = {
            "joint_position": position.detach(),
            "joint_delta": delta.detach(),
            "joint_acceleration": acceleration.detach(),
            "joint_jerk": jerk.detach(),
            "camera_position": camera_position.detach(),
            "camera_rotation": camera_rotation.detach(),
            "gripper_bce": gripper.detach(),
            "auxiliary_total": total.detach(),
        }
        return total

    def decode_actions(
        self,
        *,
        sampled_action32: torch.Tensor,
        suffix_hidden: torch.Tensor,
        current_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return native 30-step physical six-joint + binary-gripper actions."""
        if sampled_action32.shape[1:] != (self.action_horizon, ACTION_DIM):
            raise ValueError("sampled_action32 must be [B,30,32]")
        joints = joint_residual_from_flow(
            sampled_action32[..., JOINT_SLICE],
            current_raw[..., :JOINT_DIMS],
            self.joint_residual_scale_rad,
        )
        logits = self.gripper_logits(suffix_hidden)
        gripper = (logits >= 0.0).to(joints)[..., None]
        return torch.cat((joints, gripper), dim=-1), logits
