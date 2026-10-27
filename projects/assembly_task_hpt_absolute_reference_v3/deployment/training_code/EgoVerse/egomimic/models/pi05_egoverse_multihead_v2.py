"""EgoVerse + pi0.5 MultiHead V2 policy components.

This module is intentionally additive: existing PI, Clean V2, and deployment
implementations are not modified.  It defines the PIPER-only deterministic
joint head used beside the shared 9-D camera-frame pi0.5 pose head.

The joint head predicts 30 native control-rate residual waypoints and upsamples
them to the repository's 100-point action interface.  The gripper is a separate
binary classifier.  Physical FK and camera-frame consistency losses make the
7-D robot output agree with the 9-D shared pose target without using IK.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_btchw(value: torch.Tensor) -> torch.Tensor:
    """Wrap one BCHW/BHWC frame as T=1; also accept legacy BTCHW/BTHWC."""
    if value.ndim == 4:
        if value.shape[1] in (1, 3):
            return value[:, None]
        if value.shape[-1] in (1, 3):
            return value.permute(0, 3, 1, 2).contiguous()[:, None]
    if value.ndim != 5:
        raise ValueError(f"expected image [B,C,H,W] or [B,T,C,H,W], got {tuple(value.shape)}")
    if value.shape[2] in (1, 3):
        return value
    if value.shape[-1] in (1, 3):
        return value.permute(0, 1, 4, 2, 3).contiguous()
    raise ValueError(f"cannot locate RGB channel in {tuple(value.shape)}")


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Convert the first two rotation-matrix columns to a proper SO(3) matrix."""
    if rotation.shape[-1] != 6:
        raise ValueError(f"expected rotation-6D, got {tuple(rotation.shape)}")
    first = F.normalize(rotation[..., 0:3], dim=-1, eps=1e-8)
    second_raw = rotation[..., 3:6]
    second = second_raw - (second_raw * first).sum(-1, keepdim=True) * first
    second = F.normalize(second, dim=-1, eps=1e-8)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def geodesic_angle(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = prediction.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    skew = relative - relative.transpose(-1, -2)
    sine = torch.linalg.vector_norm(skew, dim=(-2, -1)) / (2.0 * math.sqrt(2.0))
    return torch.atan2(sine, cosine)


class Pi05PrefixJointHeadV2(nn.Module):
    """Deterministic residual PIPER head conditioned on pi0.5 prefix tokens.

    Args:
        prefix_dim: PaliGemma prefix hidden width (2048 for ``gemma_2b``).
        state_dim: six joint angles plus one measured gripper value.
        action_horizon: outward-facing resampled trajectory length.
        control_horizon: native 30 Hz waypoints predicted before interpolation.
    """

    contract_name = "egoverse_pi05_multihead_v2"
    contract_version = 2

    def __init__(
        self,
        prefix_dim: int = 2048,
        state_dim: int = 7,
        hidden_dim: int = 256,
        action_horizon: int = 100,
        control_horizon: int = 30,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_residual_deg: float = 90.0,
        joint_error_scale_deg: float = 5.0,
        joint_position_loss_weight: float = 1.0,
        joint_delta_loss_weight: float = 0.25,
        joint_acceleration_loss_weight: float = 0.10,
        joint_jerk_loss_weight: float = 0.02,
        fk_position_loss_weight: float = 0.10,
        fk_rotation_loss_weight: float = 0.05,
        camera_position_loss_weight: float = 0.10,
        camera_rotation_loss_weight: float = 0.05,
        fk_position_error_scale_m: float = 0.02,
        rotation_error_scale_deg: float = 10.0,
        gripper_loss_weight: float = 1.5,
        gripper_closed_weight: float = 1.76,
        gripper_open_weight: float = 0.70,
        gripper_transition_weight: float = 3.0,
        gripper_transition_radius: int = 2,
        gripper_open_prior: float = 0.716,
        joint_dimension_weights: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.joint_dims = self.state_dim - 1
        self.action_horizon = int(action_horizon)
        self.control_horizon = int(control_horizon)
        if self.state_dim != 7:
            raise ValueError("MultiHead V2 currently requires six joints + gripper")
        if not 0.0 < gripper_open_prior < 1.0:
            raise ValueError("gripper_open_prior must lie in (0, 1)")

        self.prefix_projection = nn.Linear(prefix_dim, hidden_dim)
        self.state_projection = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.control_queries = nn.Parameter(
            torch.empty(1, self.control_horizon, hidden_dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.joint_residual_head = nn.Linear(hidden_dim, self.joint_dims)
        self.gripper_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.control_queries, std=0.02)
        nn.init.zeros_(self.joint_residual_head.weight)
        nn.init.zeros_(self.joint_residual_head.bias)
        nn.init.zeros_(self.gripper_head.weight)
        nn.init.constant_(
            self.gripper_head.bias,
            math.log(gripper_open_prior / (1.0 - gripper_open_prior)),
        )

        self.max_residual_rad = math.radians(float(max_residual_deg))
        self.joint_error_scale_rad = math.radians(float(joint_error_scale_deg))
        self.joint_position_loss_weight = float(joint_position_loss_weight)
        self.joint_delta_loss_weight = float(joint_delta_loss_weight)
        self.joint_acceleration_loss_weight = float(joint_acceleration_loss_weight)
        self.joint_jerk_loss_weight = float(joint_jerk_loss_weight)
        self.fk_position_loss_weight = float(fk_position_loss_weight)
        self.fk_rotation_loss_weight = float(fk_rotation_loss_weight)
        self.camera_position_loss_weight = float(camera_position_loss_weight)
        self.camera_rotation_loss_weight = float(camera_rotation_loss_weight)
        self.fk_position_error_scale_m = float(fk_position_error_scale_m)
        self.rotation_error_scale_rad = math.radians(float(rotation_error_scale_deg))
        self.gripper_loss_weight = float(gripper_loss_weight)
        self.gripper_closed_weight = float(gripper_closed_weight)
        self.gripper_open_weight = float(gripper_open_weight)
        self.gripper_transition_weight = float(gripper_transition_weight)
        self.gripper_transition_radius = int(gripper_transition_radius)

        if joint_dimension_weights is None:
            joint_dimension_weights = [1.0] * self.joint_dims
        if len(joint_dimension_weights) != self.joint_dims or any(
            float(weight) <= 0.0 for weight in joint_dimension_weights
        ):
            raise ValueError("joint_dimension_weights must contain six positive values")
        dimension_weights = torch.as_tensor(joint_dimension_weights, dtype=torch.float32)
        self.register_buffer(
            "joint_dimension_weights",
            dimension_weights / dimension_weights.mean(),
            persistent=True,
        )

        from egomimic.rldb.embodiment.piper import Piper

        if not Piper.EXTRINSICS:
            raise ValueError("MultiHead V2 requires calibrated Piper.EXTRINSICS")
        world_from_camera = np.asarray(next(iter(Piper.EXTRINSICS.values())))
        if world_from_camera.shape != (4, 4):
            raise ValueError("Piper front-camera extrinsics must be a 4x4 matrix")
        self.register_buffer(
            "camera_from_world",
            torch.from_numpy(np.linalg.inv(world_from_camera)).float(),
            persistent=True,
        )
        self._physical_fk_chain = None
        self.last_loss_components: dict[str, torch.Tensor] = {}

    def _decode(
        self,
        prefix_features: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        normalized_state_history: torch.Tensor,
        current_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prefix_features.ndim != 3 or prefix_pad_mask.ndim != 2:
            raise ValueError("prefix features/mask must be [B,N,D] and [B,N]")
        if prefix_features.shape[:2] != prefix_pad_mask.shape:
            raise ValueError("prefix feature and mask token axes do not match")
        state = normalized_state_history.reshape(normalized_state_history.shape[0], -1, self.state_dim)
        state_memory = self.state_projection(state)
        memory = torch.cat((self.prefix_projection(prefix_features.float()), state_memory), dim=1)
        state_mask = torch.ones(
            state.shape[:2], dtype=torch.bool, device=prefix_pad_mask.device
        )
        memory_mask = torch.cat((prefix_pad_mask.bool(), state_mask), dim=1)
        state_context = state_memory[:, -1:, :]
        queries = self.control_queries.expand(memory.shape[0], -1, -1) + state_context
        decoded = self.decoder(
            queries,
            memory,
            memory_key_padding_mask=~memory_mask,
        )
        residual = torch.tanh(self.joint_residual_head(decoded)) * self.max_residual_rad
        predicted_joints = current_raw[:, None, : self.joint_dims] + residual
        return predicted_joints, self.gripper_head(decoded).squeeze(-1)

    def _control_indices(self, device: torch.device) -> torch.Tensor:
        return torch.linspace(
            0, self.action_horizon - 1, self.control_horizon, device=device
        ).round().long()

    def _weighted_huber(self, error: torch.Tensor) -> torch.Tensor:
        scaled = error / self.joint_error_scale_rad
        loss = F.smooth_l1_loss(
            scaled, torch.zeros_like(scaled), reduction="none", beta=1.0
        )
        weights = self.joint_dimension_weights.to(loss)[None, None, :]
        return (loss * weights).sum() / (
            loss.shape[0] * loss.shape[1] * weights.sum()
        )

    def _trajectory_losses(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        losses = []
        for order in range(4):
            pred_order, target_order = prediction, target
            for _ in range(order):
                pred_order = torch.diff(pred_order, dim=1)
                target_order = torch.diff(target_order, dim=1)
            losses.append(self._weighted_huber(pred_order - target_order))
        return tuple(losses)

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
                joints.float().reshape(-1, self.joint_dims)
            ).get_matrix()
        return matrices.reshape(*leading, 4, 4)

    def compute_loss(
        self,
        *,
        prefix_features: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        normalized_state_history: torch.Tensor,
        current_raw: torch.Tensor,
        target_joint_raw: torch.Tensor,
        target_pose_camera_raw: torch.Tensor,
    ) -> torch.Tensor:
        if target_joint_raw.shape[1:] != (self.action_horizon, self.state_dim):
            raise ValueError(
                f"joint target must be [B,{self.action_horizon},{self.state_dim}], "
                f"got {tuple(target_joint_raw.shape)}"
            )
        if target_pose_camera_raw.shape[1:] != (self.action_horizon, 9):
            raise ValueError("shared pose target must be [B,100,9]")
        indices = self._control_indices(target_joint_raw.device)
        target_joint_control = target_joint_raw.index_select(1, indices)
        target_pose_control = target_pose_camera_raw.index_select(1, indices)
        predicted_joints, gripper_logits = self._decode(
            prefix_features,
            prefix_pad_mask,
            normalized_state_history,
            current_raw,
        )

        position, delta, acceleration, jerk = self._trajectory_losses(
            predicted_joints, target_joint_control[..., : self.joint_dims]
        )
        gripper = self._gripper_loss(
            gripper_logits, target_joint_control[..., self.joint_dims] >= 0.5
        )

        pred_fk = self._fk(predicted_joints)
        target_fk = self._fk(target_joint_control[..., : self.joint_dims])
        fk_position = F.smooth_l1_loss(
            torch.linalg.vector_norm(
                pred_fk[..., :3, 3] - target_fk[..., :3, 3], dim=-1
            )
            / self.fk_position_error_scale_m,
            torch.zeros_like(pred_fk[..., 0, 0]),
            reduction="mean",
            beta=1.0,
        )
        fk_angle = geodesic_angle(pred_fk[..., :3, :3], target_fk[..., :3, :3])
        fk_rotation = F.smooth_l1_loss(
            fk_angle / self.rotation_error_scale_rad,
            torch.zeros_like(fk_angle),
            reduction="mean",
            beta=1.0,
        )

        camera_from_world = self.camera_from_world.to(pred_fk)
        predicted_camera = camera_from_world[None, None] @ pred_fk
        camera_position_error = torch.linalg.vector_norm(
            predicted_camera[..., :3, 3] - target_pose_control[..., :3], dim=-1
        )
        camera_position = F.smooth_l1_loss(
            camera_position_error / self.fk_position_error_scale_m,
            torch.zeros_like(camera_position_error),
            reduction="mean",
            beta=1.0,
        )
        target_camera_rotation = rotation_6d_to_matrix(target_pose_control[..., 3:9])
        camera_angle = geodesic_angle(
            predicted_camera[..., :3, :3], target_camera_rotation
        )
        camera_rotation = F.smooth_l1_loss(
            camera_angle / self.rotation_error_scale_rad,
            torch.zeros_like(camera_angle),
            reduction="mean",
            beta=1.0,
        )

        total = (
            self.joint_position_loss_weight * position
            + self.joint_delta_loss_weight * delta
            + self.joint_acceleration_loss_weight * acceleration
            + self.joint_jerk_loss_weight * jerk
            + self.fk_position_loss_weight * fk_position
            + self.fk_rotation_loss_weight * fk_rotation
            + self.camera_position_loss_weight * camera_position
            + self.camera_rotation_loss_weight * camera_rotation
            + self.gripper_loss_weight * gripper
        )
        self.last_loss_components = {
            "joint_position": position.detach(),
            "joint_delta": delta.detach(),
            "joint_acceleration": acceleration.detach(),
            "joint_jerk": jerk.detach(),
            "fk_position": fk_position.detach(),
            "fk_rotation": fk_rotation.detach(),
            "camera_position": camera_position.detach(),
            "camera_rotation": camera_rotation.detach(),
            "gripper_bce": gripper.detach(),
            "joint_total": total.detach(),
        }
        return total

    def predict(
        self,
        *,
        prefix_features: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        normalized_state_history: torch.Tensor,
        current_raw: torch.Tensor,
    ) -> torch.Tensor:
        prediction, _ = self.predict_with_details(
            prefix_features=prefix_features,
            prefix_pad_mask=prefix_pad_mask,
            normalized_state_history=normalized_state_history,
            current_raw=current_raw,
        )
        return prediction

    def predict_with_details(
        self,
        *,
        prefix_features: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        normalized_state_history: torch.Tensor,
        current_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the 100-step physical joint trajectory and native logits."""
        control_joints, gripper_logits = self._decode(
            prefix_features,
            prefix_pad_mask,
            normalized_state_history,
            current_raw,
        )
        joints = F.interpolate(
            control_joints.transpose(1, 2),
            size=self.action_horizon,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        gripper = (gripper_logits >= 0.0).to(joints)
        gripper = F.interpolate(
            gripper[:, None], size=self.action_horizon, mode="nearest"
        ).transpose(1, 2)
        return torch.cat((joints, gripper), dim=-1), gripper_logits
