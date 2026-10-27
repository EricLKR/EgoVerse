"""Native-rate HPT expert-action head for mixed PIPER/human training.

The action contract deliberately mirrors the deployable part of the pi0.5 V5
expert action representation while retaining the HPT observation trunk:

* channels 0..8: shared normalized camera-frame xyz + rotation-6D pose;
* channels 9..14: PIPER-only bounded joint residual latent;
* binary gripper: a separate classifier over the Flow transformer hidden state.

Human examples supervise only channels 0..8.  PIPER examples supervise all
15 Flow channels and the binary gripper.  The physical PIPER command is decoded
at the native 30-step horizon, without IK and without 30<->100 interpolation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.denoising_nets import CrossBlock, posemb_sincos


POSE_DIMS = 9
JOINT_DIMS = 6
ACTION_DIMS = POSE_DIMS + JOINT_DIMS
POSE_SLICE = slice(0, POSE_DIMS)
JOINT_SLICE = slice(POSE_DIMS, ACTION_DIMS)


class CrossTransformerHiddenV3(nn.Module):
    """CrossTransformer variant that can expose per-action hidden states."""

    def __init__(
        self,
        nblocks: int,
        cond_dim: int,
        hidden_dim: int,
        act_dim: int,
        act_seq: int,
        n_heads: int,
        dropout: float,
        mlp_layers: int,
        mlp_ratio: int,
        **kwargs,
    ) -> None:
        super().__init__()
        if int(act_dim) != ACTION_DIMS:
            raise ValueError(f"Expert-Action V3 requires act_dim={ACTION_DIMS}")
        self.layers = nn.ModuleList(
            [
                CrossBlock(
                    cond_dim,
                    hidden_dim,
                    n_heads,
                    dropout,
                    mlp_layers,
                    mlp_ratio,
                    **kwargs,
                )
                for _ in range(int(nblocks))
            ]
        )
        self.proj_u = nn.Linear(int(act_dim), int(hidden_dim) // 2)
        self.proj_d = nn.Linear(int(hidden_dim), int(act_dim))
        self.pos_emb = nn.Parameter(
            torch.zeros(1, int(act_seq), int(hidden_dim) // 2)
        )

    def forward(self, x, timesteps, cond, *, return_hidden: bool = False):
        hidden = self.proj_u(x) + self.pos_emb
        time_embed = (
            posemb_sincos(
                timesteps,
                self.proj_u.out_features,
                min_period=4e-3,
                max_period=4.0,
            )
            .unsqueeze(1)
            .expand(hidden.shape[0], hidden.shape[1], -1)
            .to(hidden)
        )
        hidden = torch.cat((hidden, time_embed), dim=-1)
        for layer in self.layers:
            hidden = layer(hidden, cond)
        velocity = self.proj_d(hidden)
        if return_hidden:
            return velocity, hidden
        return velocity


class HPTExpertActionFlowV3(nn.Module):
    """One masked 15-D Flow head plus a binary PIPER gripper classifier."""

    contract_name = "hpt_expert_action_v3"
    contract_version = 3
    requires_data_for_inference = True

    def __init__(
        self,
        model: CrossTransformerHiddenV3,
        action_horizon: int,
        infer_ac_dims: dict[str, int],
        num_inference_steps: int = 10,
        time_dist: str = "beta",
        piper_pose_flow_weight: float = 1.0,
        human_pose_flow_weight: float = 1.0,
        piper_joint_flow_weight: float = 1.0,
        piper_physical_loss_weight: float = 1.0,
        rotation_geodesic_weight: float = 0.1,
        joint_residual_scale_deg: Sequence[float] = (
            100.0,
            135.0,
            135.0,
            160.0,
            135.0,
            150.0,
        ),
        joint_error_scale_deg: float = 5.0,
        joint_position_loss_weight: float = 1.0,
        joint_delta_loss_weight: float = 0.25,
        joint_acceleration_loss_weight: float = 0.10,
        joint_jerk_loss_weight: float = 0.02,
        first_action_loss_weight: float = 1.0,
        fk_position_loss_weight: float = 0.10,
        fk_position_error_scale_m: float = 0.02,
        fk_rotation_loss_weight: float = 0.05,
        fk_rotation_error_scale_deg: float = 10.0,
        latent_saturation_loss_weight: float = 0.01,
        latent_saturation_threshold: float = 0.95,
        gripper_hidden_dim: int = 256,
        gripper_loss_weight: float = 1.5,
        gripper_closed_weight: float = 1.0,
        gripper_open_weight: float = 1.0,
        gripper_transition_weight: float = 3.0,
        gripper_transition_radius: int = 2,
        gripper_open_prior: float = 0.5,
        joint_dimension_weights: Sequence[float] = (1, 1, 1, 1, 1, 1),
        **_: object,
    ) -> None:
        super().__init__()
        self.model = model
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.infer_ac_dims = dict(infer_ac_dims)
        if self.action_horizon != 30:
            raise ValueError("HPT Expert-Action V3 requires the native 30-step horizon")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if set(self.infer_ac_dims.values()) != {ACTION_DIMS}:
            raise ValueError("all Expert-Action V3 Flow outputs must be 15-D")
        if time_dist not in {"beta", "uniform"}:
            raise ValueError("time_dist must be beta or uniform")
        self.time_dist = str(time_dist)

        scalar_weights = (
            piper_pose_flow_weight,
            human_pose_flow_weight,
            piper_joint_flow_weight,
            piper_physical_loss_weight,
            rotation_geodesic_weight,
            joint_position_loss_weight,
            joint_delta_loss_weight,
            joint_acceleration_loss_weight,
            joint_jerk_loss_weight,
            first_action_loss_weight,
            fk_position_loss_weight,
            fk_rotation_loss_weight,
            latent_saturation_loss_weight,
            gripper_loss_weight,
        )
        if min(float(value) for value in scalar_weights) < 0:
            raise ValueError("loss weights must be non-negative")
        self.piper_pose_flow_weight = float(piper_pose_flow_weight)
        self.human_pose_flow_weight = float(human_pose_flow_weight)
        self.piper_joint_flow_weight = float(piper_joint_flow_weight)
        self.piper_physical_loss_weight = float(piper_physical_loss_weight)
        self.rotation_geodesic_weight = float(rotation_geodesic_weight)
        self.joint_position_loss_weight = float(joint_position_loss_weight)
        self.joint_delta_loss_weight = float(joint_delta_loss_weight)
        self.joint_acceleration_loss_weight = float(joint_acceleration_loss_weight)
        self.joint_jerk_loss_weight = float(joint_jerk_loss_weight)
        self.first_action_loss_weight = float(first_action_loss_weight)
        self.fk_position_loss_weight = float(fk_position_loss_weight)
        self.fk_rotation_loss_weight = float(fk_rotation_loss_weight)
        self.latent_saturation_loss_weight = float(latent_saturation_loss_weight)
        self.latent_saturation_threshold = float(latent_saturation_threshold)
        if not 0.0 < self.latent_saturation_threshold < 1.0:
            raise ValueError("latent_saturation_threshold must be in (0, 1)")

        scale_deg = torch.as_tensor(tuple(joint_residual_scale_deg), dtype=torch.float32)
        if scale_deg.shape != (JOINT_DIMS,) or (scale_deg <= 0).any():
            raise ValueError("joint_residual_scale_deg must contain six positive values")
        self.register_buffer("joint_residual_scale_deg", scale_deg, persistent=True)
        self.register_buffer(
            "joint_residual_scale_rad", torch.deg2rad(scale_deg), persistent=True
        )
        dimension_weights = torch.as_tensor(
            tuple(joint_dimension_weights), dtype=torch.float32
        )
        if dimension_weights.shape != (JOINT_DIMS,) or (dimension_weights <= 0).any():
            raise ValueError("joint_dimension_weights must contain six positive values")
        self.register_buffer(
            "joint_dimension_weights",
            dimension_weights / dimension_weights.mean(),
            persistent=True,
        )

        self.joint_error_scale_rad = math.radians(float(joint_error_scale_deg))
        self.fk_position_error_scale_m = float(fk_position_error_scale_m)
        self.fk_rotation_error_scale_rad = math.radians(
            float(fk_rotation_error_scale_deg)
        )
        self.gripper_loss_weight = float(gripper_loss_weight)
        self.gripper_closed_weight = float(gripper_closed_weight)
        self.gripper_open_weight = float(gripper_open_weight)
        self.gripper_transition_weight = float(gripper_transition_weight)
        self.gripper_transition_radius = int(gripper_transition_radius)
        if not 0.0 < float(gripper_open_prior) < 1.0:
            raise ValueError("gripper_open_prior must be in (0, 1)")
        hidden_dim = int(self.model.proj_d.in_features)
        self.gripper_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, int(gripper_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(gripper_hidden_dim), 1),
        )
        self.gripper_open_prior = float(gripper_open_prior)
        self.reset_output_prior()
        self._physical_fk_chain = None
        self.last_loss_components: dict[str, torch.Tensor] = {}

    def reset_output_prior(self) -> None:
        nn.init.zeros_(self.gripper_head[-1].weight)
        nn.init.constant_(
            self.gripper_head[-1].bias,
            math.log(self.gripper_open_prior / (1.0 - self.gripper_open_prior)),
        )

    @staticmethod
    def _unwrap_features(global_cond):
        return global_cond[0] if isinstance(global_cond, tuple) else global_cond

    @staticmethod
    def _norm_parameters(data: dict, prefix: str, reference: torch.Tensor):
        mode = data.get(f"_{prefix}_norm_mode")
        names = {
            "quantile": ("quantile_1", "quantile_99"),
            "minmax": ("min", "max"),
            "zscore": ("mean", "std"),
        }.get(mode)
        if names is None:
            raise ValueError(f"missing or unsupported {prefix} normalization metadata")
        first = data[f"_{prefix}_norm_{names[0]}"].to(reference)
        second = data[f"_{prefix}_norm_{names[1]}"].to(reference)

        # Action normalization statistics are stored per source timestep
        # (normally [100, D]), whereas Expert-Action V3 deliberately operates
        # at the native 30-step control rate.  Select exactly the same source
        # indices used for the action targets instead of broadcasting a
        # mismatched 100-step statistic across a 30-step tensor.
        if reference.ndim >= 3 and first.ndim == reference.ndim - 1:
            if first.shape[-1] != reference.shape[-1]:
                raise ValueError(
                    f"{prefix} normalization feature dimension "
                    f"{first.shape[-1]} does not match {reference.shape[-1]}"
                )
            target_horizon = reference.shape[-2]
            source_horizon = first.shape[-2]
            if source_horizon != target_horizon:
                if source_horizon < target_horizon:
                    raise ValueError(
                        f"{prefix} normalization horizon {source_horizon} is "
                        f"shorter than target horizon {target_horizon}"
                    )
                indices = torch.linspace(
                    0,
                    source_horizon - 1,
                    target_horizon,
                    device=first.device,
                ).round().long()
                first = first.index_select(-2, indices)
                second = second.index_select(-2, indices)
        return mode, first, second

    @classmethod
    def _unnormalize(cls, values: torch.Tensor, data: dict, prefix: str):
        mode, first, second = cls._norm_parameters(data, prefix, values)
        if mode in {"quantile", "minmax"}:
            return (values + 1.0) * 0.5 * (second - first + 1e-6) + first
        return values * (second + 1e-6) + first

    @classmethod
    def _normalize(cls, values: torch.Tensor, data: dict, prefix: str):
        mode, first, second = cls._norm_parameters(data, prefix, values)
        if mode in {"quantile", "minmax"}:
            return 2.0 * (values - first) / (second - first + 1e-6) - 1.0
        return (values - first) / (second + 1e-6)

    @staticmethod
    def _is_piper(data: dict) -> bool:
        return "state_joints" in data

    def _control_indices(self, source_horizon: int, device: torch.device):
        if source_horizon < self.action_horizon:
            raise ValueError(
                f"source horizon {source_horizon} is shorter than native horizon "
                f"{self.action_horizon}"
            )
        return torch.linspace(
            0, source_horizon - 1, self.action_horizon, device=device
        ).round().long()

    def _current_raw(self, data: dict, reference: torch.Tensor) -> torch.Tensor:
        state = data["state_joints"].to(reference)
        state = state.reshape(state.shape[0], -1, state.shape[-1])[:, -1]
        raw = self._unnormalize(state, data, "proprio")
        if raw.shape[-1] != JOINT_DIMS + 1:
            raise ValueError("PIPER proprio must contain six joints and gripper")
        return raw

    def _joint_to_latent(
        self, target_joint: torch.Tensor, current_joint: torch.Tensor
    ) -> torch.Tensor:
        scale = self.joint_residual_scale_rad.to(target_joint)
        utilization = (target_joint - current_joint[:, None]) / scale[None, None]
        limit = 1.0 - 1e-4
        maximum = utilization.detach().abs().max().item()
        if maximum >= limit:
            raise ValueError(
                "joint residual reaches Expert-Action V3 scale: "
                f"max_utilization={maximum:.6f}, limit={limit:.6f}"
            )
        return torch.atanh(utilization.clamp(-limit, limit))

    def _joint_from_latent(
        self, latent: torch.Tensor, current_joint: torch.Tensor
    ) -> torch.Tensor:
        scale = self.joint_residual_scale_rad.to(latent)
        return current_joint[:, None] + scale[None, None] * torch.tanh(latent)

    def _targets(self, data: dict):
        pose = data["action"]
        if pose.ndim != 3 or pose.shape[-1] != POSE_DIMS:
            raise ValueError("shared pose target must be [B,T,9]")
        indices = self._control_indices(pose.shape[1], pose.device)
        pose30 = pose.index_select(1, indices)
        target = torch.zeros(
            pose30.shape[0], self.action_horizon, ACTION_DIMS,
            dtype=pose30.dtype, device=pose30.device,
        )
        target[..., POSE_SLICE] = pose30
        pose_mask = torch.zeros_like(target)
        pose_mask[..., POSE_SLICE] = 1.0
        joint_mask = torch.zeros_like(target)
        current_raw = target_joint_raw = target_open = None
        if self._is_piper(data):
            if "actions_joints" not in data:
                raise ValueError("PIPER batch lacks actions_joints")
            normalized_joint = data["actions_joints"]
            if normalized_joint.ndim != 3 or normalized_joint.shape[-1] != 7:
                raise ValueError("actions_joints must be [B,T,7]")
            joint_indices = self._control_indices(
                normalized_joint.shape[1], normalized_joint.device
            )
            target_joint_raw = self._unnormalize(
                normalized_joint.index_select(1, joint_indices),
                data,
                "actions_joints",
            )
            current_raw = self._current_raw(data, pose30)
            target[..., JOINT_SLICE] = self._joint_to_latent(
                target_joint_raw[..., :JOINT_DIMS],
                current_raw[..., :JOINT_DIMS],
            )
            joint_mask[..., JOINT_SLICE] = 1.0
            raw_gripper = target_joint_raw[..., JOINT_DIMS]
            binary_distance = torch.minimum(
                raw_gripper.abs(), (raw_gripper - 1.0).abs()
            )
            if binary_distance.max().item() > 2e-4:
                raise ValueError("PIPER gripper target is not binary")
            target_open = raw_gripper >= 0.5
        return (
            target,
            pose_mask,
            joint_mask,
            pose30,
            current_raw,
            target_joint_raw,
            target_open,
        )

    def _sample_time(self, batch_size: int, device: torch.device):
        if self.time_dist == "beta":
            time = torch.distributions.Beta(1.5, 1.0).sample((batch_size,)).to(device)
        else:
            time = torch.rand(batch_size, device=device)
        return time * 0.999 + 0.001

    @staticmethod
    def _masked_mse(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (error.square() * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
        c1 = F.normalize(rotation_6d[..., :3], dim=-1, eps=1e-6)
        c2_raw = rotation_6d[..., 3:6]
        c2 = F.normalize(
            c2_raw - (c1 * c2_raw).sum(dim=-1, keepdim=True) * c1,
            dim=-1,
            eps=1e-6,
        )
        c3 = torch.cross(c1, c2, dim=-1)
        return torch.stack((c1, c2, c3), dim=-1)

    def _rotation_loss(
        self, predicted_pose: torch.Tensor, target_pose: torch.Tensor, data: dict
    ) -> torch.Tensor:
        pred_raw = self._unnormalize(predicted_pose, data, "action")
        target_raw = self._unnormalize(target_pose, data, "action")
        pred_rot = self._rotation_6d_to_matrix(pred_raw[..., 3:9])
        target_rot = self._rotation_6d_to_matrix(target_raw[..., 3:9])
        relative = pred_rot.transpose(-1, -2) @ target_rot
        cosine = (
            (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
        ).clamp(-1.0, 1.0)
        skew = relative - relative.transpose(-1, -2)
        sine = torch.linalg.vector_norm(skew, dim=(-2, -1)) / (
            2.0 * math.sqrt(2.0)
        )
        return torch.atan2(sine, cosine).mean()

    def _weighted_huber(self, error: torch.Tensor) -> torch.Tensor:
        scaled = error / self.joint_error_scale_rad
        values = F.smooth_l1_loss(
            scaled, torch.zeros_like(scaled), reduction="none", beta=1.0
        )
        weights = self.joint_dimension_weights.to(values)[None, None]
        return (values * weights).sum() / (
            values.shape[0] * values.shape[1] * weights.sum()
        )

    def _trajectory_losses(self, prediction: torch.Tensor, target: torch.Tensor):
        outputs = []
        for order in range(4):
            pred_order, target_order = prediction, target
            for _ in range(order):
                pred_order = torch.diff(pred_order, dim=1)
                target_order = torch.diff(target_order, dim=1)
            outputs.append(self._weighted_huber(pred_order - target_order))
        return tuple(outputs)

    def _gripper_loss(self, logits: torch.Tensor, target_open: torch.Tensor):
        class_weights = torch.where(
            target_open,
            torch.full_like(logits, self.gripper_open_weight),
            torch.full_like(logits, self.gripper_closed_weight),
        )
        transition = torch.zeros_like(logits)
        transition[:, 1:] = (target_open[:, 1:] != target_open[:, :-1]).to(logits)
        if self.gripper_transition_radius:
            transition = F.max_pool1d(
                transition[:, None],
                kernel_size=2 * self.gripper_transition_radius + 1,
                stride=1,
                padding=self.gripper_transition_radius,
            ).squeeze(1)
        weights = class_weights * (
            1.0 + (self.gripper_transition_weight - 1.0) * transition
        )
        values = F.binary_cross_entropy_with_logits(
            logits, target_open.to(logits), reduction="none"
        )
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    def _get_physical_fk_chain(self, reference: torch.Tensor):
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
            device=reference.device, dtype=torch.float32
        )
        return self._physical_fk_chain

    def _fk_matrices(self, joints: torch.Tensor) -> torch.Tensor:
        lead_shape = joints.shape[:-1]
        with torch.autocast(device_type=joints.device.type, enabled=False):
            matrices = self._get_physical_fk_chain(joints.float()).forward_kinematics(
                joints.float().reshape(-1, JOINT_DIMS)
            ).get_matrix()
        return matrices.reshape(*lead_shape, 4, 4)

    def _fk_losses(self, predicted: torch.Tensor, target: torch.Tensor):
        if self.fk_position_loss_weight == 0 and self.fk_rotation_loss_weight == 0:
            zero = predicted.sum() * 0.0
            return zero, zero
        pred_fk = self._fk_matrices(predicted)
        target_fk = self._fk_matrices(target)
        position = torch.linalg.vector_norm(
            pred_fk[..., :3, 3] - target_fk[..., :3, 3], dim=-1
        ) / self.fk_position_error_scale_m
        position_loss = F.smooth_l1_loss(
            position, torch.zeros_like(position), reduction="mean", beta=1.0
        )
        relative = pred_fk[..., :3, :3].transpose(-1, -2) @ target_fk[..., :3, :3]
        cosine = (
            (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
        ).clamp(-1.0, 1.0)
        skew = relative - relative.transpose(-1, -2)
        sine = torch.linalg.vector_norm(skew, dim=(-2, -1)) / (
            2.0 * math.sqrt(2.0)
        )
        angle = torch.atan2(sine, cosine) / self.fk_rotation_error_scale_rad
        rotation_loss = F.smooth_l1_loss(
            angle, torch.zeros_like(angle), reduction="mean", beta=1.0
        )
        return position_loss, rotation_loss

    def compute_loss(self, global_cond, data: dict) -> torch.Tensor:
        features = self._unwrap_features(global_cond)
        (
            target,
            pose_mask,
            joint_mask,
            target_pose,
            current_raw,
            target_joint_raw,
            target_open,
        ) = self._targets(data)
        noise = torch.randn_like(target)
        time = self._sample_time(target.shape[0], target.device)
        x_t = time[:, None, None] * noise + (1.0 - time[:, None, None]) * target
        velocity_target = noise - target
        velocity, hidden = self.model(x_t, time, features, return_hidden=True)
        error = velocity - velocity_target
        pose_flow = self._masked_mse(error, pose_mask)
        is_piper = self._is_piper(data)
        pose_weight = (
            self.piper_pose_flow_weight if is_piper else self.human_pose_flow_weight
        )
        total = pose_weight * pose_flow
        predicted_clean = x_t - time[:, None, None] * velocity
        rotation = self._rotation_loss(
            predicted_clean[..., POSE_SLICE], target_pose, data
        )
        total = total + self.rotation_geodesic_weight * rotation
        components = {
            "pose_flow_loss": pose_flow.detach(),
            "pose_rotation_geodesic_loss": rotation.detach(),
        }

        if is_piper:
            joint_flow = self._masked_mse(error, joint_mask)
            predicted_joint = self._joint_from_latent(
                predicted_clean[..., JOINT_SLICE],
                current_raw[..., :JOINT_DIMS],
            )
            target_joint = target_joint_raw[..., :JOINT_DIMS]
            position, delta, acceleration, jerk = self._trajectory_losses(
                predicted_joint, target_joint
            )
            first_action = self._weighted_huber(
                predicted_joint[:, :1] - target_joint[:, :1]
            )
            fk_position, fk_rotation = self._fk_losses(
                predicted_joint, target_joint
            )
            utilization = torch.tanh(predicted_clean[..., JOINT_SLICE]).abs()
            saturation = F.relu(
                utilization - self.latent_saturation_threshold
            ).square().mean()
            gripper_logits = self.gripper_head(hidden.float()).squeeze(-1)
            gripper = self._gripper_loss(gripper_logits, target_open)
            physical = (
                self.joint_position_loss_weight * position
                + self.joint_delta_loss_weight * delta
                + self.joint_acceleration_loss_weight * acceleration
                + self.joint_jerk_loss_weight * jerk
                + self.first_action_loss_weight * first_action
                + self.fk_position_loss_weight * fk_position
                + self.fk_rotation_loss_weight * fk_rotation
                + self.latent_saturation_loss_weight * saturation
                + self.gripper_loss_weight * gripper
            )
            total = (
                total
                + self.piper_joint_flow_weight * joint_flow
                + self.piper_physical_loss_weight * physical
            )
            components.update(
                {
                    "joint_flow_loss": joint_flow.detach(),
                    "joint_position_loss": position.detach(),
                    "joint_delta_loss": delta.detach(),
                    "joint_acceleration_loss": acceleration.detach(),
                    "joint_jerk_loss": jerk.detach(),
                    "joint_first_action_loss": first_action.detach(),
                    "joint_fk_position_loss": fk_position.detach(),
                    "joint_fk_rotation_loss": fk_rotation.detach(),
                    "joint_latent_saturation_loss": saturation.detach(),
                    "joint_max_predicted_utilization": utilization.detach().max(),
                    "gripper_bce": gripper.detach(),
                    "joint_total_loss": physical.detach(),
                }
            )
        components["expert_action_total_loss"] = total.detach()
        self.last_loss_components = components
        return total

    @torch.no_grad()
    def sample_outputs(self, global_cond, data: dict, generator=None) -> dict:
        features = self._unwrap_features(global_cond)
        sample = torch.randn(
            features.shape[0],
            self.action_horizon,
            ACTION_DIMS,
            dtype=features.dtype,
            device=features.device,
            generator=generator,
        )
        dt = 1.0 / self.num_inference_steps
        hidden = None
        for step in range(self.num_inference_steps):
            time = torch.full(
                (features.shape[0],),
                1.0 - step * dt,
                dtype=features.dtype,
                device=features.device,
            )
            velocity, hidden = self.model(
                sample, time, features, return_hidden=True
            )
            sample = sample - dt * velocity
        pose_norm = sample[..., POSE_SLICE]
        outputs = {
            "flow_action": sample,
            "pose_normalized": pose_norm,
            "pose_raw": self._unnormalize(pose_norm, data, "action"),
        }
        if self._is_piper(data):
            current_raw = self._current_raw(data, sample)
            joints = self._joint_from_latent(
                sample[..., JOINT_SLICE], current_raw[..., :JOINT_DIMS]
            )
            gripper_logits = self.gripper_head(hidden.float()).squeeze(-1)
            gripper = (gripper_logits >= 0.0).to(joints)
            outputs["joint_raw"] = torch.cat((joints, gripper[..., None]), dim=-1)
            outputs["gripper_logits"] = gripper_logits
        return outputs

    def forward(self, global_cond, data: dict | None = None) -> torch.Tensor:
        if data is None:
            raise ValueError("Expert-Action V3 inference requires observation data")
        outputs = self.sample_outputs(global_cond, data)
        if self._is_piper(data):
            return self._normalize(outputs["joint_raw"], data, "actions_joints")
        return outputs["pose_normalized"]
