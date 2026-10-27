"""Clean V2 deterministic residual joint trajectory head for PIPER.

The head predicts 30 native control-rate residual waypoints relative to the
current measured joint state.  It uses a separate deterministic binary
gripper classifier and only upsamples to the legacy 100-point interface after
prediction, so existing evaluation and deployment wrappers remain compatible.

Clean V2 deliberately has a distinct class/module identity.  Its training
entrypoint refuses every robot-policy initialization checkpoint and computes
normalization statistics from the training split.  The ImageNet ResNet18
visual prior is still allowed and is not a robot-policy checkpoint.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualJointPolicyCleanV2(nn.Module):
    contract_name = "piper_residual_joint_clean_v2"
    contract_version = 2
    requires_data_for_inference = True

    def __init__(
        self,
        action_horizon: int,
        infer_ac_dims: dict[str, int],
        input_dim: int = 256,
        hidden_dim: int = 256,
        control_horizon: int = 30,
        joint_dims: int = 6,
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
        fk_position_error_scale_m: float = 0.02,
        fk_rotation_loss_weight: float = 0.05,
        fk_rotation_error_scale_deg: float = 10.0,
        gripper_loss_weight: float = 1.5,
        gripper_closed_weight: float = 1.76,
        gripper_open_weight: float = 0.70,
        gripper_transition_weight: float = 3.0,
        gripper_transition_radius: int = 2,
        gripper_open_prior: float = 0.716,
        joint_dimension_weights: tuple[float, ...] | list[float] | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.control_horizon = int(control_horizon)
        self.joint_dims = int(joint_dims)
        self.infer_ac_dims = dict(infer_ac_dims)
        if not self.infer_ac_dims or set(self.infer_ac_dims.values()) != {
            self.joint_dims + 1
        }:
            raise ValueError("ResidualJointPolicyCleanV2 requires 6 joints + 1 gripper")
        if min(self.action_horizon, self.control_horizon, self.joint_dims) <= 0:
            raise ValueError("trajectory dimensions must be positive")
        if not 0.0 < gripper_open_prior < 1.0:
            raise ValueError("gripper_open_prior must lie strictly between 0 and 1")

        self.context_projection = nn.Linear(input_dim, hidden_dim)
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
        self.gripper_open_prior = float(gripper_open_prior)
        self.reset_output_prior()

        self.max_residual_rad = math.radians(float(max_residual_deg))
        self.joint_error_scale_rad = math.radians(float(joint_error_scale_deg))
        self.joint_position_loss_weight = float(joint_position_loss_weight)
        self.joint_delta_loss_weight = float(joint_delta_loss_weight)
        self.joint_acceleration_loss_weight = float(
            joint_acceleration_loss_weight
        )
        self.joint_jerk_loss_weight = float(joint_jerk_loss_weight)
        self.fk_position_loss_weight = float(fk_position_loss_weight)
        self.fk_position_error_scale_m = float(fk_position_error_scale_m)
        self.fk_rotation_loss_weight = float(fk_rotation_loss_weight)
        self.fk_rotation_error_scale_rad = math.radians(
            float(fk_rotation_error_scale_deg)
        )
        self.gripper_loss_weight = float(gripper_loss_weight)
        self.gripper_closed_weight = float(gripper_closed_weight)
        self.gripper_open_weight = float(gripper_open_weight)
        self.gripper_transition_weight = float(gripper_transition_weight)
        self.gripper_transition_radius = int(gripper_transition_radius)
        if joint_dimension_weights is None:
            joint_dimension_weights = [1.0] * self.joint_dims
        weights = torch.as_tensor(joint_dimension_weights, dtype=torch.float32)
        if weights.numel() != self.joint_dims or (weights <= 0).any():
            raise ValueError("joint_dimension_weights must contain positive J1-J6 weights")
        self.register_buffer(
            "joint_dimension_weights", weights / weights.mean(), persistent=False
        )
        loss_weights = (
            self.joint_position_loss_weight,
            self.joint_delta_loss_weight,
            self.joint_acceleration_loss_weight,
            self.joint_jerk_loss_weight,
            self.fk_position_loss_weight,
            self.fk_rotation_loss_weight,
            self.gripper_loss_weight,
        )
        if min(loss_weights) < 0:
            raise ValueError("loss weights must be non-negative")
        self._physical_fk_chain = None
        self.last_loss_components: dict[str, torch.Tensor] = {}

    def reset_output_prior(self) -> None:
        """Restore persistence/open priors after HPT's global initializer."""
        nn.init.zeros_(self.joint_residual_head.weight)
        nn.init.zeros_(self.joint_residual_head.bias)
        nn.init.zeros_(self.gripper_head.weight)
        nn.init.constant_(
            self.gripper_head.bias,
            math.log(self.gripper_open_prior / (1.0 - self.gripper_open_prior)),
        )

    @staticmethod
    def _unwrap_features(global_cond):
        return global_cond[0] if isinstance(global_cond, tuple) else global_cond

    def _decode(self, global_cond) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._unwrap_features(global_cond)
        memory = self.context_projection(features)
        queries = self.control_queries.expand(features.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        residual = torch.tanh(self.joint_residual_head(decoded)) * self.max_residual_rad
        gripper_logits = self.gripper_head(decoded).squeeze(-1)
        return residual, gripper_logits

    @staticmethod
    def _norm_parameters(data: dict, prefix: str, reference: torch.Tensor):
        mode = data.get(f"_{prefix}_norm_mode")
        if mode == "quantile":
            first, second = "quantile_1", "quantile_99"
        elif mode == "minmax":
            first, second = "min", "max"
        elif mode == "zscore":
            first, second = "mean", "std"
        else:
            raise ValueError(f"missing or unsupported {prefix} normalization metadata")
        return (
            mode,
            data[f"_{prefix}_norm_{first}"].to(reference),
            data[f"_{prefix}_norm_{second}"].to(reference),
        )

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

    def _current_raw(self, data: dict, reference: torch.Tensor) -> torch.Tensor:
        state = data["state_joints"].to(reference)
        state = state.reshape(state.shape[0], -1, state.shape[-1])[:, -1]
        return self._unnormalize(state, data, "proprio")

    def _validate_data_contract(self, data: dict, *, require_action: bool) -> None:
        """Fail before optimization when a dataset/config contract drifts."""
        if "state_joints" not in data:
            raise ValueError("Clean V2 requires normalized state_joints")
        if require_action and "action" not in data:
            raise ValueError("Clean V2 training requires normalized action labels")
        state = data["state_joints"]
        if state.ndim < 2 or state.shape[-1] != self.joint_dims + 1:
            raise ValueError(
                "Clean V2 proprio contract mismatch: expected final dimension "
                f"{self.joint_dims + 1}, got {tuple(state.shape)}"
            )
        if not torch.isfinite(state).all():
            raise ValueError("Clean V2 received non-finite normalized proprio")
        if "action" not in data:
            return
        actions = data["action"]
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (
            self.action_horizon,
            self.joint_dims + 1,
        ):
            raise ValueError(
                "Clean V2 action contract mismatch: expected "
                f"[B,{self.action_horizon},{self.joint_dims + 1}], got "
                f"{tuple(actions.shape)}"
            )
        if not torch.isfinite(actions).all():
            raise ValueError("Clean V2 received non-finite normalized actions")
        raw_actions = self._unnormalize(actions, data, "action")
        raw_gripper = raw_actions[..., self.joint_dims]
        binary_distance = torch.minimum(raw_gripper.abs(), (raw_gripper - 1.0).abs())
        if torch.max(binary_distance).item() > 2e-4:
            raise ValueError(
                "Clean V2 gripper target is not binary after unnormalization: "
                f"max distance={torch.max(binary_distance).item():.6g}"
            )

    def _control_indices(self, device: torch.device) -> torch.Tensor:
        return torch.linspace(
            0, self.action_horizon - 1, self.control_horizon, device=device
        ).round().long()

    def _raw_control_targets(self, data: dict) -> torch.Tensor:
        actions = data["action"]
        raw = self._unnormalize(actions, data, "action")
        return raw.index_select(1, self._control_indices(raw.device))

    def _raw_control_prediction(
        self, global_cond, data: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual, gripper_logits = self._decode(global_cond)
        current = self._current_raw(data, residual)[..., : self.joint_dims]
        return current[:, None, :] + residual, gripper_logits

    def _weighted_huber(self, error: torch.Tensor) -> torch.Tensor:
        scaled = error / self.joint_error_scale_rad
        loss = F.smooth_l1_loss(
            scaled, torch.zeros_like(scaled), reduction="none", beta=1.0
        )
        weights = self.joint_dimension_weights.to(loss)[None, None, :]
        return (loss * weights).sum() / (loss.shape[0] * loss.shape[1] * weights.sum())

    def _joint_losses(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        losses = []
        for order in range(4):
            pred_order, target_order = predicted, target
            for _ in range(order):
                pred_order = torch.diff(pred_order, dim=1)
                target_order = torch.diff(target_order, dim=1)
            losses.append(self._weighted_huber(pred_order - target_order))
        return tuple(losses)

    def _gripper_loss(self, logits: torch.Tensor, target_open: torch.Tensor):
        class_weights = torch.where(
            target_open,
            torch.full_like(logits, self.gripper_open_weight),
            torch.full_like(logits, self.gripper_closed_weight),
        )
        transition = torch.zeros_like(logits)
        transition[:, 1:] = (target_open[:, 1:] != target_open[:, :-1]).to(
            logits.dtype
        )
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
        loss = F.binary_cross_entropy_with_logits(
            logits, target_open.to(logits.dtype), reduction="none"
        )
        return (loss * weights).sum() / weights.sum().clamp_min(1e-6)

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
            device=reference.device, dtype=reference.dtype
        )
        return self._physical_fk_chain

    def _fk_matrices(self, joints: torch.Tensor) -> torch.Tensor:
        lead_shape = joints.shape[:-1]
        with torch.autocast(device_type=joints.device.type, enabled=False):
            joints_fp32 = joints.float()
            matrices = self._get_physical_fk_chain(joints_fp32).forward_kinematics(
                joints_fp32.reshape(-1, self.joint_dims)
            ).get_matrix()
        return matrices.reshape(*lead_shape, 4, 4)

    def _fk_losses(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        sine = torch.linalg.vector_norm(skew, dim=(-2, -1)) / (2.0 * math.sqrt(2.0))
        angle = torch.atan2(sine, cosine) / self.fk_rotation_error_scale_rad
        rotation_loss = F.smooth_l1_loss(
            angle, torch.zeros_like(angle), reduction="mean", beta=1.0
        )
        return position_loss, rotation_loss

    def compute_loss(self, global_cond, data: dict) -> torch.Tensor:
        self._validate_data_contract(data, require_action=True)
        predicted_joints, gripper_logits = self._raw_control_prediction(
            global_cond, data
        )
        targets = self._raw_control_targets(data)
        target_joints = targets[..., : self.joint_dims]
        target_open = targets[..., self.joint_dims] >= 0.5
        position, delta, acceleration, jerk = self._joint_losses(
            predicted_joints, target_joints
        )
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
        )
        self.last_loss_components = {
            "clean_v2_joint_position_loss": position.detach(),
            "clean_v2_joint_delta_loss": delta.detach(),
            "clean_v2_joint_acceleration_loss": acceleration.detach(),
            "clean_v2_joint_jerk_loss": jerk.detach(),
            "clean_v2_fk_position_loss": fk_position.detach(),
            "clean_v2_fk_rotation_loss": fk_rotation.detach(),
            "clean_v2_gripper_bce": gripper.detach(),
            "clean_v2_total_loss": total.detach(),
        }
        return total

    def forward(self, global_cond, data: dict | None = None) -> torch.Tensor:
        if data is None:
            raise ValueError("ResidualJointPolicyCleanV2 inference requires current proprio data")
        self._validate_data_contract(data, require_action=False)
        control_joints, gripper_logits = self._raw_control_prediction(global_cond, data)
        joints = F.interpolate(
            control_joints.transpose(1, 2),
            size=self.action_horizon,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        gripper_open = (gripper_logits >= 0.0).to(joints.dtype)
        gripper = F.interpolate(
            gripper_open[:, None], size=self.action_horizon, mode="nearest"
        ).transpose(1, 2)
        raw_actions = torch.cat((joints, gripper), dim=-1)
        return self._normalize(raw_actions, data, "action")
