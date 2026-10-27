"""Flow-matching joint policy with an explicitly binary gripper channel.

The first ``joint_dims`` outputs are trained with the usual flow-matching
velocity loss plus direct clean-action, velocity, and acceleration trajectory
losses.  The gripper is not treated as another continuous joint: its clean-
action estimate receives a BCE loss and is hard-thresholded to the two
normalized endpoints during inference.  With quantile/min-max normalization
of 0/1 labels those endpoints are -1/+1 and unnormalize exactly to 0/1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from egomimic.models.fm_policy import FMPolicy


class JointFMPolicy(FMPolicy):
    def __init__(
        self,
        *args,
        joint_dims: int = 6,
        gripper_index: int = 6,
        gripper_loss_weight: float = 1.0,
        gripper_logit_scale: float = 4.0,
        gripper_normalized_threshold: float = 0.0,
        gripper_closed_weight: float = 1.76,
        gripper_open_weight: float = 0.70,
        gripper_transition_weight: float = 2.0,
        gripper_transition_radius: int = 3,
        joint_position_loss_weight: float = 1.0,
        joint_position_beta: float = 0.05,
        joint_delta_loss_weight: float = 0.15,
        joint_delta_beta: float = 0.05,
        joint_acceleration_loss_weight: float = 0.05,
        joint_acceleration_beta: float = 0.02,
        joint_dimension_weights: tuple[float, ...] | list[float] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.joint_dims = int(joint_dims)
        self.gripper_index = int(gripper_index)
        self.gripper_loss_weight = float(gripper_loss_weight)
        self.gripper_logit_scale = float(gripper_logit_scale)
        self.gripper_normalized_threshold = float(gripper_normalized_threshold)
        self.gripper_closed_weight = float(gripper_closed_weight)
        self.gripper_open_weight = float(gripper_open_weight)
        self.gripper_transition_weight = float(gripper_transition_weight)
        self.gripper_transition_radius = int(gripper_transition_radius)
        self.joint_position_loss_weight = float(joint_position_loss_weight)
        self.joint_position_beta = float(joint_position_beta)
        self.joint_delta_loss_weight = float(joint_delta_loss_weight)
        self.joint_delta_beta = float(joint_delta_beta)
        self.joint_acceleration_loss_weight = float(joint_acceleration_loss_weight)
        self.joint_acceleration_beta = float(joint_acceleration_beta)
        if joint_dimension_weights is None:
            joint_dimension_weights = [1.0] * self.joint_dims
        if len(joint_dimension_weights) != self.joint_dims:
            raise ValueError(
                "joint_dimension_weights must contain exactly "
                f"{self.joint_dims} entries, got {len(joint_dimension_weights)}"
            )
        dimension_weights = torch.as_tensor(joint_dimension_weights, dtype=torch.float32)
        if not torch.isfinite(dimension_weights).all() or (dimension_weights <= 0).any():
            raise ValueError("joint_dimension_weights must be finite and positive")
        # Mean-one weights preserve the overall scale when emphasizing selected joints.
        self.register_buffer(
            "joint_dimension_weights",
            dimension_weights / dimension_weights.mean(),
            persistent=False,
        )
        # Keep this head self-contained so it also works with the upstream
        # FMPolicy version that predates temporally weighted Cartesian loss.
        self.early_horizon = int(kwargs.get("early_horizon", 0))
        self.early_weight = float(kwargs.get("early_weight", 1.0))
        if self.joint_dims <= 0:
            raise ValueError("joint_dims must be positive")
        if self.gripper_loss_weight <= 0 or self.gripper_logit_scale <= 0:
            raise ValueError("gripper loss weight and logit scale must be positive")
        if min(self.gripper_closed_weight, self.gripper_open_weight) <= 0:
            raise ValueError("gripper class weights must be positive")
        if self.gripper_transition_weight < 1 or self.gripper_transition_radius < 0:
            raise ValueError("invalid gripper transition weighting")
        if self.joint_position_loss_weight < 0 or self.joint_position_beta <= 0:
            raise ValueError("invalid joint position loss configuration")
        if self.joint_delta_loss_weight < 0 or self.joint_delta_beta <= 0:
            raise ValueError("invalid joint delta loss configuration")
        if (
            self.joint_acceleration_loss_weight < 0
            or self.joint_acceleration_beta <= 0
        ):
            raise ValueError("invalid joint acceleration loss configuration")
        self.last_loss_components: dict[str, torch.Tensor] = {}

    def _validate_action_shape(self, actions: torch.Tensor) -> None:
        expected = self.joint_dims + 1
        if actions.shape[-1] != expected or self.gripper_index != self.joint_dims:
            raise ValueError(
                f"JointFMPolicy expects {self.joint_dims} joints followed by one gripper "
                f"channel (D={expected}), got shape {tuple(actions.shape)} and "
                f"gripper_index={self.gripper_index}"
            )

    def _joint_temporal_weights(self, actions: torch.Tensor) -> torch.Tensor:
        weights = torch.ones(
            (1, actions.shape[1], 1), dtype=actions.dtype, device=actions.device
        )
        if self.early_horizon and self.early_weight != 1.0:
            weights[:, : min(self.early_horizon, actions.shape[1])] = self.early_weight
        return weights

    def _predict_joint_flow(self, actions, global_cond):
        """Return flow state without depending on optional FMPolicy extensions."""
        noise = torch.randn(actions.shape, device=actions.device)
        batch_shape = (actions.shape[0],)
        if self.time_dist == "beta":
            time = torch.distributions.Beta(1.5, 1.0).sample(batch_shape).to(
                actions.device
            )
        elif self.time_dist == "uniform":
            time = torch.rand(batch_shape, device=actions.device)
        else:
            raise ValueError(f"Unsupported time_dist: {self.time_dist}")
        time = time * 0.999 + 0.001
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions
        target_velocity = noise - actions
        pred_velocity = self.model(x_t, time, global_cond)
        return pred_velocity, target_velocity, x_t, time

    def _gripper_weights(
        self, target_open: torch.Tensor, temporal_weights: torch.Tensor
    ) -> torch.Tensor:
        temporal_weights = temporal_weights.expand(target_open.shape[0], -1)
        class_weights = torch.where(
            target_open,
            torch.full_like(temporal_weights, self.gripper_open_weight),
            torch.full_like(temporal_weights, self.gripper_closed_weight),
        )
        transition = torch.zeros_like(temporal_weights)
        transition[:, 1:] = (target_open[:, 1:] != target_open[:, :-1]).to(
            temporal_weights.dtype
        )
        if self.gripper_transition_radius:
            radius = self.gripper_transition_radius
            transition = F.max_pool1d(
                transition.unsqueeze(1),
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            ).squeeze(1)
        transition_weights = 1.0 + (
            self.gripper_transition_weight - 1.0
        ) * transition
        return temporal_weights * class_weights * transition_weights

    def _joint_delta_loss(
        self,
        clean_estimate: torch.Tensor,
        actions: torch.Tensor,
        temporal_weights: torch.Tensor,
    ) -> torch.Tensor:
        if actions.shape[1] < 2:
            return clean_estimate.new_zeros(())
        predicted_delta = torch.diff(
            clean_estimate[..., : self.joint_dims], dim=1
        )
        target_delta = torch.diff(actions[..., : self.joint_dims], dim=1)
        error = F.smooth_l1_loss(
            predicted_delta,
            target_delta,
            reduction="none",
            beta=self.joint_delta_beta,
        )
        delta_weights = temporal_weights[:, 1:, None]
        dimension_weights = self.joint_dimension_weights.to(error)[None, None, :]
        return (error * delta_weights * dimension_weights).sum() / (
            actions.shape[0]
            * dimension_weights.sum()
            * delta_weights.sum().clamp_min(1e-6)
        )

    def _joint_position_loss(
        self,
        clean_estimate: torch.Tensor,
        actions: torch.Tensor,
        temporal_weights: torch.Tensor,
    ) -> torch.Tensor:
        error = F.smooth_l1_loss(
            clean_estimate[..., : self.joint_dims],
            actions[..., : self.joint_dims],
            reduction="none",
            beta=self.joint_position_beta,
        )
        position_weights = temporal_weights[..., None]
        dimension_weights = self.joint_dimension_weights.to(error)[None, None, :]
        return (error * position_weights * dimension_weights).sum() / (
            actions.shape[0]
            * dimension_weights.sum()
            * position_weights.sum().clamp_min(1e-6)
        )

    def _joint_acceleration_loss(
        self,
        clean_estimate: torch.Tensor,
        actions: torch.Tensor,
        temporal_weights: torch.Tensor,
    ) -> torch.Tensor:
        if actions.shape[1] < 3:
            return clean_estimate.new_zeros(())
        predicted_joints = clean_estimate[..., : self.joint_dims]
        target_joints = actions[..., : self.joint_dims]
        predicted_acceleration = torch.diff(
            torch.diff(predicted_joints, dim=1), dim=1
        )
        target_acceleration = torch.diff(torch.diff(target_joints, dim=1), dim=1)
        error = F.smooth_l1_loss(
            predicted_acceleration,
            target_acceleration,
            reduction="none",
            beta=self.joint_acceleration_beta,
        )
        acceleration_weights = temporal_weights[:, 2:, None]
        dimension_weights = self.joint_dimension_weights.to(error)[None, None, :]
        return (error * acceleration_weights * dimension_weights).sum() / (
            actions.shape[0]
            * dimension_weights.sum()
            * acceleration_weights.sum().clamp_min(1e-6)
        )

    def compute_loss(self, global_cond, data):
        actions, global_cond = self.preprocess_compute_loss(global_cond, data)
        self._validate_action_shape(actions)
        pred_velocity, target_velocity, x_t, time = self._predict_joint_flow(
            actions, global_cond
        )
        weights = self._joint_temporal_weights(actions)

        joint_error = (
            pred_velocity[..., : self.joint_dims]
            - target_velocity[..., : self.joint_dims]
        ).square()
        joint_loss = (joint_error * weights).sum() / (
            actions.shape[0]
            * self.joint_dims
            * weights.sum().clamp_min(1e-6)
        )

        # x_t = x_0 + t*v, hence x_0_hat = x_t - t*v_hat.  Binary labels are
        # normalized to opposite sides of zero by quantile/min-max statistics.
        clean_estimate = x_t - time[:, None, None] * pred_velocity
        target_open_bool = (
            actions[..., self.gripper_index]
            > self.gripper_normalized_threshold
        )
        target_open = target_open_bool.to(actions.dtype)
        gripper_logits = (
            clean_estimate[..., self.gripper_index]
            - self.gripper_normalized_threshold
        ) * self.gripper_logit_scale
        gripper_bce = F.binary_cross_entropy_with_logits(
            gripper_logits, target_open, reduction="none"
        )
        time_weights = weights.squeeze(-1)
        gripper_weights = self._gripper_weights(target_open_bool, time_weights)
        gripper_loss = (gripper_bce * gripper_weights).sum() / (
            gripper_weights.sum().clamp_min(1e-6)
        )
        joint_position_loss = self._joint_position_loss(
            clean_estimate, actions, time_weights
        )
        joint_delta_loss = self._joint_delta_loss(
            clean_estimate, actions, time_weights
        )
        joint_acceleration_loss = self._joint_acceleration_loss(
            clean_estimate, actions, time_weights
        )
        total_loss = (
            joint_loss
            + self.joint_position_loss_weight * joint_position_loss
            + self.gripper_loss_weight * gripper_loss
            + self.joint_delta_loss_weight * joint_delta_loss
            + self.joint_acceleration_loss_weight * joint_acceleration_loss
        )
        self.last_loss_components = {
            "joint_flow_loss": joint_loss.detach(),
            "joint_position_loss": joint_position_loss.detach(),
            "gripper_bce": gripper_loss.detach(),
            "joint_delta_loss": joint_delta_loss.detach(),
            "joint_acceleration_loss": joint_acceleration_loss.detach(),
            "joint_total_loss": total_loss.detach(),
        }
        return total_loss

    def inference(self, noise, global_cond, generator=None) -> torch.Tensor:
        actions = super().inference(noise, global_cond, generator)
        self._validate_action_shape(actions)
        actions = actions.clone()
        actions[..., self.gripper_index] = torch.where(
            actions[..., self.gripper_index] >= self.gripper_normalized_threshold,
            torch.ones_like(actions[..., self.gripper_index]),
            -torch.ones_like(actions[..., self.gripper_index]),
        )
        return actions
