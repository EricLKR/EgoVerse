from typing import Tuple

import torch
import torch.nn.functional as F
from overrides import override

from egomimic.models.denoising_nets import ConditionalUnet1D
from egomimic.models.denoising_policy import DenoisingPolicy


class FMPolicy(DenoisingPolicy):
    """
    A diffusion-based policy head.

    Args:
        model (ConditionalUnet1D): The model used for prediction.
        noise_scheduler: The noise scheduler used for the diffusion process.
        action_horizon (int): The number of time steps in the action horizon.
        output_dim (int): The dimension of the output.
        num_inference_steps (int, optional): The number of inference steps.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        action_horizon,
        infer_ac_dims,
        num_inference_steps=None,
        **kwargs,
    ):
        super().__init__(
            model, action_horizon, infer_ac_dims, num_inference_steps, **kwargs
        )
        self.time_dist = kwargs.get("time_dist", "beta")
        self.early_horizon = int(kwargs.get("early_horizon", 0))
        self.early_weight = float(kwargs.get("early_weight", 1.0))
        self.rotation_geodesic_weight = float(
            kwargs.get("rotation_geodesic_weight", 0.0)
        )
        if self.early_horizon < 0:
            raise ValueError("early_horizon must be non-negative")
        if self.early_weight <= 0:
            raise ValueError("early_weight must be positive")
        if self.rotation_geodesic_weight < 0:
            raise ValueError("rotation_geodesic_weight must be non-negative")

    def step(self, x_t, t, global_cond):
        if len(t.shape) != 1:
            t = torch.tensor([t], device=global_cond.device)
        v_t = self.model(x_t, t, global_cond)
        return x_t + self.dt * v_t, t + self.dt

    @override
    def inference(self, noise, global_cond, generator=None) -> torch.Tensor:
        self.dt = -1.0 / self.num_inference_steps
        x_t = noise
        time = torch.ones((len(global_cond)), device=global_cond.device)
        while time[0] >= -self.dt / 2:
            x_t, time = self.step(x_t, time, global_cond)
        return x_t

    @override
    def predict(self, actions, global_cond) -> Tuple[torch.Tensor, torch.Tensor]:
        pred, target, _, _ = self._predict_with_state(actions, global_cond)
        return pred, target

    def _predict_with_state(self, actions, global_cond):
        noise = torch.randn(actions.shape, device=actions.device)
        batch_shape = (actions.shape[0],)
        if self.time_dist == "beta":
            a, b = 1.5, 1.0
            time = torch.distributions.Beta(a, b).sample(batch_shape).to(actions.device)
        elif self.time_dist == "uniform":
            time = (
                torch.distributions.Uniform(0, 1).sample(batch_shape).to(actions.device)
            )
        else:
            raise ValueError(f"Unsupported time_dist: {self.time_dist}")
        time = time * 0.999 + 0.001

        time_expanded = time.unsqueeze(-1).unsqueeze(-1)
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        v_t = self.model(x_t, time, global_cond)

        return v_t, u_t, x_t, time

    def _temporal_weights(self, actions):
        weights = torch.ones(
            (1, actions.shape[1], 1), dtype=actions.dtype, device=actions.device
        )
        if self.early_horizon and self.early_weight != 1.0:
            stop = min(self.early_horizon, actions.shape[1])
            weights[:, :stop] = self.early_weight
        return weights

    @staticmethod
    def _rotation_6d_to_matrix(rotation_6d):
        """Differentiable Zhou 6D rotation conversion using matrix columns."""
        c1 = F.normalize(rotation_6d[..., 0:3], dim=-1, eps=1e-6)
        c2_raw = rotation_6d[..., 3:6]
        c2 = F.normalize(
            c2_raw - (c1 * c2_raw).sum(dim=-1, keepdim=True) * c1,
            dim=-1,
            eps=1e-6,
        )
        c3 = torch.cross(c1, c2, dim=-1)
        return torch.stack((c1, c2, c3), dim=-1)

    @staticmethod
    def _unnormalize_actions(actions, data):
        mode = data.get("_action_norm_mode")
        if mode == "quantile":
            low = data["_action_norm_quantile_1"].to(actions)
            high = data["_action_norm_quantile_99"].to(actions)
            return (actions + 1.0) * 0.5 * (high - low + 1e-6) + low
        if mode == "minmax":
            low = data["_action_norm_min"].to(actions)
            high = data["_action_norm_max"].to(actions)
            return (actions + 1.0) * 0.5 * (high - low + 1e-6) + low
        if mode == "zscore":
            mean = data["_action_norm_mean"].to(actions)
            std = data["_action_norm_std"].to(actions)
            return actions * (std + 1e-6) + mean
        raise ValueError(
            "rotation_geodesic_weight requires action normalization metadata"
        )

    def _rotation_geodesic_loss(self, predicted_actions, target_actions, data, weights):
        if predicted_actions.shape[-1] < 9:
            raise ValueError(
                "rotation geodesic loss requires xyz + 6D rotation action dimensions"
            )
        pred_raw = self._unnormalize_actions(predicted_actions, data)
        target_raw = self._unnormalize_actions(target_actions, data)
        pred_rot = self._rotation_6d_to_matrix(pred_raw[..., 3:9])
        target_rot = self._rotation_6d_to_matrix(target_raw[..., 3:9])
        relative = pred_rot.transpose(-1, -2) @ target_rot
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(
            -1.0, 1.0
        )
        skew = relative - relative.transpose(-1, -2)
        sine = torch.linalg.vector_norm(skew, dim=(-2, -1)) / (2.0 * (2.0**0.5))
        angle = torch.atan2(sine, cosine)
        time_weights = weights.squeeze(-1)
        return (angle * time_weights).sum() / (
            angle.shape[0] * time_weights.sum().clamp_min(1e-6)
        )

    @override
    def compute_loss(self, global_cond, data):
        actions, global_cond = self.preprocess_compute_loss(global_cond, data)
        pred, target, x_t, time = self._predict_with_state(actions, global_cond)
        weights = self._temporal_weights(actions)
        squared_error = (pred - target).square()
        flow_loss = (squared_error * weights).sum() / (
            actions.shape[0] * actions.shape[-1] * weights.sum().clamp_min(1e-6)
        )

        if self.rotation_geodesic_weight == 0.0:
            return flow_loss

        # x_t = action + t * velocity, hence action_hat = x_t - t * v_hat.
        predicted_actions = x_t - time[:, None, None] * pred
        rotation_loss = self._rotation_geodesic_loss(
            predicted_actions, actions, data, weights
        )
        return flow_loss + self.rotation_geodesic_weight * rotation_loss
