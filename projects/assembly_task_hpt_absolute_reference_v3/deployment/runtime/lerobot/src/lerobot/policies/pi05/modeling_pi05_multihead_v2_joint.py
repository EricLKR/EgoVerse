#!/usr/bin/env python
"""Inference-only PIPER joint head for EgoVerse pi0.5 MultiHead V2.

This module mirrors the parameter layout of
``egomimic.models.pi05_egoverse_multihead_v2.Pi05PrefixJointHeadV2`` while
deliberately excluding training-only FK, camera-consistency, and loss code.
Keeping deployment independent from EgoMimic prevents a robot runtime from
silently importing a different training checkout.

The checkpoint contract is:

* ``nets.policy.*`` produces the shared PaliGemma visual/language prefix;
* ``nets.joint_head.*`` predicts 30 native PIPER control waypoints;
* those waypoints are linearly resampled to 100 six-joint targets;
* the seventh channel is a separately classified binary gripper state.

No flow-action channels, FK, or IK are used to produce robot commands.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks


class Pi05PrefixJointHeadV2Deployment(nn.Module):
    """Strict inference mirror of the trained MultiHead V2 PIPER head."""

    contract_name = "egoverse_pi05_multihead_v2"
    contract_version = 2

    def __init__(
        self,
        *,
        prefix_dim: int = 2048,
        state_dim: int = 7,
        hidden_dim: int = 256,
        action_horizon: int = 100,
        control_horizon: int = 30,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_residual_deg: float = 90.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.joint_dims = self.state_dim - 1
        self.action_horizon = int(action_horizon)
        self.control_horizon = int(control_horizon)
        if self.state_dim != 7:
            raise ValueError("MultiHead V2 requires six joints plus one gripper state")

        # Names and shapes below intentionally match the training module.
        self.prefix_projection = nn.Linear(prefix_dim, hidden_dim)
        self.state_projection = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
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
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.joint_residual_head = nn.Linear(hidden_dim, self.joint_dims)
        self.gripper_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.control_queries, std=0.02)
        nn.init.zeros_(self.joint_residual_head.weight)
        nn.init.zeros_(self.joint_residual_head.bias)
        nn.init.zeros_(self.gripper_head.weight)
        gripper_open_prior = 0.716
        nn.init.constant_(
            self.gripper_head.bias,
            math.log(gripper_open_prior / (1.0 - gripper_open_prior)),
        )
        self.max_residual_rad = math.radians(float(max_residual_deg))

        # Both buffers are persistent in the training checkpoint.  They are
        # unused for inference, but registering them makes strict loading prove
        # that the deployment head has the complete training contract.
        self.register_buffer(
            "joint_dimension_weights", torch.ones(self.joint_dims), persistent=True
        )
        self.register_buffer("camera_from_world", torch.eye(4), persistent=True)

    def _decode(
        self,
        *,
        prefix_features: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        normalized_state_history: torch.Tensor,
        current_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prefix_features.ndim != 3 or prefix_pad_mask.ndim != 2:
            raise ValueError("prefix features/mask must be [B,N,D] and [B,N]")
        if prefix_features.shape[:2] != prefix_pad_mask.shape:
            raise ValueError("prefix feature and mask token axes do not match")
        state = normalized_state_history.reshape(
            normalized_state_history.shape[0], -1, self.state_dim
        )
        if current_raw.shape != (state.shape[0], self.state_dim):
            raise ValueError(
                f"current_raw must be [B,{self.state_dim}], got {tuple(current_raw.shape)}"
            )

        state_memory = self.state_projection(state.float())
        memory = torch.cat(
            (self.prefix_projection(prefix_features.float()), state_memory), dim=1
        )
        state_mask = torch.ones(
            state.shape[:2], dtype=torch.bool, device=prefix_pad_mask.device
        )
        memory_mask = torch.cat((prefix_pad_mask.bool(), state_mask), dim=1)
        queries = (
            self.control_queries.expand(memory.shape[0], -1, -1)
            + state_memory[:, -1:, :]
        )
        decoded = self.decoder(
            queries,
            memory,
            memory_key_padding_mask=~memory_mask,
        )
        residual = (
            torch.tanh(self.joint_residual_head(decoded)) * self.max_residual_rad
        )
        predicted_joints = current_raw[:, None, : self.joint_dims] + residual
        gripper_logits = self.gripper_head(decoded).squeeze(-1)
        return predicted_joints, gripper_logits

    def predict_with_details(
        self,
        *,
        prefix_features: torch.Tensor,
        prefix_pad_mask: torch.Tensor,
        normalized_state_history: torch.Tensor,
        current_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the 100x7 physical action chunk and 30 native logits."""
        control_joints, gripper_logits = self._decode(
            prefix_features=prefix_features,
            prefix_pad_mask=prefix_pad_mask,
            normalized_state_history=normalized_state_history,
            current_raw=current_raw,
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

    def predict(self, **kwargs) -> torch.Tensor:
        return self.predict_with_details(**kwargs)[0]


@torch.no_grad()
def extract_pi05_prefix_features(
    policy: nn.Module,
    images: list[torch.Tensor],
    image_masks: list[torch.Tensor],
    tokens: torch.Tensor,
    token_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exactly the prefix path used by MultiHead V2 evaluation."""
    prefix, prefix_pad, prefix_attention = policy.embed_prefix(
        images, image_masks, tokens, token_masks
    )
    first_q_projection = (
        policy.paligemma_with_expert.paligemma.language_model.layers[0]
        .self_attn.q_proj.weight
    )
    if first_q_projection.dtype == torch.bfloat16:
        prefix = prefix.to(torch.bfloat16)
    attention_4d = policy._prepare_attention_masks_4d(
        make_att_2d_masks(prefix_pad, prefix_attention)
    )
    position_ids = torch.cumsum(prefix_pad, dim=1) - 1
    # LeRobot's pinned Transformers build otherwise selects SDPA, whose BF16
    # query path rejects the FP32 additive mask produced by the OpenPI helper.
    # The existing PI05 ``sample_actions`` path makes this same explicit eager
    # selection before its prefix forward.
    policy.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"
    )
    (prefix_output, _), _ = policy.paligemma_with_expert.forward(
        attention_mask=attention_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix, None],
        use_cache=False,
    )
    return prefix_output.float(), prefix_pad
