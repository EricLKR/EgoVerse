#!/usr/bin/env python
"""Additive RTC deployment for EgoVerse pi0.5 PIPER V3/V4.

This file intentionally does not modify or replace
``deploy_egoverse_pi05_piper_v3_v4_joint.py``.  It reuses that audited
checkpoint loader, image/proprio preprocessing, joint decoder, driver safety
filter, and gripper state machine, while adding Physical Intelligence style
Real-Time Chunking (RTC):

* a background inference thread replans while the control loop keeps running;
* the unconsumed 32-D Flow-Matching chunk and decoded 6-D joint trajectory are
  supplied to the next sampler;
* the new chunk is prefix-guided during Euler integration using continuity in
  decoded physical joint space (with a latent compatibility mode);
* the measured number of successfully executed control ticks is discarded
  from the new chunk when it replaces the old queue;
* gripper logits stay outside continuous RTC guidance and retain the existing
  binary, feedback-aware state machine.

``checkpoint`` and ``offline`` never construct a robot. ``live-dry-run`` opens
camera/CAN feedback observation-only. ``execute`` requires ``--allow-motion``.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import queue
import sys
import threading
import time
import traceback
import types
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import deploy_egoverse_hpt_joint as legacy
import deploy_egoverse_pi05_piper_v3_v4_joint as base
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger("deploy_egoverse_pi05_piper_v3_v4_rtc_joint")

CONTROL_HZ = 30.0
MODEL_HORIZON = 30
MODEL_ACTION_DIM = 32
PIPER_POSE_AND_JOINT_CHANNELS = slice(0, 15)
PIPER_JOINT_CHANNELS = slice(9, 15)


def _decode_physical_joints(
    action32: torch.Tensor,
    current_raw: torch.Tensor,
    auxiliary: torch.nn.Module,
) -> torch.Tensor:
    """Differentiably decode V3/V4 joint channels to absolute radians."""
    if action32.ndim != 3 or action32.shape[-1] != MODEL_ACTION_DIM:
        raise ValueError("action32 must be [B,T,32]")
    if current_raw.ndim != 2 or current_raw.shape[-1] < 6:
        raise ValueError("current_raw must be [B,6+] for RTC joint decoding")
    latent = action32[..., PIPER_JOINT_CHANNELS]
    version = int(getattr(auxiliary, "contract_version", -1))
    if version == 4:
        scale = auxiliary.joint_residual_scale_rad_vector.to(latent)
        delta = scale[None, None, :] * torch.tanh(latent)
    elif version == 3:
        scale = torch.as_tensor(
            auxiliary.joint_residual_scale_rad,
            device=latent.device,
            dtype=latent.dtype,
        )
        delta = scale * latent
    else:
        raise RuntimeError(
            f"decoded-joint RTC does not support auxiliary contract version {version}"
        )
    return current_raw[:, None, :6].to(latent) + delta


def _guided_channel_mask(mode: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros((1, 1, MODEL_ACTION_DIM), device=device, dtype=dtype)
    if mode == "joint":
        mask[..., PIPER_JOINT_CHANNELS] = 1.0
    elif mode == "pose-joint":
        mask[..., PIPER_POSE_AND_JOINT_CHANNELS] = 1.0
    elif mode == "all":
        mask.fill_(1.0)
    else:  # pragma: no cover - guarded by argparse.
        raise ValueError(f"unsupported RTC guided-channel mode: {mode}")
    return mask


class MaskedRTCProcessor(RTCProcessor):
    """RTC guidance in decoded joints, with masked latent compatibility.

    Expert-Action V3/V4 supervises shared pose channels 0:9 and PIPER joint
    channels 9:15.  The remaining padded OpenPI channels are not robot actions,
    so using them as RTC constraints can inject arbitrary gradients into the
    action expert.  This processor keeps LeRobot's RTC time-prefix weighting
    while multiplying its correction by an explicit action-channel mask.
    """

    def __init__(
        self,
        rtc_config: RTCConfig,
        guided_channels: str,
        guidance_space: str = "latent",
    ) -> None:
        super().__init__(rtc_config)
        self.guided_channels = str(guided_channels)
        self.guidance_space = str(guidance_space)
        if self.guidance_space not in ("latent", "decoded-joint"):
            raise ValueError(f"unsupported RTC guidance space: {guidance_space}")

    def denoise_step(
        self,
        x_t: torch.Tensor,
        prev_chunk_left_over: torch.Tensor | None,
        inference_delay: int,
        time_value: torch.Tensor,
        original_denoise_step_partial,
        execution_horizon: int | None = None,
        prev_chunk_decoded_joints: torch.Tensor | None = None,
        current_raw: torch.Tensor | None = None,
        joint_decoder: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        if prev_chunk_left_over is None or prev_chunk_left_over.shape[-2] == 0:
            return original_denoise_step_partial(x_t)

        squeeze = x_t.ndim == 2
        x = x_t.unsqueeze(0) if squeeze else x_t
        x = x.clone().detach().requires_grad_(True)
        previous = (
            prev_chunk_left_over.unsqueeze(0)
            if prev_chunk_left_over.ndim == 2
            else prev_chunk_left_over
        ).to(device=x.device, dtype=x.dtype)
        if previous.shape[0] == 1 and x.shape[0] > 1:
            previous = previous.expand(x.shape[0], -1, -1)
        if previous.shape[0] != x.shape[0] or previous.shape[2] != x.shape[2]:
            raise ValueError(
                "RTC previous chunk has incompatible shape: "
                f"previous={tuple(previous.shape)} current={tuple(x.shape)}"
            )

        horizon = int(
            self.rtc_config.execution_horizon
            if execution_horizon is None
            else execution_horizon
        )
        horizon = min(horizon, previous.shape[1], x.shape[1])
        if horizon <= 0:
            return original_denoise_step_partial(x_t)
        delay = max(0, min(int(inference_delay), horizon))

        padded = torch.zeros_like(x)
        copy_steps = min(previous.shape[1], x.shape[1])
        padded[:, :copy_steps] = previous[:, :copy_steps]
        weights = self.get_prefix_weights(delay, horizon, x.shape[1]).to(
            device=x.device, dtype=x.dtype
        )[None, :, None]
        channel_mask = _guided_channel_mask(
            self.guided_channels, device=x.device, dtype=x.dtype
        )

        # Keep the official RTC Jacobian-vector correction.  It is not
        # equivalent to a direct residual because the action-expert velocity
        # itself depends on x_t.  The channel mask both selects supervised
        # targets for the VJP and prevents padded OpenPI channels from being
        # altered by the correction.
        with torch.enable_grad():
            velocity = original_denoise_step_partial(x)
            estimate_clean = x - time_value * velocity
            if self.guidance_space == "decoded-joint":
                if (
                    prev_chunk_decoded_joints is None
                    or current_raw is None
                    or joint_decoder is None
                ):
                    raise ValueError(
                        "decoded-joint RTC requires old physical joints, "
                        "current_raw, and the checkpoint joint decoder"
                    )
                old_joints = prev_chunk_decoded_joints
                if old_joints.ndim == 2:
                    old_joints = old_joints.unsqueeze(0)
                old_joints = old_joints.to(device=x.device, dtype=x.dtype)
                if old_joints.shape[0] == 1 and x.shape[0] > 1:
                    old_joints = old_joints.expand(x.shape[0], -1, -1)
                if old_joints.ndim != 3 or old_joints.shape[-1] != 6:
                    raise ValueError(
                        "previous decoded joints must be [B,T,6], got "
                        f"{tuple(old_joints.shape)}"
                    )
                decoded = _decode_physical_joints(
                    estimate_clean, current_raw.to(x), joint_decoder
                )
                padded_joints = torch.zeros_like(decoded)
                joint_steps = min(old_joints.shape[1], decoded.shape[1])
                padded_joints[:, :joint_steps] = old_joints[:, :joint_steps]
                version = int(getattr(joint_decoder, "contract_version", -1))
                if version == 4:
                    error_scale = joint_decoder.joint_residual_scale_rad_vector.to(
                        device=x.device, dtype=x.dtype
                    )[None, None, :]
                elif version == 3:
                    error_scale = torch.as_tensor(
                        joint_decoder.joint_residual_scale_rad,
                        device=x.device,
                        dtype=x.dtype,
                    ).reshape(1, 1, 1)
                else:  # Kept explicit even though the decoder checked above.
                    raise RuntimeError(
                        f"unsupported decoded-joint RTC contract version {version}"
                    )
                error_scale = error_scale.clamp_min(math.radians(1.0))
                # Normalize exactly once per physical degree of freedom using
                # the checkpoint's decoder range.  A small training-loss scale
                # (for example 5 degrees) would enter both the residual and
                # Jacobian and amplify guidance quadratically.
                decoded_normalized = decoded / error_scale
                target_normalized = padded_joints / error_scale
                error = (target_normalized - decoded_normalized) * weights
                correction = torch.autograd.grad(
                    decoded_normalized,
                    x,
                    grad_outputs=error.detach(),
                    retain_graph=False,
                )[0]
            else:
                error = (padded - estimate_clean) * weights * channel_mask
                correction = torch.autograd.grad(
                    estimate_clean,
                    x,
                    grad_outputs=error.detach(),
                    retain_graph=False,
                )[0]
                correction = correction * channel_mask

        tau = 1.0 - time_value
        max_weight = torch.as_tensor(
            self.rtc_config.max_guidance_weight,
            device=x.device,
            dtype=x.dtype,
        )
        one_minus_tau_sq = (1.0 - tau) ** 2
        inv_r2 = (one_minus_tau_sq + tau**2) / one_minus_tau_sq.clamp_min(1e-12)
        coefficient = torch.nan_to_num(
            (1.0 - tau) / tau,
            posinf=float(self.rtc_config.max_guidance_weight),
        )
        guidance_weight = torch.minimum(
            torch.nan_to_num(
                coefficient * inv_r2,
                posinf=float(self.rtc_config.max_guidance_weight),
            ),
            max_weight,
        )
        result = velocity - guidance_weight * correction
        if squeeze:
            result = result.squeeze(0)
            estimate_clean = estimate_clean.squeeze(0)
            error = error.squeeze(0)
            correction = correction.squeeze(0)
            weights = weights.squeeze(0)

        self.track(
            time=time_value,
            x1_t=estimate_clean,
            correction=correction,
            err=error,
            weights=weights,
            guidance_weight=guidance_weight,
            inference_delay=delay,
            execution_horizon=horizon,
            guided_channels=self.guided_channels,
            guidance_space=self.guidance_space,
        )
        return result


@torch.no_grad()
def _sample_actions_with_hidden_rtc(
    policy,
    *,
    device: torch.device | str,
    observation,
    rtc_processor: MaskedRTCProcessor,
    prev_chunk_left_over: torch.Tensor | None,
    prev_chunk_decoded_joints: torch.Tensor | None,
    inference_delay: int,
    current_raw: torch.Tensor,
    joint_decoder: torch.nn.Module,
    noise: torch.Tensor | None = None,
    num_steps: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the existing V3/V4 Euler sampler with RTC prefix guidance."""
    from egomimic.algo.pi_egoverse_expert_action_v3 import (
        _denoise_step_with_hidden_v3,
    )
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = policy.sample_noise(
            (batch_size, policy.config.action_horizon, policy.config.action_dim),
            device,
        )
    if tuple(noise.shape) != (batch_size, MODEL_HORIZON, MODEL_ACTION_DIM):
        raise ValueError(f"unexpected Flow noise shape: {tuple(noise.shape)}")

    images, masks, lang, lang_masks, state = policy._preprocess_observation(
        observation, train=False
    )
    prefix, prefix_pad, prefix_attention = policy.embed_prefix(
        images, masks, lang, lang_masks
    )
    prefix_mask = make_att_2d_masks(prefix_pad, prefix_attention)
    prefix_position_ids = torch.cumsum(prefix_pad, dim=1) - 1
    policy.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"
    )
    _, cache = policy.paligemma_with_expert.forward(
        attention_mask=policy._prepare_attention_masks_4d(prefix_mask),
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix, None],
        use_cache=True,
    )

    previous = None
    if prev_chunk_left_over is not None:
        previous = prev_chunk_left_over.to(device=device, dtype=noise.dtype)
        if previous.ndim == 2:
            previous = previous.unsqueeze(0)
        if previous.ndim != 3 or previous.shape[0] != batch_size:
            raise ValueError(
                "prev_chunk_left_over must be [T,32] or [B,T,32], got "
                f"{tuple(previous.shape)}"
            )

    delta_time = torch.tensor(-1.0 / int(num_steps), dtype=torch.float32, device=device)
    noisy_action = noise
    time_value = torch.tensor(1.0, dtype=torch.float32, device=device)
    final_hidden = None
    while time_value >= -delta_time / 2:
        hidden_box: dict[str, torch.Tensor] = {}

        def base_denoise(current: torch.Tensor) -> torch.Tensor:
            velocity, hidden = _denoise_step_with_hidden_v3(
                policy,
                state,
                prefix_pad,
                cache,
                current,
                time_value.expand(batch_size),
            )
            hidden_box["hidden"] = hidden
            return velocity

        velocity = rtc_processor.denoise_step(
            noisy_action,
            previous,
            int(inference_delay),
            time_value,
            base_denoise,
            prev_chunk_decoded_joints=prev_chunk_decoded_joints,
            current_raw=current_raw,
            joint_decoder=joint_decoder,
        )
        final_hidden = hidden_box.get("hidden")
        noisy_action = noisy_action + delta_time * velocity
        time_value += delta_time

    if final_hidden is None:
        raise RuntimeError("RTC Flow sampler produced no action-expert hidden state")
    return noisy_action, final_hidden


@torch.no_grad()
def _forward_joint_deploy_rtc(
    algo,
    *,
    rtc_processor: MaskedRTCProcessor,
    prev_chunk_left_over: torch.Tensor | None,
    prev_chunk_decoded_joints: torch.Tensor | None,
    inference_delay: int,
    front_image: torch.Tensor,
    wrist_image: torch.Tensor,
    current_raw: torch.Tensor,
    prompt: str | None = None,
    noise: torch.Tensor | None = None,
    num_steps: int | None = None,
) -> dict[str, torch.Tensor | list[str]]:
    """Exact training wrapper forward with RTC substituted only in sampling."""
    from egomimic.rldb.embodiment.embodiment import get_embodiment

    embodiment_id = algo.piper_embodiment_id
    state_key = next(
        (key for key in algo.proprio_keys[embodiment_id] if key.endswith(".joints")),
        None,
    )
    if state_key is None:
        raise RuntimeError("checkpoint has no PIPER joint proprio key")
    current_raw = current_raw.to(algo.device, dtype=torch.float32)
    if current_raw.ndim != 2 or current_raw.shape[1] != 7:
        raise ValueError("current_raw must be [B,7]")
    if not torch.isfinite(current_raw).all():
        raise ValueError("current_raw contains NaN/Inf")
    batch_size = current_raw.shape[0]
    if front_image.shape[0] != batch_size or wrist_image.shape[0] != batch_size:
        raise ValueError("image and proprio batch sizes differ")

    stats = algo.norm_stats.norm_stats.get(embodiment_id, {}).get(state_key)
    if stats is None:
        raise RuntimeError(f"checkpoint lacks normalization stats for {state_key}")
    normalized_state = algo.norm_stats._apply_norm_one(current_raw, stats)
    name = get_embodiment(embodiment_id).lower()
    if prompt is None:
        prompts = algo._build_prompts({}, name, batch_size)
    else:
        embodiment_label = name.replace("_", " ")
        prompts = []
        for _ in range(batch_size):
            blocks = [str(prompt)]
            if algo.embodiment_label:
                blocks.append(f"Embodiment: {embodiment_label}")
            if algo.control_mode:
                blocks.append(
                    f"Control mode: {algo._control_mode_for(embodiment_label)}"
                )
            prompts.append(", ".join(blocks))
    states = (
        [row.detach().cpu().numpy() for row in normalized_state]
        if algo.proprio_in_prompt
        else [None] * batch_size
    )
    tokenized = {
        key: value.to(algo.device)
        for key, value in algo._tokenize_prompts(prompts, states).items()
    }
    model_batch = {
        "observations.images.front_img_1": front_image.to(algo.device),
        "observations.images.wrist_img": wrist_image.to(algo.device),
        state_key: normalized_state[:, None],
        **tokenized,
    }
    ac_key = algo.ac_keys[embodiment_id]
    model_batch[ac_key] = torch.zeros(
        batch_size, MODEL_HORIZON, 9, device=algo.device, dtype=torch.float32
    )
    observation, _, _ = algo._robomimic_to_pi_data(
        model_batch,
        algo.camera_keys.get(embodiment_id, algo.pi_cam_keys),
        algo.proprio_keys[embodiment_id],
        algo.lang_keys[embodiment_id],
        ac_key,
        name,
    )
    sampled, hidden = _sample_actions_with_hidden_rtc(
        algo.nets["policy"],
        device=algo.device,
        observation=observation,
        rtc_processor=rtc_processor,
        prev_chunk_left_over=prev_chunk_left_over,
        prev_chunk_decoded_joints=prev_chunk_decoded_joints,
        inference_delay=int(inference_delay),
        current_raw=current_raw,
        joint_decoder=algo.nets["auxiliary"],
        noise=noise,
        num_steps=algo.num_steps if num_steps is None else int(num_steps),
    )
    actions, gripper_logits = algo.nets["auxiliary"].decode_actions(
        sampled_action32=sampled,
        suffix_hidden=hidden,
        current_raw=current_raw,
    )
    return {
        "actions": actions,
        "action32": sampled,
        "gripper_logits": gripper_logits,
        "normalized_state": normalized_state,
        "prompts": prompts,
    }


class EgoVersePI05PiperV3V4RTCJointPolicy(
    base.EgoVersePI05PiperV3V4JointPolicy
):
    """V3/V4 policy whose Flow sampler accepts an RTC action prefix."""

    name = "egoverse_pi05_piper_v3_v4_rtc_joint"

    def __init__(
        self,
        args: argparse.Namespace,
        joint_limits_deg: list[tuple[float, float]],
        gripper_stroke_mm: float,
    ) -> None:
        super().__init__(args, joint_limits_deg, gripper_stroke_mm)
        schedule = RTCAttentionSchedule(str(args.rtc_prefix_attention_schedule).upper())
        self.rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=schedule,
            max_guidance_weight=float(args.rtc_max_guidance_weight),
            execution_horizon=int(args.rtc_execution_horizon),
            debug=bool(args.rtc_debug),
            debug_maxlen=int(args.rtc_debug_maxlen),
        )
        self.rtc_processor = MaskedRTCProcessor(
            self.rtc_config,
            args.rtc_guided_channels,
            guidance_space=args.rtc_guidance_space,
        )
        # RTC works on all 30 native model ticks. Replanning cadence is managed
        # by the external real-time queue, not by the legacy query_frequency
        # deque or temporal ensemble.
        self.config.query_frequency = MODEL_HORIZON
        self.config.temporal_ensemble = False
        self.config.chunk_smoothing = False
        self.checkpoint_report.update(
            {
                "rtc": True,
                "rtc_mode": (
                    "decoded_joint_guided_flow_inpainting"
                    if args.rtc_guidance_space == "decoded-joint"
                    else "latent_guided_flow_inpainting"
                ),
                "rtc_execution_horizon": int(args.rtc_execution_horizon),
                "rtc_max_guidance_weight": float(args.rtc_max_guidance_weight),
                "rtc_prefix_attention_schedule": str(
                    args.rtc_prefix_attention_schedule
                ),
                "rtc_guided_channels": str(args.rtc_guided_channels),
                "rtc_guidance_space": str(args.rtc_guidance_space),
                "rtc_async_control": True,
                "rtc_control_hz": float(args.rtc_control_hz),
            }
        )

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
        *,
        prev_chunk_left_over: torch.Tensor | None = None,
        prev_chunk_decoded_joints: torch.Tensor | None = None,
        inference_delay: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs:
            raise TypeError(f"unexpected RTC prediction arguments: {sorted(kwargs)}")
        if inference_delay < 0 or inference_delay >= MODEL_HORIZON:
            raise ValueError(
                f"inference_delay must be in [0,{MODEL_HORIZON - 1}]"
            )
        context_previous = (
            None
            if prev_chunk_left_over is None
            else prev_chunk_left_over.detach().clone()
        )
        context_decoded_joints = (
            None
            if prev_chunk_decoded_joints is None
            else prev_chunk_decoded_joints.detach().clone()
        )
        had_instance_override = "forward_joint_deploy" in self._algo.__dict__
        prior_instance_value = self._algo.__dict__.get("forward_joint_deploy")

        def rtc_forward(algo_self, **call_kwargs):
            return _forward_joint_deploy_rtc(
                algo_self,
                rtc_processor=self.rtc_processor,
                prev_chunk_left_over=context_previous,
                prev_chunk_decoded_joints=context_decoded_joints,
                inference_delay=int(inference_delay),
                **call_kwargs,
            )

        setattr(
            self._algo,
            "forward_joint_deploy",
            types.MethodType(rtc_forward, self._algo),
        )
        try:
            result = super().predict_action_chunk(batch)
        finally:
            if had_instance_override:
                setattr(self._algo, "forward_joint_deploy", prior_instance_value)
            else:
                delattr(self._algo, "forward_joint_deploy")

        report = self.last_prediction_report
        if report is None:
            raise RuntimeError("RTC prediction did not produce an audit report")
        report["rtc"] = {
            "enabled": True,
            "has_previous_prefix": context_previous is not None,
            "previous_prefix_steps": (
                0 if context_previous is None else int(context_previous.shape[-2])
            ),
            "predicted_inference_delay_ticks": int(inference_delay),
            "execution_horizon": int(self.rtc_config.execution_horizon),
            "max_guidance_weight": float(self.rtc_config.max_guidance_weight),
            "prefix_attention_schedule": str(
                self.rtc_config.prefix_attention_schedule.value
            ),
            "guided_channels": self.rtc_processor.guided_channels,
            "guidance_space": self.rtc_processor.guidance_space,
            "previous_decoded_joint_steps": (
                0
                if context_decoded_joints is None
                else int(context_decoded_joints.shape[-2])
            ),
        }
        return result


@dataclass(frozen=True)
class ReplanSnapshot:
    generation: int
    start_index: int
    start_tick: int
    original_left_over: torch.Tensor
    physical_left_over: torch.Tensor
    processed_left_over: torch.Tensor


class RTCCommandQueue:
    """Thread-safe peek/commit queue preserving model-space RTC prefixes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._original: torch.Tensor | None = None
        self._physical: torch.Tensor | None = None
        self._processed: torch.Tensor | None = None
        self._index = 0
        self._tick = 0
        self._generation = 0

    @staticmethod
    def _validate(
        original: torch.Tensor,
        physical: torch.Tensor,
        processed: torch.Tensor,
    ) -> None:
        if original.ndim != 2 or tuple(original.shape[1:]) != (MODEL_ACTION_DIM,):
            raise ValueError(f"original RTC chunk must be [T,32], got {tuple(original.shape)}")
        if processed.ndim != 2 or tuple(processed.shape[1:]) != (7,):
            raise ValueError(f"processed RTC chunk must be [T,7], got {tuple(processed.shape)}")
        if physical.ndim != 2 or tuple(physical.shape[1:]) != (7,):
            raise ValueError(f"physical RTC chunk must be [T,7], got {tuple(physical.shape)}")
        if (
            len(original) != len(physical)
            or len(original) != len(processed)
            or len(original) != MODEL_HORIZON
        ):
            raise ValueError(
                "RTC chunks must contain 30 aligned steps: "
                f"original={tuple(original.shape)} physical={tuple(physical.shape)} "
                f"processed={tuple(processed.shape)}"
            )
        if (
            not torch.isfinite(original).all()
            or not torch.isfinite(physical).all()
            or not torch.isfinite(processed).all()
        ):
            raise ValueError("RTC queue refuses NaN/Inf")

    def load_initial(
        self,
        original: torch.Tensor,
        physical: torch.Tensor,
        processed: torch.Tensor,
    ) -> int:
        self._validate(original, physical, processed)
        with self._lock:
            if self._original is not None:
                raise RuntimeError("RTC queue was already initialized")
            self._original = original.detach().cpu().clone()
            self._physical = physical.detach().cpu().clone()
            self._processed = processed.detach().cpu().clone()
            self._index = 0
            self._generation = 1
            return self._generation

    def qsize(self) -> int:
        with self._lock:
            if self._processed is None:
                return 0
            return max(0, len(self._processed) - self._index)

    def tick(self) -> int:
        with self._lock:
            return int(self._tick)

    def peek(self) -> tuple[torch.Tensor, dict[str, int]] | None:
        with self._lock:
            if self._processed is None or self._index >= len(self._processed):
                return None
            return self._processed[self._index].clone(), {
                "generation": int(self._generation),
                "index": int(self._index),
                "tick": int(self._tick),
                "pending_actions": int(len(self._processed) - self._index),
            }

    def advance(self) -> dict[str, int]:
        with self._lock:
            if self._processed is None or self._index >= len(self._processed):
                raise RuntimeError("cannot advance an empty RTC queue")
            self._index += 1
            self._tick += 1
            return {
                "generation": int(self._generation),
                "index": int(self._index),
                "tick": int(self._tick),
                "pending_actions": int(len(self._processed) - self._index),
            }

    def snapshot_for_replan(self) -> ReplanSnapshot:
        with self._lock:
            if self._original is None or self._physical is None or self._processed is None:
                raise RuntimeError("cannot replan before initial RTC chunk")
            return ReplanSnapshot(
                generation=int(self._generation),
                start_index=int(self._index),
                start_tick=int(self._tick),
                original_left_over=self._original[self._index :].clone(),
                physical_left_over=self._physical[self._index :].clone(),
                processed_left_over=self._processed[self._index :].clone(),
            )

    def merge(
        self,
        snapshot: ReplanSnapshot,
        original: torch.Tensor,
        physical: torch.Tensor,
        processed: torch.Tensor,
    ) -> dict[str, Any]:
        self._validate(original, physical, processed)
        with self._lock:
            if snapshot.generation != self._generation:
                raise RuntimeError(
                    "RTC queue generation changed during inference: "
                    f"request={snapshot.generation} current={self._generation}"
                )
            actual_delay = int(self._tick - snapshot.start_tick)
            index_delay = int(self._index - snapshot.start_index)
            if actual_delay != index_delay:
                raise RuntimeError(
                    "RTC action/tick lockstep broke during inference: "
                    f"tick_delay={actual_delay} index_delay={index_delay}"
                )
            if actual_delay >= MODEL_HORIZON:
                raise RuntimeError(
                    "RTC inference exhausted the entire new chunk before merge: "
                    f"delay={actual_delay} horizon={MODEL_HORIZON}"
                )

            old_model = None
            old_physical = None
            old_driver = None
            if actual_delay < len(snapshot.original_left_over):
                old_model = snapshot.original_left_over[actual_delay]
                old_physical = snapshot.physical_left_over[actual_delay]
                old_driver = snapshot.processed_left_over[actual_delay]
            new_model = original[actual_delay]
            new_physical = physical[actual_delay]
            new_driver = processed[actual_delay]
            model_joint_rms = (
                None
                if old_model is None
                else float(
                    torch.sqrt(
                        torch.mean(
                            (new_model[PIPER_JOINT_CHANNELS] - old_model[PIPER_JOINT_CHANNELS])
                            ** 2
                        )
                    ).item()
                )
            )
            driver_arm_rms = (
                None
                if old_driver is None
                else float(torch.sqrt(torch.mean((new_driver[:6] - old_driver[:6]) ** 2)).item())
            )
            physical_joint_rms_rad = (
                None
                if old_physical is None
                else float(
                    torch.sqrt(
                        torch.mean((new_physical[:6] - old_physical[:6]) ** 2)
                    ).item()
                )
            )

            self._original = original[actual_delay:].detach().cpu().clone()
            self._physical = physical[actual_delay:].detach().cpu().clone()
            self._processed = processed[actual_delay:].detach().cpu().clone()
            self._index = 0
            self._generation += 1
            return {
                "generation": int(self._generation),
                "actual_delay_ticks": actual_delay,
                "remaining_new_actions": int(len(self._processed)),
                "switch_model_joint_rms": model_joint_rms,
                "switch_physical_joint_rms_rad": physical_joint_rms_rad,
                "switch_physical_joint_rms_deg": (
                    None
                    if physical_joint_rms_rad is None
                    else math.degrees(physical_joint_rms_rad)
                ),
                "switch_driver_arm_rms": driver_arm_rms,
                "tick": int(self._tick),
            }

    def status(self) -> dict[str, int]:
        with self._lock:
            total = 0 if self._processed is None else len(self._processed)
            return {
                "generation": int(self._generation),
                "index": int(self._index),
                "tick": int(self._tick),
                "chunk_size_after_delay": int(total),
                "pending_actions": int(max(0, total - self._index)),
            }


class RTCGripperStateMachine(legacy.IndependentGripperStateMachine):
    def observe_prediction_horizon_at(
        self,
        chunk_id: int,
        model_commands_pct: Any,
        absolute_start_tick: int,
    ) -> dict[str, Any]:
        commands = self._validate_prediction_horizon(model_commands_pct)
        chunk = int(chunk_id)
        if any(existing == chunk for existing, _, _ in self._prediction_horizons):
            raise RuntimeError(f"duplicate gripper prediction horizon for chunk {chunk}")
        self._prediction_horizons.append(
            (chunk, int(absolute_start_tick), commands.copy())
        )
        close_offsets = np.flatnonzero(commands == 0.0)
        return {
            "chunk_id": chunk,
            "absolute_start_tick": int(absolute_start_tick),
            "horizon_steps": MODEL_HORIZON,
            "close_steps": int(close_offsets.size),
            "first_close_offset": int(close_offsets[0]) if close_offsets.size else None,
            "last_close_offset": int(close_offsets[-1]) if close_offsets.size else None,
            "commands_pct": commands.tolist(),
        }


@dataclass(frozen=True)
class ReplanRequest:
    snapshot: ReplanSnapshot
    observation: dict[str, Any]
    predicted_delay_ticks: int


def _copy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in observation.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
    return copied


class RTCInferenceWorker:
    """Single producer that replans without touching robot or logger objects."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        policy: EgoVersePI05PiperV3V4RTCJointPolicy,
        command_queue: RTCCommandQueue,
        device: torch.device,
        initial_inference_latency_s: float | None = None,
    ) -> None:
        self.args = args
        self.policy = policy
        self.command_queue = command_queue
        self.device = device
        self._jobs: queue.Queue[ReplanRequest | None] = queue.Queue(maxsize=1)
        self._completions: queue.Queue[dict[str, Any]] = queue.Queue()
        self._errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)
        self._latencies_s: deque[float] = deque(maxlen=int(args.rtc_latency_window))
        self._observed_delay_ticks: deque[int] = deque(
            maxlen=int(args.rtc_latency_window)
        )
        if initial_inference_latency_s is not None:
            latency = float(initial_inference_latency_s)
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(
                    "initial RTC inference latency must be finite and positive"
                )
            self._latencies_s.append(latency)
        self._busy_lock = threading.Lock()
        self._busy = False
        self._thread = threading.Thread(
            target=self._run,
            name="piper-pi05-rtc-inference",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _set_busy(self, value: bool) -> None:
        with self._busy_lock:
            self._busy = bool(value)

    def busy(self) -> bool:
        with self._busy_lock:
            return bool(self._busy)

    def estimated_delay_ticks(self) -> int:
        # Wall-clock / control-rate is only a conservative bootstrap.  Once a
        # replan has completed, the queue's successful action-tick delta is
        # the exact RTC alignment quantity (the main loop can briefly pause or
        # hold without advancing policy time).
        if self._observed_delay_ticks:
            delay = max(self._observed_delay_ticks)
        elif self._latencies_s:
            delay = math.ceil(max(self._latencies_s) * float(self.args.rtc_control_hz))
        else:
            delay = int(self.args.rtc_initial_delay_ticks)
        return max(0, min(int(self.args.rtc_max_delay_ticks), int(delay)))

    def maybe_submit(self, observation: dict[str, Any]) -> dict[str, Any] | None:
        if self.busy() or self.command_queue.qsize() > int(self.args.rtc_replan_threshold):
            return None
        request = ReplanRequest(
            snapshot=self.command_queue.snapshot_for_replan(),
            observation=_copy_observation(observation),
            predicted_delay_ticks=self.estimated_delay_ticks(),
        )
        self._set_busy(True)
        try:
            self._jobs.put_nowait(request)
        except Exception:
            self._set_busy(False)
            raise
        return {
            "event": "rtc_replan_submitted",
            "generation": request.snapshot.generation,
            "start_tick": request.snapshot.start_tick,
            "prefix_steps": int(len(request.snapshot.original_left_over)),
            "predicted_delay_ticks": request.predicted_delay_ticks,
            "queue_status": self.command_queue.status(),
            "command_sent": False,
        }

    def _run(self) -> None:
        while True:
            request = self._jobs.get()
            if request is None:
                return
            started = time.perf_counter()
            try:
                batch = legacy.live_observation_to_batch(request.observation, self.device)
                processed = self.policy.predict_action_chunk(
                    batch,
                    prev_chunk_left_over=request.snapshot.original_left_over,
                    prev_chunk_decoded_joints=(
                        request.snapshot.physical_left_over[:, :6]
                    ),
                    inference_delay=request.predicted_delay_ticks,
                ).detach().cpu()
                report = copy.deepcopy(self.policy.last_prediction_report)
                if report is None:
                    raise RuntimeError("RTC worker inference produced no report")
                original = torch.as_tensor(
                    report["full_model_action32"], dtype=torch.float32
                )
                physical = _physical_from_report(report)
                latency_s = time.perf_counter() - started
                self._latencies_s.append(latency_s)
                merge_report = self.command_queue.merge(
                    request.snapshot, original, physical, processed
                )
                self._observed_delay_ticks.append(
                    int(merge_report["actual_delay_ticks"])
                )
                self._completions.put(
                    {
                        "event": "rtc_replan_completed",
                        "generation": merge_report["generation"],
                        "start_tick": request.snapshot.start_tick,
                        "predicted_delay_ticks": request.predicted_delay_ticks,
                        "inference_latency_ms": latency_s * 1e3,
                        "estimated_next_delay_ticks": self.estimated_delay_ticks(),
                        "merge": merge_report,
                        "chunk_report": report,
                        "command_sent": False,
                    }
                )
            except BaseException as error:  # propagate worker failures to main.
                if self._errors.empty():
                    self._errors.put(error)
                self._completions.put(
                    {
                        "event": "rtc_replan_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "command_sent": False,
                    }
                )
            finally:
                self._set_busy(False)

    def drain_completions(self) -> list[dict[str, Any]]:
        records = []
        while True:
            try:
                records.append(self._completions.get_nowait())
            except queue.Empty:
                return records

    def raise_if_failed(self) -> None:
        try:
            error = self._errors.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("RTC inference worker failed") from error

    def stop(self) -> None:
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            return
        self._thread.join(timeout=2.0)


def _action32_from_report(report: dict[str, Any]) -> torch.Tensor:
    tensor = torch.as_tensor(report["full_model_action32"], dtype=torch.float32)
    if tuple(tensor.shape) != (MODEL_HORIZON, MODEL_ACTION_DIM):
        raise RuntimeError(f"invalid action32 report shape: {tuple(tensor.shape)}")
    return tensor


def _physical_from_report(report: dict[str, Any]) -> torch.Tensor:
    tensor = torch.as_tensor(report["full_model_chunk_physical"], dtype=torch.float32)
    if tuple(tensor.shape) != (MODEL_HORIZON, 7):
        raise RuntimeError(f"invalid physical chunk report shape: {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise RuntimeError("physical chunk report contains NaN/Inf")
    return tensor


def _make_gripper_controller(args: argparse.Namespace) -> RTCGripperStateMachine:
    return RTCGripperStateMachine(
        mode=args.gripper_control_mode,
        close_confirm_frames=args.gripper_close_confirm_frames,
        open_confirm_frames=args.gripper_open_confirm_frames,
        min_close_hold_frames=args.gripper_min_close_hold_frames,
        min_open_hold_frames=args.gripper_min_open_hold_frames,
        close_feedback_threshold=args.gripper_close_feedback_threshold,
        open_feedback_threshold=args.gripper_open_feedback_threshold,
        terminal_open_after_release=args.gripper_terminal_open_after_release,
    )


def run_live_rtc(
    args: argparse.Namespace,
    policy: EgoVersePI05PiperV3V4RTCJointPolicy,
    log: legacy.JsonlLogger,
    device: torch.device,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
    run_timestamp: str,
) -> None:
    execute = args.mode == "execute"
    if execute and not args.allow_motion:
        raise RuntimeError("execute mode requires --allow-motion")

    robot = legacy.make_robot(args, joint_limits, gripper_stroke_mm)
    robot.connect(calibrate=False)
    command_queue = RTCCommandQueue()
    gripper = _make_gripper_controller(args)
    worker: RTCInferenceWorker | None = None
    try:
        camera_report = legacy.configure_front_camera_auto_controls(robot)
        log.write(
            {
                "event": "camera_config",
                "camera": "435if",
                **camera_report,
                "command_sent": False,
            }
        )
        observation = robot.get_observation()
        warmup_started = time.perf_counter()
        initial_processed = policy.predict_action_chunk(
            legacy.live_observation_to_batch(observation, device),
            prev_chunk_left_over=None,
            inference_delay=0,
        ).detach().cpu()
        initial_report = copy.deepcopy(policy.last_prediction_report)
        if initial_report is None:
            raise RuntimeError("initial RTC inference produced no report")
        initial_original = _action32_from_report(initial_report)
        initial_physical = _physical_from_report(initial_report)
        generation = command_queue.load_initial(
            initial_original, initial_physical, initial_processed
        )
        initial_gripper_report = gripper.observe_prediction_horizon_at(
            generation,
            initial_report["full_control_gripper_commands_pct"],
            absolute_start_tick=0,
        )

        current_driver = legacy.observation_to_driver_state(
            observation, joint_limits, gripper_stroke_mm
        )
        first_safe, _ = policy._filter_driver_action(
            initial_processed[0].numpy(),
            current_driver[:6],
            policy._last_planned_velocity_driver,
        )
        continuity = legacy.initial_continuity_report(
            observation,
            torch.from_numpy(first_safe),
            joint_limits,
            args.max_initial_joint_delta_deg,
        )
        log.write(
            {
                "event": "warmup",
                "mode": args.mode,
                "total_step_ms": (time.perf_counter() - warmup_started) * 1e3,
                "initial_continuity": continuity,
                "chunk_report": initial_report,
                "gripper_horizon_update": initial_gripper_report,
                "rtc_queue": command_queue.status(),
                "command_sent": False,
            }
        )
        if execute:
            held = robot.enable_motion_hold_current()
            log.write(
                {"event": "motion_enable", "held_position": held, "command_sent": True}
            )

        worker = RTCInferenceWorker(
            args=args,
            policy=policy,
            command_queue=command_queue,
            device=device,
            initial_inference_latency_s=(
                float(initial_report["model_inference_ms"]) / 1e3
            ),
        )
        worker.start()
        started_run = time.perf_counter()
        last_raw_action: torch.Tensor | None = None
        starvation_hold_frames = 0
        for frame_index in range(args.max_steps):
            loop_started = time.perf_counter()
            for completion in worker.drain_completions():
                if completion["event"] == "rtc_replan_completed":
                    chunk_report = completion["chunk_report"]
                    completion["gripper_horizon_update"] = (
                        gripper.observe_prediction_horizon_at(
                            int(completion["generation"]),
                            chunk_report["full_control_gripper_commands_pct"],
                            int(completion["start_tick"]),
                        )
                    )
                log.write(completion)
            worker.raise_if_failed()

            observation = robot.get_observation()
            submitted = worker.maybe_submit(observation)
            if submitted is not None:
                log.write(submitted)
            queued = command_queue.peek()
            if queued is None:
                if not worker.busy() or last_raw_action is None:
                    raise RuntimeError(
                        "RTC action queue is empty and no inference can refill it"
                    )
                starvation_hold_frames += 1
                if starvation_hold_frames > int(args.rtc_max_starvation_hold_frames):
                    raise RuntimeError(
                        "RTC inference exceeded the bounded queue-starvation hold: "
                        f"{starvation_hold_frames} control frames"
                    )
                raw_action = last_raw_action.clone()
                queue_before = {
                    **command_queue.status(),
                    "starvation_hold": True,
                    "starvation_hold_frame": starvation_hold_frames,
                }
                queue_starved_hold = True
            else:
                raw_action, queue_before = queued
                last_raw_action = raw_action.clone()
                starvation_hold_frames = 0
                queue_starved_hold = False
            driver_state = legacy.observation_to_driver_state(
                observation, joint_limits, gripper_stroke_mm
            )
            safe_action, pending_velocity = policy._filter_driver_action(
                raw_action.numpy(),
                driver_state[:6],
                policy._last_planned_velocity_driver,
            )
            model_action = legacy.action_to_robot_dict(torch.from_numpy(safe_action))
            gripper_state = gripper.update_from_prediction_horizons(
                model_action["joint_6.pos"], driver_state[6]
            )
            robot_action = dict(model_action)
            robot_action["joint_6.pos"] = gripper_state["command_pct"]
            sent_action = robot.send_action(robot_action) if execute else None
            arm_clipped = None
            gripper_clipped = None
            if sent_action is not None:
                arm_clipped = any(
                    abs(
                        robot_action[f"joint_{index}.pos"]
                        - sent_action[f"joint_{index}.pos"]
                    )
                    > 1e-6
                    for index in range(6)
                )
                gripper_clipped = (
                    abs(robot_action["joint_6.pos"] - sent_action["joint_6.pos"])
                    > 1e-6
                )
            action_advanced = (
                not queue_starved_hold
                and (not execute or not bool(arm_clipped))
            )
            if action_advanced:
                queue_after = command_queue.advance()
                policy._last_planned_velocity_driver = pending_velocity
                gripper.advance_prediction_tick()
            else:
                queue_after = command_queue.status()

            image_paths = None
            if args.save_inference_images and submitted is not None:
                image_paths = legacy.save_inference_images(
                    args.log_dir,
                    run_timestamp,
                    int(queue_before["generation"]),
                    observation["435if"],
                    observation["405"],
                )
            log.write(
                {
                    "mode": args.mode,
                    "frame": frame_index,
                    "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": time.perf_counter() - started_run,
                    "total_step_ms": (time.perf_counter() - loop_started) * 1e3,
                    "raw_state": [
                        float(observation[f"joint_{index}.pos"]) for index in range(7)
                    ],
                    "driver_state": driver_state.tolist(),
                    "raw_queued_action": raw_action.tolist(),
                    "model_action": model_action,
                    "proposed_action": robot_action,
                    "sent_action": sent_action,
                    "arm_command_was_clipped": arm_clipped,
                    "gripper_command_was_clipped": gripper_clipped,
                    "action_advanced": action_advanced,
                    "rtc_queue_starvation_hold": queue_starved_hold,
                    "rtc_queue_starvation_hold_frames": starvation_hold_frames,
                    "rtc_queue_before": queue_before,
                    "rtc_queue_after": queue_after,
                    "rtc_worker_busy": worker.busy(),
                    "rtc_estimated_delay_ticks": worker.estimated_delay_ticks(),
                    "gripper_state_machine": gripper_state,
                    "inference_images": image_paths,
                    "command_sent": execute,
                }
            )
            precise_sleep(
                1.0 / float(args.rtc_control_hz)
                - (time.perf_counter() - loop_started)
            )
    finally:
        if worker is not None:
            worker.stop()
            for completion in worker.drain_completions():
                completion["event"] = (
                    "rtc_replan_completed_after_control_loop"
                    if completion.get("event") == "rtc_replan_completed"
                    else completion.get("event")
                )
                log.write(completion)
        robot.disconnect()


def run_offline_rtc_pair(
    args: argparse.Namespace,
    policy: EgoVersePI05PiperV3V4RTCJointPolicy,
    log: legacy.JsonlLogger,
    device: torch.device,
) -> None:
    """Two-observation sampler A/B that sends no robot command."""
    if args.offline_npz is None:
        raise ValueError("offline RTC mode requires --offline-npz")
    keys = (
        "observation.state",
        "observation.images.435if",
        "observation.images.405",
    )
    with np.load(args.offline_npz, allow_pickle=False) as archive:
        missing = [key for key in keys if key not in archive]
        if missing:
            raise KeyError(f"offline NPZ is missing {missing}")
        arrays = {key: archive[key] for key in keys}
    count = min(len(arrays[key]) for key in keys)
    delay = int(args.rtc_offline_delay_ticks)
    if count <= delay:
        raise ValueError(
            f"offline NPZ needs more than {delay} frames for RTC pair test, got {count}"
        )

    first_batch = legacy.make_policy_batch(
        arrays[keys[0]][0], arrays[keys[1]][0], arrays[keys[2]][0], device
    )
    first_processed = policy.predict_action_chunk(
        first_batch, prev_chunk_left_over=None, inference_delay=0
    ).detach().cpu()
    first_report = copy.deepcopy(policy.last_prediction_report)
    first_original = _action32_from_report(first_report)
    first_physical = _physical_from_report(first_report)

    second_batch = legacy.make_policy_batch(
        arrays[keys[0]][delay],
        arrays[keys[1]][delay],
        arrays[keys[2]][delay],
        device,
    )
    second_processed = policy.predict_action_chunk(
        second_batch,
        prev_chunk_left_over=first_original,
        prev_chunk_decoded_joints=first_physical[:, :6],
        inference_delay=delay,
    ).detach().cpu()
    second_report = copy.deepcopy(policy.last_prediction_report)
    second_original = _action32_from_report(second_report)
    second_physical = _physical_from_report(second_report)
    joint_boundary_rms = float(
        torch.sqrt(
            torch.mean(
                (
                    second_original[delay, PIPER_JOINT_CHANNELS]
                    - first_original[delay, PIPER_JOINT_CHANNELS]
                )
                ** 2
            )
        ).item()
    )
    driver_boundary_rms = float(
        torch.sqrt(
            torch.mean((second_processed[delay, :6] - first_processed[delay, :6]) ** 2)
        ).item()
    )
    physical_boundary_rms_rad = float(
        torch.sqrt(
            torch.mean(
                (second_physical[delay, :6] - first_physical[delay, :6]) ** 2
            )
        ).item()
    )
    log.write(
        {
            "event": "rtc_offline_pair",
            "delay_ticks": delay,
            "first_chunk_report": first_report,
            "second_chunk_report": second_report,
            "boundary_model_joint_rms": joint_boundary_rms,
            "boundary_physical_joint_rms_rad": physical_boundary_rms_rad,
            "boundary_physical_joint_rms_deg": math.degrees(
                physical_boundary_rms_rad
            ),
            "boundary_driver_arm_rms": driver_boundary_rms,
            "command_sent": False,
        }
    )


def parse_args() -> argparse.Namespace:
    rtc_parser = argparse.ArgumentParser(add_help=False)
    rtc_parser.add_argument("--rtc-control-hz", type=float, default=CONTROL_HZ)
    rtc_parser.add_argument("--rtc-execution-horizon", type=int, default=20)
    rtc_parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    rtc_parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=("linear", "exp", "ones", "zeros"),
        default="exp",
    )
    rtc_parser.add_argument(
        "--rtc-guided-channels",
        choices=("joint", "pose-joint", "all"),
        default="pose-joint",
    )
    rtc_parser.add_argument(
        "--rtc-guidance-space",
        choices=("latent", "decoded-joint"),
        default="decoded-joint",
        help=(
            "Apply RTC continuity in decoded physical 6D joint space by "
            "default; latent preserves the earlier compatibility path."
        ),
    )
    rtc_parser.add_argument("--rtc-replan-threshold", type=int, default=20)
    rtc_parser.add_argument("--rtc-initial-delay-ticks", type=int, default=16)
    rtc_parser.add_argument("--rtc-max-delay-ticks", type=int, default=25)
    rtc_parser.add_argument("--rtc-latency-window", type=int, default=16)
    rtc_parser.add_argument(
        "--rtc-max-starvation-hold-frames", type=int, default=60
    )
    rtc_parser.add_argument("--rtc-offline-delay-ticks", type=int, default=6)
    rtc_parser.add_argument("--rtc-debug", action="store_true")
    rtc_parser.add_argument("--rtc-debug-maxlen", type=int, default=100)
    rtc_args, remaining = rtc_parser.parse_known_args()

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = base.parse_args()
    finally:
        sys.argv = old_argv
    for key, value in vars(rtc_args).items():
        setattr(args, key, value)

    if args.temporal_ensemble:
        raise SystemExit("RTC deployment forbids legacy --temporal-ensemble")
    if args.chunk_smoothing:
        raise SystemExit(
            "RTC deployment forbids --chunk-smoothing because post-sampling "
            "smoothing would make the executed queue differ from the RTC prefix"
        )
    if args.query_frequency != MODEL_HORIZON:
        raise SystemExit(
            f"RTC requires --query-frequency {MODEL_HORIZON}; replanning is "
            "controlled by --rtc-replan-threshold"
        )
    if args.rtc_control_hz <= 0:
        raise SystemExit("--rtc-control-hz must be positive")
    if not 1 <= args.rtc_execution_horizon < MODEL_HORIZON:
        raise SystemExit("--rtc-execution-horizon must be in [1,29]")
    if args.rtc_max_guidance_weight <= 0:
        raise SystemExit("--rtc-max-guidance-weight must be positive")
    if not 1 <= args.rtc_replan_threshold < MODEL_HORIZON:
        raise SystemExit("--rtc-replan-threshold must be in [1,29]")
    if not 0 <= args.rtc_initial_delay_ticks < MODEL_HORIZON:
        raise SystemExit("--rtc-initial-delay-ticks must be in [0,29]")
    if not args.rtc_initial_delay_ticks <= args.rtc_max_delay_ticks < MODEL_HORIZON:
        raise SystemExit("RTC delay limits must satisfy 0 <= initial <= max < 30")
    if (
        args.rtc_latency_window <= 0
        or args.rtc_debug_maxlen <= 0
        or args.rtc_max_starvation_hold_frames <= 0
    ):
        raise SystemExit("RTC history lengths must be positive")
    if not 0 <= args.rtc_offline_delay_ticks < MODEL_HORIZON:
        raise SystemExit("--rtc-offline-delay-ticks must be in [0,29]")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.list_checkpoint_profiles:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    actual_sha256 = legacy.sha256_file(args.checkpoint)
    if actual_sha256 != args.checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {args.checkpoint_sha256}, got {actual_sha256}"
        )
    joint_limits, gripper_stroke_mm = legacy.load_calibration(args.calibration_json)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    policy = EgoVersePI05PiperV3V4RTCJointPolicy(
        args, joint_limits, gripper_stroke_mm
    )
    if args.mode == "checkpoint":
        print(
            "PI05_PIPER_V3_V4_RTC_CHECKPOINT_OK "
            f"profile={args.checkpoint_profile} sha256={actual_sha256} "
            "rtc=true robot_constructed=false camera_opened=false can_opened=false"
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = legacy.JsonlLogger(
        args.log_dir / f"pi05_piper_v3_v4_rtc_{args.mode}_{timestamp}.jsonl"
    )
    try:
        log.write(
            {
                "event": "start",
                "mode": args.mode,
                "checkpoint_profile": args.checkpoint_profile,
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": actual_sha256,
                "checkpoint_contract": policy.checkpoint_report,
                "rtc_config": {
                    "control_hz": args.rtc_control_hz,
                    "execution_horizon": args.rtc_execution_horizon,
                    "max_guidance_weight": args.rtc_max_guidance_weight,
                    "prefix_attention_schedule": args.rtc_prefix_attention_schedule,
                    "guided_channels": args.rtc_guided_channels,
                    "guidance_space": args.rtc_guidance_space,
                    "replan_threshold": args.rtc_replan_threshold,
                    "initial_delay_ticks": args.rtc_initial_delay_ticks,
                    "max_delay_ticks": args.rtc_max_delay_ticks,
                    "max_starvation_hold_frames": (
                        args.rtc_max_starvation_hold_frames
                    ),
                },
                "command_sent": False,
            }
        )
        if args.mode == "offline":
            run_offline_rtc_pair(args, policy, log, device)
        else:
            run_live_rtc(
                args,
                policy,
                log,
                device,
                joint_limits,
                gripper_stroke_mm,
                timestamp,
            )
    finally:
        log.close()
        logger.info("wrote RTC deployment audit log to %s", log.path)


if __name__ == "__main__":
    main()
