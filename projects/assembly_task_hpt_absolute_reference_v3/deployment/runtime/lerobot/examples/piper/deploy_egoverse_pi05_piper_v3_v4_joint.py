#!/usr/bin/env python
"""Guarded PIPER deployment for validation-selected pi0.5 PIPER V3/V4.

This additive entry point leaves every earlier HPT/pi0.5 deployment unchanged.
It reconstructs the complete training Lightning ``ModelWrapper`` from the
immutable training config + base pi0.5 weights, then loads the compact LoRA
checkpoint selected only on validation.  The robot command is produced by the
same Gemma action expert and joint decoder used in training:

* OpenPI channels 0..8: shared camera-frame pose (training/evaluation only);
* OpenPI channels 9..14: V3 linear or V4 bounded native joint residuals;
* action-expert hidden state: native 30-step binary gripper logits.

There is no 30<->100 interpolation, IK, or deployment-only joint decoder.
``checkpoint`` mode constructs no robot and opens no camera or CAN interface.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# Load the private OpenPI-compatible Transformers overlay before importing the
# legacy LeRobot deployment module.  The latter imports Transformers
# transitively; inserting this path afterwards is too late because Python will
# keep using the already-loaded unpatched package.
_BUNDLE_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP_RUNTIME = _BUNDLE_ROOT / "pi05_runtime/python_overlay"
if _BOOTSTRAP_RUNTIME.is_dir() and str(_BOOTSTRAP_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_RUNTIME))

_TRAINING_ROOT = Path(
    os.environ.get(
        "EGOVERSE_TRAINING_ROOT",
        str(_BUNDLE_ROOT / "training_code/EgoVerse"),
    )
).expanduser().resolve()
_OPENPI_SRC = _TRAINING_ROOT / "external" / "openpi" / "src"
for _path in (_TRAINING_ROOT, _OPENPI_SRC):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch

import deploy_egoverse_hpt_joint as legacy
import deploy_egoverse_pi05_joint as pi05_legacy
from lerobot.policies.egoverse_hpt.modeling_egoverse_hpt_joint import (
    adjacent_joint_rms,
    model_indices_for_control_ticks,
    prepare_rgb_chw_image,
    smooth_joint_action_chunk,
)

logger = logging.getLogger("deploy_egoverse_pi05_piper_v3_v4_joint")

PIPER_V3_V4_ROOT = Path("/mnt/wd10/kerun/pi05_piper_v3_v4")
PIPER_V3_V4_CHECKPOINT_PROFILES: dict[str, dict[str, object]] = {
    "pi05-piper-v3-15k": {
        "checkpoint": str(PIPER_V3_V4_ROOT / "checkpoints/v3_selected_step15000_trainable.ckpt"),
        "sha256": "538c8582dd6b8e7cff281022c4047acc120265bc26b71505250b7f6d5ae102d2",
        "model_action_horizon": 30,
        "description": (
            "EgoVerse pi0.5 PIPER-only Expert-Action V3 LoRA, "
            "validation-selected step 15000"
        ),
        "model_config": str(PIPER_V3_V4_ROOT / "configs/v3_config.yaml"),
        "model_config_sha256": "5092bf6aaae1ab51cfdf028e6bfe11ee321375ec549ed0f178a8526011e57317",
        "norm_stats": str(PIPER_V3_V4_ROOT / "norm_stats/norm_stats.json"),
        "norm_stats_sha256": "c20c3dcaaef0233c033215f373b9181b0a5c5ae5d0fc127d4f6320b2dd00e92f",
        "pytorch_weights": "/mnt/wd10/kerun/pi05_mixed_lora_v1/assets/pi05_base_pytorch",
        "expected_model_target": "egomimic.algo.pi_egoverse_expert_action_v3_lora.PIEgoVerseExpertActionV3LoRA",
        "expected_contract": "egoverse_pi05_expert_action_v3",
        "expected_contract_version": 3,
        "expected_step": 15000,
        "expected_parameter_count": 31984161,
    },
    "pi05-piper-v4-10k": {
        "checkpoint": str(PIPER_V3_V4_ROOT / "checkpoints/v4_selected_step10000_trainable.ckpt"),
        "sha256": "b09d2c7cb7bb899908e03b00e008bf90815c2c9cd0399b0b9a46a511dea004e8",
        "model_action_horizon": 30,
        "description": (
            "EgoVerse pi0.5 PIPER-only Expert-Action V4 bounded LoRA, "
            "validation-selected step 10000"
        ),
        "model_config": str(PIPER_V3_V4_ROOT / "configs/v4_config.yaml"),
        "model_config_sha256": "984d59e45dfe9a0f38ca035c8357e43cdd9ef6321053613edae6fa94573f3e4b",
        "norm_stats": str(PIPER_V3_V4_ROOT / "norm_stats/norm_stats.json"),
        "norm_stats_sha256": "c20c3dcaaef0233c033215f373b9181b0a5c5ae5d0fc127d4f6320b2dd00e92f",
        "pytorch_weights": "/mnt/wd10/kerun/pi05_mixed_lora_v1/assets/pi05_base_pytorch",
        "expected_model_target": "egomimic.algo.pi_egoverse_expert_action_v4_bounded_lora.PIEgoVerseExpertActionV4BoundedLoRA",
        "expected_contract": "egoverse_pi05_expert_action_v4_bounded",
        "expected_contract_version": 4,
        "expected_step": 10000,
        "expected_parameter_count": 31984161,
    },
}


def _profile_for_args(args: argparse.Namespace) -> dict[str, object]:
    if args.checkpoint_profile not in PIPER_V3_V4_CHECKPOINT_PROFILES:
        raise RuntimeError(
            "this deployment entry point requires an audited V3/V4 "
            "--checkpoint-profile; explicit custom checkpoints are disabled"
        )
    return PIPER_V3_V4_CHECKPOINT_PROFILES[args.checkpoint_profile]


def _norm_state_from_json(path: Path) -> dict:
    """Convert the cached training normalization JSON into ModelWrapper state."""
    payload = json.loads(path.read_text())
    raw_stats = payload.get("stats")
    if not isinstance(raw_stats, dict) or not raw_stats:
        raise RuntimeError(f"normalization file has no stats mapping: {path}")
    stats: dict[int, dict] = {}
    key_types: dict[int, dict[str, str]] = {}
    zarr_keys: dict[int, dict[str, str]] = {}
    shapes: dict[int, dict[str, tuple[int, ...]]] = {}
    for raw_embodiment, per_key in raw_stats.items():
        embodiment = int(raw_embodiment)
        stats[embodiment] = per_key
        key_types[embodiment] = {}
        zarr_keys[embodiment] = {}
        shapes[embodiment] = {}
        for key, values in per_key.items():
            if key.startswith("observations.state"):
                key_type = "proprio_keys"
            elif key.startswith("actions"):
                key_type = "action_keys"
            else:
                continue
            key_types[embodiment][key] = key_type
            zarr_keys[embodiment][key] = key
            if "mean" not in values:
                raise RuntimeError(f"normalization entry {key!r} lacks mean")
            shapes[embodiment][key] = tuple(np.asarray(values["mean"]).shape)
    candidates = [
        embodiment
        for embodiment, per_key in stats.items()
        if "observations.state.joints" in per_key and "actions_joints" in per_key
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "normalization must contain exactly one PIPER embodiment; "
            f"found {candidates}"
        )
    return {
        "norm_mode": "quantile",
        "embodiments": sorted(stats),
        "key_types": key_types,
        "zarr_keys": zarr_keys,
        "shapes": shapes,
        "norm_stats": stats,
    }


def load_compact_model_wrapper(args: argparse.Namespace):
    """Rebuild the frozen pi0.5 model and apply the selected compact state."""
    profile = _profile_for_args(args)
    model_config = Path(str(profile["model_config"]))
    norm_stats = Path(str(profile["norm_stats"]))
    pytorch_weights = Path(str(profile["pytorch_weights"]))
    for path in (model_config, norm_stats):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not pytorch_weights.is_dir():
        raise FileNotFoundError(pytorch_weights)
    if legacy.sha256_file(model_config) != profile["model_config_sha256"]:
        raise RuntimeError("training model config SHA-256 mismatch")
    if legacy.sha256_file(norm_stats) != profile["norm_stats_sha256"]:
        raise RuntimeError("training normalization SHA-256 mismatch")
    if not _TRAINING_ROOT.is_dir() or not _OPENPI_SRC.is_dir():
        raise FileNotFoundError(
            "isolated V3/V4 training checkout is unavailable: "
            f"{_TRAINING_ROOT}"
        )

    os.environ.setdefault(
        "OPENPI_PALIGEMMA_TOKENIZER_MODEL", str(args.pi05_tokenizer_model)
    )
    from omegaconf import OmegaConf

    from egomimic.pl_utils.pl_model import ModelWrapper

    full_cfg = OmegaConf.load(model_config)
    actual_target = str(full_cfg.model.robomimic_model._target_)
    if actual_target != profile["expected_model_target"]:
        raise RuntimeError(
            "training model target mismatch: "
            f"expected={profile['expected_model_target']} actual={actual_target}"
        )
    model_cfg = copy.deepcopy(full_cfg.model)
    model_cfg.robomimic_model.config.pytorch_weight_path = str(pytorch_weights)
    if "norm_stats" in model_cfg.robomimic_model:
        model_cfg.robomimic_model.norm_stats = None
    config_tree = OmegaConf.create({"model": model_cfg})
    wrapper = ModelWrapper(
        config_tree=config_tree,
        norm_stats_state=_norm_state_from_json(norm_stats),
        optimizer=None,
        scheduler=None,
        scheduler_interval="step",
        scheduler_frequency=1,
        evaluator=None,
        enable_grad_norm=False,
    )

    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if payload.get("checkpoint_format") != "egomimic_trainable_state_v1":
        raise RuntimeError("checkpoint is not an audited compact trainable state")
    if int(payload.get("global_step", -1)) != int(profile["expected_step"]):
        raise RuntimeError("compact checkpoint step does not match its profile")
    if int(payload.get("trainable_parameter_count", -1)) != int(
        profile["expected_parameter_count"]
    ):
        raise RuntimeError("compact checkpoint parameter count mismatch")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError("compact checkpoint has no state_dict")
    declared = set(payload.get("trainable_state_keys", ()))
    if declared != set(state_dict):
        raise RuntimeError("compact checkpoint key manifest is inconsistent")
    model_state = wrapper.state_dict()
    missing_from_model = sorted(set(state_dict).difference(model_state))
    shape_mismatches = {
        name: (tuple(value.shape), tuple(model_state[name].shape))
        for name, value in state_dict.items()
        if name in model_state and tuple(value.shape) != tuple(model_state[name].shape)
    }
    if missing_from_model or shape_mismatches:
        raise RuntimeError(
            "compact checkpoint is incompatible with reconstructed model: "
            f"unknown={missing_from_model[:10]} shapes={shape_mismatches}"
        )
    nonfinite = [
        name for name, value in state_dict.items() if not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise RuntimeError(f"compact checkpoint contains NaN/Inf: {nonfinite[:10]}")
    trainable_names = {
        name for name, parameter in wrapper.named_parameters() if parameter.requires_grad
    }
    if trainable_names != set(state_dict):
        raise RuntimeError(
            "compact state is not the exact reconstructed trainable parameter set: "
            f"missing={sorted(trainable_names - set(state_dict))[:10]} "
            f"extra={sorted(set(state_dict) - trainable_names)[:10]}"
        )
    incompatible = wrapper.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected compact checkpoint keys: {incompatible.unexpected_keys}"
        )
    wrapper = wrapper.to(args.device).eval()
    wrapper.model.device = torch.device(args.device)
    wrapper.compact_deployment_report = {
        "checkpoint_format": payload["checkpoint_format"],
        "global_step": int(payload["global_step"]),
        "loaded_state_keys": len(state_dict),
        "loaded_parameter_count": int(payload["trainable_parameter_count"]),
        "frozen_state_keys_from_base": len(incompatible.missing_keys),
        "model_config": str(model_config),
        "norm_stats": str(norm_stats),
        "pytorch_weights": str(pytorch_weights),
        "model_target": actual_target,
    }
    del payload, state_dict, model_state
    return wrapper


def validate_v3_v4_deploy_result(
    result: dict[str, torch.Tensor | list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and unpack the exact V3 joint-only deployment contract."""
    required = {"actions", "action32", "gripper_logits", "normalized_state"}
    missing = required.difference(result)
    if missing:
        raise RuntimeError(f"V3/V4 deployment result lacks {sorted(missing)}")
    actions = torch.as_tensor(result["actions"]).detach().float().cpu().numpy()
    action32 = torch.as_tensor(result["action32"]).detach().float().cpu().numpy()
    logits = (
        torch.as_tensor(result["gripper_logits"]).detach().float().cpu().numpy()
    )
    normalized_state = (
        torch.as_tensor(result["normalized_state"])
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    expected = {
        "actions": ((1, 30, 7), actions.shape),
        "action32": ((1, 30, 32), action32.shape),
        "gripper_logits": ((1, 30), logits.shape),
        "normalized_state": ((1, 7), normalized_state.shape),
    }
    malformed = {
        name: {"expected": wanted, "actual": actual}
        for name, (wanted, actual) in expected.items()
        if tuple(actual) != wanted
    }
    if malformed:
        raise RuntimeError(f"malformed V3/V4 deployment result: {malformed}")
    if not all(
        np.isfinite(value).all()
        for value in (actions, action32, logits, normalized_state)
    ):
        raise RuntimeError("V3/V4 deployment result contains NaN/Inf")
    gripper = actions[..., 6]
    if not np.isin(gripper, (0.0, 1.0)).all():
        raise RuntimeError("V3/V4 deployment gripper output is not binary")
    return actions[0], action32[0], logits[0], normalized_state[0]


class EgoVersePI05PiperV3V4JointPolicy(
    pi05_legacy.EgoVersePI05JointPolicy
):
    """Reconstructed V3/V4 Gemma expert behind the audited joint queue."""

    name = "egoverse_pi05_piper_v3_v4_joint"

    def __init__(
        self,
        args: argparse.Namespace,
        joint_limits_deg: list[tuple[float, float]],
        gripper_stroke_mm: float,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            device=str(args.device),
            chunk_size=30,
            action_horizon_frames=30,
            query_frequency=int(args.query_frequency),
            num_inference_steps=int(args.inference_steps),
            inference_seed=int(args.seed),
            gripper_threshold=float(args.gripper_threshold),
            joint_limit_margin_rad=float(np.radians(args.joint_limit_margin_deg)),
            max_joint_command_step=float(args.max_joint_command_step),
            max_joint_command_accel=float(args.max_joint_command_accel),
            temporal_ensemble=bool(args.temporal_ensemble),
            temporal_replan_interval=int(args.temporal_replan_interval),
            temporal_ensemble_decay=float(args.temporal_ensemble_decay),
            chunk_smoothing=bool(args.chunk_smoothing),
            chunk_smoothing_window=int(args.chunk_smoothing_window),
            chunk_smoothing_sigma=float(args.chunk_smoothing_sigma),
            gripper_stroke_mm=float(gripper_stroke_mm),
            front_camera_key="435if",
            wrist_camera_key="405",
            side_camera_key="side",
            expected_image_height=480,
            expected_image_width=640,
            resize_mismatched_images=False,
        )
        self._joint_low_rad = np.radians(
            np.asarray([pair[0] for pair in joint_limits_deg], dtype=np.float64)
        )
        self._joint_high_rad = np.radians(
            np.asarray([pair[1] for pair in joint_limits_deg], dtype=np.float64)
        )
        self._action_queue: deque[torch.Tensor] = deque()
        self._chunk_sequence_id = 0
        self._active_chunk_size = 0
        self._active_chunk_consumed = 0
        self._last_planned_velocity_driver = np.zeros(6, dtype=np.float64)
        self._pending_planned_velocity_driver: np.ndarray | None = None
        self._temporal_plans: list[tuple[int, int, np.ndarray]] = []
        self._policy_tick = 0
        self._last_temporal_replan_tick: int | None = None
        self.last_ensemble_report: dict | None = None
        self.last_prediction_report: dict | None = None

        profile = _profile_for_args(args)
        self._model_wrapper = load_compact_model_wrapper(args)
        self._algo = self._model_wrapper.model
        if not callable(getattr(self._algo, "forward_joint_deploy", None)):
            raise RuntimeError(
                "checkpoint model does not implement forward_joint_deploy"
            )
        if int(self._algo.config.model.action_dim) != 32:
            raise RuntimeError("checkpoint action_dim is not 32")
        if int(self._algo.config.model.action_horizon) != 30:
            raise RuntimeError("checkpoint action_horizon is not 30")
        auxiliary = (
            self._algo.nets["auxiliary"]
            if "auxiliary" in self._algo.nets
            else None
        )
        if auxiliary is None:
            raise RuntimeError("checkpoint lacks nets.auxiliary")
        if getattr(auxiliary, "contract_name", None) != profile["expected_contract"]:
            raise RuntimeError(
                "checkpoint auxiliary contract mismatch: "
                f"expected={profile['expected_contract']} "
                f"actual={getattr(auxiliary, 'contract_name', None)}"
            )
        if int(getattr(auxiliary, "contract_version", -1)) != int(
            profile["expected_contract_version"]
        ):
            raise RuntimeError("checkpoint auxiliary contract version mismatch")

        self._prompt = str(self._algo.default_prompt)
        if str(args.task_prompt) != self._prompt:
            raise RuntimeError(
                "deployment task prompt differs from checkpoint training prompt: "
                f"cli={args.task_prompt!r}, checkpoint={self._prompt!r}"
            )
        self._generator = torch.Generator(device=str(args.device))
        self._generator.manual_seed(int(args.seed))
        # Flow Matching normally starts each sample from Gaussian noise.  A
        # freshly sampled tensor on every replan made a static observation
        # produce a different robot trajectory each time.  Reuse one tensor
        # for the complete rollout so replanning responds only to observation
        # changes, while --seed still selects a reproducible rollout mode.
        self._fixed_rollout_noise = torch.randn(
            (1, 30, 32),
            generator=self._generator,
            device=str(args.device),
            dtype=torch.float32,
        )
        self._fixed_rollout_noise_fingerprint = {
            "mean": float(self._fixed_rollout_noise.mean().item()),
            "std": float(self._fixed_rollout_noise.std().item()),
            "l2_norm": float(self._fixed_rollout_noise.norm().item()),
        }
        self.checkpoint_report = {
            "architecture": str(profile["expected_contract"]),
            "checkpoint_profile": args.checkpoint_profile or "custom",
            "loader": "base_pi0.5_plus_compact_LoRA_ModelWrapper",
            "deployment_forward": "ModelWrapper.model.forward_joint_deploy",
            "shared_pose_channels": "Gemma_action_expert_0_8",
            "robot_joint_channels": "Gemma_action_expert_9_14",
            "gripper_head": "action_expert_hidden_to_binary_gripper",
            "action_space": (
                "six_absolute_PIPER_joint_angles_radians_plus_binary_gripper"
            ),
            "model_action_dim": 32,
            "model_action_horizon": 30,
            "native_control_horizon": 30,
            "observation_history": 1,
            "normalization": "checkpoint_ModelWrapper_norm_stats",
            "prompt": self._prompt,
            "command_generation": (
                "native_30_step_Gemma_expert_direct_joint_no_interpolation_no_ik"
            ),
            "flow_noise_mode": "fixed_per_rollout",
            "flow_noise_seed": int(args.seed),
            "flow_noise_fingerprint": self._fixed_rollout_noise_fingerprint,
            "compact_checkpoint": self._model_wrapper.compact_deployment_report,
            "contract_name": auxiliary.contract_name,
            "contract_version": auxiliary.contract_version,
            "joint_residual_scale_deg": (
                auxiliary.joint_residual_scale_deg_vector.detach().cpu().tolist()
                if hasattr(auxiliary, "joint_residual_scale_deg_vector")
                else float(np.degrees(auxiliary.joint_residual_scale_rad))
            ),
        }

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        del kwargs
        device = torch.device(self.config.device)
        raw_state_value = batch["observation.state"]
        raw_state = (
            raw_state_value.detach().cpu().numpy()
            if isinstance(raw_state_value, torch.Tensor)
            else np.asarray(raw_state_value)
        )
        raw_state = raw_state.reshape(-1)[:7].astype(np.float64)
        if raw_state.shape != (7,) or not np.isfinite(raw_state).all():
            raise RuntimeError(f"invalid raw PIPER state: {raw_state}")
        obs_joint = (raw_state[:6] * 1e-3 * np.pi / 180.0).astype(np.float32)
        obs_gripper = np.clip(
            raw_state[6:7] * 1e-3 / self.config.gripper_stroke_mm,
            0.0,
            1.0,
        ).astype(np.float32)
        proprio_physical = np.concatenate((obs_joint, obs_gripper)).astype(
            np.float32
        )
        current_raw = torch.from_numpy(proprio_physical).unsqueeze(0).to(device)

        front_key = f"observation.images.{self.config.front_camera_key}"
        wrist_key = f"observation.images.{self.config.wrist_camera_key}"
        side_key = f"observation.images.{self.config.side_camera_key}"
        front = prepare_rgb_chw_image(
            batch[front_key],
            expected_height=self.config.expected_image_height,
            expected_width=self.config.expected_image_width,
            resize_mismatch=self.config.resize_mismatched_images,
            name=front_key,
        )
        wrist = prepare_rgb_chw_image(
            batch[wrist_key],
            expected_height=self.config.expected_image_height,
            expected_width=self.config.expected_image_width,
            resize_mismatch=self.config.resize_mismatched_images,
            name=wrist_key,
        )
        side = prepare_rgb_chw_image(
            batch[side_key],
            expected_height=self.config.expected_image_height,
            expected_width=self.config.expected_image_width,
            resize_mismatch=self.config.resize_mismatched_images,
            name=side_key,
        )
        noise = self._fixed_rollout_noise.clone()
        started = time.perf_counter()
        result = self._algo.forward_joint_deploy(
            front_image=front.unsqueeze(0).to(device),
            wrist_image=wrist.unsqueeze(0).to(device),
            side_image=side.unsqueeze(0).to(device),
            current_raw=current_raw,
            prompt=self._prompt,
            noise=noise,
            num_steps=self.config.num_inference_steps,
        )
        inference_ms = (time.perf_counter() - started) * 1e3
        pred, action32, gripper_logits, normalized_state = (
            validate_v3_v4_deploy_result(result)
        )
        gripper_probability = 1.0 / (1.0 + np.exp(-gripper_logits))

        pred_for_control = (
            smooth_joint_action_chunk(
                pred,
                self.config.chunk_smoothing_window,
                self.config.chunk_smoothing_sigma,
            )
            if self.config.chunk_smoothing
            else pred.copy()
        )
        raw_adjacent_rms = adjacent_joint_rms(pred)
        smoothed_adjacent_rms = adjacent_joint_rms(pred_for_control)
        selected_ticks = (
            self.config.action_horizon_frames
            if self.config.temporal_ensemble
            else self.config.query_frequency
        )
        query_indices = model_indices_for_control_ticks(30, 30, selected_ticks)
        full_control_indices = model_indices_for_control_ticks(30, 30, 30)
        full_gripper_raw = pred[full_control_indices, 6].copy()
        full_gripper_pct = np.where(
            full_gripper_raw >= self.config.gripper_threshold, 100.0, 0.0
        ).astype(np.float32)
        selected_unsmoothed = pred[query_indices].copy()
        selected = pred_for_control[query_indices].copy()

        margin = self.config.joint_limit_margin_rad
        low = self._joint_low_rad + margin
        high = self._joint_high_rad - margin
        if np.any(high <= low):
            raise RuntimeError("joint margin collapses a calibrated range")
        before_clip = selected[:, :6].copy()
        selected[:, :6] = np.clip(selected[:, :6], low, high)
        clipped_mask = np.abs(selected[:, :6] - before_clip) > 1e-9
        selected[:, 6] = (
            selected[:, 6] >= self.config.gripper_threshold
        ).astype(np.float32)

        raw_driver_actions = np.empty_like(selected, dtype=np.float32)
        raw_driver_actions[:, :6] = np.stack(
            [self._joint_rad_to_driver(row) for row in selected[:, :6]], axis=0
        )
        raw_driver_actions[:, 6] = selected[:, 6] * 100.0
        current_driver = self._joint_rad_to_driver(obs_joint)
        preview = self._preview_filtered_chunk(raw_driver_actions, current_driver)

        self._chunk_sequence_id += 1
        self.last_prediction_report = {
            "policy_space": "absolute_joint_radians_plus_binary_gripper",
            "checkpoint_contract": self.checkpoint_report,
            "output_head": "full_ModelWrapper_Gemma_action_expert_channels_9_14",
            "joint_only_forward": True,
            "prompt": self._prompt,
            "proprio_physical": proprio_physical.tolist(),
            "proprio_model_input": normalized_state.tolist(),
            "model_chunk_shape": [30, 7],
            "model_action32_shape": [30, 32],
            "native_control_horizon": 30,
            "native_gripper_logits": gripper_logits.tolist(),
            "native_gripper_open_probability": gripper_probability.tolist(),
            "query_frequency": self.config.query_frequency,
            "selected_control_ticks": selected_ticks,
            "temporal_ensemble": self.config.temporal_ensemble,
            "temporal_replan_interval": self.config.temporal_replan_interval,
            "temporal_ensemble_decay": self.config.temporal_ensemble_decay,
            "chunk_smoothing": self.config.chunk_smoothing,
            "chunk_smoothing_window": self.config.chunk_smoothing_window,
            "chunk_smoothing_sigma": self.config.chunk_smoothing_sigma,
            "chunk_raw_adjacent_joint_rms_rad_mean": raw_adjacent_rms[0],
            "chunk_raw_adjacent_joint_rms_rad_max": raw_adjacent_rms[1],
            "chunk_smoothed_adjacent_joint_rms_rad_mean": smoothed_adjacent_rms[0],
            "chunk_smoothed_adjacent_joint_rms_rad_max": smoothed_adjacent_rms[1],
            "query_indices": query_indices,
            "full_control_indices": full_control_indices,
            "full_control_gripper_raw": full_gripper_raw.tolist(),
            "full_control_gripper_commands_pct": full_gripper_pct.tolist(),
            "num_inference_steps": self.config.num_inference_steps,
            "inference_seed": self.config.inference_seed,
            "inference_noise_mode": "fixed_per_rollout",
            "inference_noise_fingerprint": self._fixed_rollout_noise_fingerprint,
            "model_inference_ms": inference_ms,
            "joint_limit_clipped_values": int(clipped_mask.sum()),
            "joint_limit_clipped_steps": int(clipped_mask.any(axis=1).sum()),
            "full_model_chunk_columns": [
                "joint_1_rad",
                "joint_2_rad",
                "joint_3_rad",
                "joint_4_rad",
                "joint_5_rad",
                "joint_6_rad",
                "gripper_open",
            ],
            "full_model_chunk_physical": pred.tolist(),
            "full_model_action32": action32.tolist(),
            "full_model_chunk_smoothed": pred_for_control.tolist(),
            "selected_physical_actions_unsmoothed": selected_unsmoothed.tolist(),
            "selected_physical_actions": selected.tolist(),
            "selected_raw_driver_actions": raw_driver_actions.tolist(),
            "selected_driver_actions": preview.tolist(),
            "trajectory_filter": {
                "max_joint_command_step": self.config.max_joint_command_step,
                "max_joint_command_accel": self.config.max_joint_command_accel,
                "current_driver_anchor": current_driver.tolist(),
            },
        }
        return torch.from_numpy(raw_driver_actions)


def parse_args() -> argparse.Namespace:
    old_profiles = pi05_legacy.PI05_CHECKPOINT_PROFILES
    try:
        pi05_legacy.PI05_CHECKPOINT_PROFILES = (
            PIPER_V3_V4_CHECKPOINT_PROFILES
        )
        args = pi05_legacy.parse_args()
    finally:
        pi05_legacy.PI05_CHECKPOINT_PROFILES = old_profiles
    if not args.list_checkpoint_profiles and args.chunk_smoothing_window > 30:
        raise SystemExit(
            "--chunk-smoothing-window must not exceed the native 30-step horizon"
        )
    if not args.list_checkpoint_profiles and args.checkpoint_profile is None:
        raise SystemExit(
            "this entry point only accepts an audited --checkpoint-profile"
        )
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    if args.list_checkpoint_profiles:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    actual_sha256 = legacy.sha256_file(args.checkpoint)
    if actual_sha256 != args.checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {args.checkpoint_sha256}, "
            f"got {actual_sha256}"
        )
    joint_limits, gripper_stroke_mm = legacy.load_calibration(
        args.calibration_json
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    policy = EgoVersePI05PiperV3V4JointPolicy(
        args, joint_limits, gripper_stroke_mm
    )
    if args.mode == "checkpoint":
        logger.info(
            "PIPER V3/V4 checkpoint loaded without robot/camera/CAN: "
            "sha256=%s profile=%s contract=%s",
            actual_sha256,
            args.checkpoint_profile or "custom",
            policy.checkpoint_report,
        )
        print(
            "PI05_PIPER_V3_V4_CHECKPOINT_OK "
            f"profile={args.checkpoint_profile or 'custom'} "
            f"sha256={actual_sha256} robot_constructed=false "
            "camera_opened=false can_opened=false"
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = legacy.JsonlLogger(
        args.log_dir / f"pi05_piper_v3_v4_{args.mode}_{timestamp}.jsonl"
    )
    try:
        log.write(
            {
                "event": "start",
                "mode": args.mode,
                "checkpoint_profile": args.checkpoint_profile or "custom",
                "checkpoint_description": args.checkpoint_description,
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": actual_sha256,
                "checkpoint_contract": policy.checkpoint_report,
                "query_frequency": args.query_frequency,
                "inference_steps": args.inference_steps,
                "seed": args.seed,
                "command_sent": False,
            }
        )
        if args.mode == "offline":
            legacy.run_offline(args, policy, log, device)
        else:
            legacy.run_live(
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
        logger.info("wrote PIPER V3/V4 audit log to %s", log.path)


if __name__ == "__main__":
    main()
