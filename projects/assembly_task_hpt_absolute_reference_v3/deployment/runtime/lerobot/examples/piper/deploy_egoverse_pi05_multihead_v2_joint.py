#!/usr/bin/env python
"""Guarded PIPER deployment for EgoVerse pi0.5 MultiHead V2.

This is an additive deployment entry point.  It leaves the earlier pi0.5
Flow-head deployment untouched and loads the formal MultiHead V2 contract:

* ``nets.policy.*``: shared SigLIP/PaliGemma visual-language prefix;
* ``nets.joint_head.*``: deterministic 7-D PIPER joint/gripper output.

The shared 9-D Cartesian flow output is never interpreted as a robot joint
command.  No FK or IK is used at deployment.  Robot/camera/command safety and
the independent gripper state machine remain provided by
``deploy_egoverse_hpt_joint.py`` through the already audited pi0.5 wrapper.

``checkpoint`` mode constructs no robot and opens no camera or CAN interface.
Use it for static verification before any live mode is considered.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

_BOOTSTRAP_RUNTIME = Path("/mnt/wd10/kerun/pi05_runtime")
if _BOOTSTRAP_RUNTIME.is_dir() and str(_BOOTSTRAP_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_RUNTIME))

# The formal deployment loads the training Lightning ModelWrapper itself.
# The environment variable keeps this portable for offline/local validation;
# the robot command sets the immutable kerun-owned checkout explicitly.
import os

_TRAINING_ROOT = Path(
    os.environ.get(
        "EGOVERSE_TRAINING_ROOT",
        "/mnt/wd10/kerun/egoverse_piper/EgoVerse",
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

logger = logging.getLogger("deploy_egoverse_pi05_multihead_v2_joint")

MULTIHEAD_V2_CHECKPOINT_PROFILES: dict[str, dict[str, str]] = {
    "pi05-multihead-v2-mixed": {
        "checkpoint": (
            "/mnt/wd10/kerun/checkpoints/"
            "pi05_multihead_v2_mixed_best_step30000.ckpt"
        ),
        "sha256": "90eeec553150f35aff303dac20fa27e1023fc6be589792b01bfed93351eff6f0",
        "description": (
            "EgoVerse pi0.5 MultiHead V2 mixed PIPER+human, "
            "held-out PIPER validation selected step 30000"
        ),
    },
}

MULTIHEAD_V2_CONTROL_MODE = (
    "shared camera-frame hand pose plus PIPER residual joint control"
)


class EgoVersePI05MultiHeadV2JointPolicy(pi05_legacy.EgoVersePI05JointPolicy):
    """MultiHead V2 prefix + joint-head adapter for the audited joint queue."""

    name = "egoverse_pi05_multihead_v2_joint"

    def __init__(
        self,
        args: argparse.Namespace,
        joint_limits_deg: list[tuple[float, float]],
        gripper_stroke_mm: float,
    ) -> None:
        # Initialize the audited direct-joint queue/filter state without
        # constructing the old hand-written pi0.5 runtime adapter.
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            device=str(args.device),
            chunk_size=100,
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

        self._model_wrapper = self._load_full_model_wrapper(args)
        self._algo = self._model_wrapper.model
        self._prompt = str(self._algo.default_prompt)
        if str(args.task_prompt) != self._prompt:
            raise RuntimeError(
                "deployment task prompt differs from the checkpoint training prompt: "
                f"cli={args.task_prompt!r}, checkpoint={self._prompt!r}"
            )
        self.checkpoint_report = {
            "architecture": "EgoVerse_pi0.5_MultiHead_V2",
            "checkpoint_profile": args.checkpoint_profile or "custom",
            "loader": "complete_Lightning_ModelWrapper_strict_assign",
            "deployment_forward": "ModelWrapper.model.forward_joint_deploy",
            "shared_head": "nets.policy.*_visual_language_prefix_only",
            "robot_output_head": "nets.joint_head.*_from_full_wrapper",
            "action_space": (
                "six_absolute_PIPER_joint_angles_radians_plus_binary_gripper"
            ),
            "action_horizon": 100,
            "native_control_horizon": 30,
            "observation_history": 1,
            "normalization": "checkpoint_ModelWrapper_norm_stats",
            "prompt": self._prompt,
            "control_mode": MULTIHEAD_V2_CONTROL_MODE,
            "command_generation": "direct_absolute_joint_radians_no_flow_no_ik_no_fk",
            "model_wrapper_state_dict_strict": True,
            "shared_pose_flow_sampled_for_robot": False,
            "contract_name": self._algo.nets["joint_head"].contract_name,
            "contract_version": self._algo.nets["joint_head"].contract_version,
        }

    @staticmethod
    def _load_full_model_wrapper(args: argparse.Namespace):
        if not _TRAINING_ROOT.is_dir() or not _OPENPI_SRC.is_dir():
            raise FileNotFoundError(
                "full training checkout is unavailable; set EGOVERSE_TRAINING_ROOT "
                f"(resolved root={_TRAINING_ROOT})"
            )
        os.environ.setdefault(
            "OPENPI_PALIGEMMA_TOKENIZER_MODEL", str(args.pi05_tokenizer_model)
        )
        from omegaconf import OmegaConf

        from egomimic.pl_utils.pl_model import ModelWrapper

        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        state_dict = checkpoint.get("state_dict")
        hparams = checkpoint.get("hyper_parameters", {})
        if not isinstance(state_dict, dict):
            raise RuntimeError("MultiHead V2 checkpoint has no Lightning state_dict")
        if hparams.get("config_tree") is None or hparams.get("norm_stats_state") is None:
            raise RuntimeError(
                "checkpoint lacks config_tree or norm_stats_state required by ModelWrapper"
            )
        config_tree = copy.deepcopy(hparams["config_tree"])
        config_tree = OmegaConf.create(OmegaConf.to_container(config_tree, resolve=False))
        config_tree.model.robomimic_model.config.pytorch_weight_path = None

        with torch.device("meta"):
            wrapper = ModelWrapper(
                config_tree=config_tree,
                norm_stats_state=hparams["norm_stats_state"],
                optimizer=None,
                scheduler=None,
                scheduler_interval=hparams.get("scheduler_interval", "step"),
                scheduler_frequency=hparams.get("scheduler_frequency", 1),
                evaluator=None,
                enable_grad_norm=False,
            )
        incompatible = wrapper.load_state_dict(state_dict, strict=True, assign=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"full ModelWrapper state mismatch: {incompatible}")
        meta_parameters = [
            name for name, parameter in wrapper.named_parameters() if parameter.is_meta
        ]
        if meta_parameters:
            raise RuntimeError(
                "checkpoint did not materialize ModelWrapper parameters: "
                f"{meta_parameters[:10]}"
            )
        for module in wrapper.modules():
            for buffer_name, buffer in tuple(module._buffers.items()):
                if buffer is not None and buffer.is_meta:
                    module._buffers[buffer_name] = torch.zeros_like(buffer, device="cpu")
        wrapper = wrapper.to(args.device).eval()
        wrapper.model.device = torch.device(args.device)
        del checkpoint, state_dict
        return wrapper

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
            raw_state[6:7] * 1e-3 / self.config.gripper_stroke_mm, 0.0, 1.0
        ).astype(np.float32)
        proprio_physical = np.concatenate((obs_joint, obs_gripper)).astype(np.float32)
        state_tensor = torch.from_numpy(proprio_physical).unsqueeze(0).to(device)
        front_key = f"observation.images.{self.config.front_camera_key}"
        wrist_key = f"observation.images.{self.config.wrist_camera_key}"
        front = prepare_rgb_chw_image(
            batch[front_key],
            expected_height=480,
            expected_width=640,
            resize_mismatch=False,
            name=front_key,
        )
        wrist = prepare_rgb_chw_image(
            batch[wrist_key],
            expected_height=480,
            expected_width=640,
            resize_mismatch=False,
            name=wrist_key,
        )
        started = time.perf_counter()
        result = self._algo.forward_joint_deploy(
            front_image=front.unsqueeze(0).to(device),
            wrist_image=wrist.unsqueeze(0).to(device),
            current_raw=state_tensor,
        )
        inference_ms = (time.perf_counter() - started) * 1e3
        predicted = result["actions"]
        gripper_logits = result["gripper_logits"]
        normalized_state = result["normalized_state"]
        prefix_features = result["prefix_features"]
        prefix_pad_mask = result["prefix_pad_mask"]
        pred = predicted[0].float().cpu().numpy()
        native_gripper_logits = gripper_logits[0].float().cpu().numpy()
        native_gripper_open_probability = 1.0 / (
            1.0 + np.exp(-native_gripper_logits)
        )
        if pred.shape != (100, 7) or not np.isfinite(pred).all():
            raise RuntimeError(
                f"MultiHead V2 joint head produced malformed actions: {pred.shape}"
            )

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
        query_indices = model_indices_for_control_ticks(100, 30, selected_ticks)
        full_control_indices = model_indices_for_control_ticks(100, 30, 30)
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
            "output_head": "full_ModelWrapper.nets.joint_head.*",
            "joint_only_forward": True,
            "flow_action_channels_used_for_robot": False,
            "prompt": self._prompt,
            "proprio_physical": proprio_physical.tolist(),
            "proprio_model_input": normalized_state[0].float().cpu().tolist(),
            "prefix_shape": list(prefix_features.shape),
            "prefix_valid_tokens": int(prefix_pad_mask.sum().item()),
            "model_chunk_shape": [100, 7],
            "native_control_horizon": 30,
            "native_gripper_logits": native_gripper_logits.tolist(),
            "native_gripper_open_probability": (
                native_gripper_open_probability.tolist()
            ),
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
        pi05_legacy.PI05_CHECKPOINT_PROFILES = MULTIHEAD_V2_CHECKPOINT_PROFILES
        return pi05_legacy.parse_args()
    finally:
        pi05_legacy.PI05_CHECKPOINT_PROFILES = old_profiles


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
            f"checkpoint SHA-256 mismatch: expected {args.checkpoint_sha256}, "
            f"got {actual_sha256}"
        )
    joint_limits, gripper_stroke_mm = legacy.load_calibration(args.calibration_json)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    policy = EgoVersePI05MultiHeadV2JointPolicy(
        args, joint_limits, gripper_stroke_mm
    )
    if args.mode == "checkpoint":
        logger.info(
            "MultiHead V2 checkpoint loaded without robot/camera/CAN: "
            "sha256=%s profile=%s contract=%s",
            actual_sha256,
            args.checkpoint_profile or "custom",
            policy.checkpoint_report,
        )
        print(
            "PI05_MULTIHEAD_V2_CHECKPOINT_OK "
            f"profile={args.checkpoint_profile or 'custom'} "
            f"sha256={actual_sha256} robot_constructed=false "
            "camera_opened=false can_opened=false"
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = legacy.JsonlLogger(
        args.log_dir / f"pi05_multihead_v2_{args.mode}_{timestamp}.jsonl"
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
        logger.info("wrote MultiHead V2 audit log to %s", log.path)


if __name__ == "__main__":
    main()
