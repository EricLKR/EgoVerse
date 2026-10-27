#!/usr/bin/env python
"""Guarded PIPER deployment for EgoVerse pi0.5 Expert-Action V3.

This additive entry point leaves every earlier HPT/pi0.5 deployment unchanged.
It strictly restores the complete training Lightning ``ModelWrapper`` and
calls ``PIEgoVerseExpertActionV3.forward_joint_deploy``.  Consequently the
robot command is produced by the same Gemma action expert used in training:

* OpenPI channels 0..8: shared camera-frame pose (training/evaluation only);
* OpenPI channels 9..14: native 30-step PIPER joint residual trajectory;
* action-expert hidden state: native 30-step binary gripper logits.

There is no 30<->100 interpolation, IK, or deployment-only joint decoder.
``checkpoint`` mode constructs no robot and opens no camera or CAN interface.
"""

from __future__ import annotations

import argparse
import logging
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
_BOOTSTRAP_RUNTIME = Path("/mnt/wd10/kerun/pi05_runtime")
if _BOOTSTRAP_RUNTIME.is_dir() and str(_BOOTSTRAP_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_RUNTIME))

import numpy as np
import torch

import deploy_egoverse_hpt_joint as legacy
import deploy_egoverse_pi05_joint as pi05_legacy
import deploy_egoverse_pi05_multihead_v2_joint as pi05_multihead
from lerobot.policies.egoverse_hpt.modeling_egoverse_hpt_joint import (
    adjacent_joint_rms,
    model_indices_for_control_ticks,
    prepare_rgb_chw_image,
    smooth_joint_action_chunk,
)

logger = logging.getLogger("deploy_egoverse_pi05_expert_action_v3_joint")

EXPERT_ACTION_V3_CHECKPOINT_PROFILES: dict[str, dict[str, str]] = {
    "pi05-expert-action-v3-20k": {
        "checkpoint": (
            "/mnt/wd10/kerun/checkpoints/"
            "pi05_expert_action_v3_step20000.ckpt"
        ),
        "sha256": "1fc97357220bd6531523b2fa93aad563a163279979d6e84419d41a3e5db638d2",
        "model_action_horizon": 30,
        "description": (
            "EgoVerse pi0.5 Expert-Action V3 mixed PIPER+human, "
            "formal final step 20000 candidate"
        ),
    },
}


def validate_v3_deploy_result(
    result: dict[str, torch.Tensor | list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and unpack the exact V3 joint-only deployment contract."""
    required = {"actions", "action32", "gripper_logits", "normalized_state"}
    missing = required.difference(result)
    if missing:
        raise RuntimeError(f"V3 deployment result lacks {sorted(missing)}")
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
        raise RuntimeError(f"malformed V3 deployment result: {malformed}")
    if not all(
        np.isfinite(value).all()
        for value in (actions, action32, logits, normalized_state)
    ):
        raise RuntimeError("V3 deployment result contains NaN/Inf")
    gripper = actions[..., 6]
    if not np.isin(gripper, (0.0, 1.0)).all():
        raise RuntimeError("V3 deployment gripper output is not binary")
    return actions[0], action32[0], logits[0], normalized_state[0]


class EgoVersePI05ExpertActionV3JointPolicy(
    pi05_legacy.EgoVersePI05JointPolicy
):
    """Full V3 Gemma action expert behind the audited direct-joint queue."""

    name = "egoverse_pi05_expert_action_v3_joint"

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

        self._model_wrapper = (
            pi05_multihead.EgoVersePI05MultiHeadV2JointPolicy
            ._load_full_model_wrapper(args)
        )
        self._algo = self._model_wrapper.model
        if not callable(getattr(self._algo, "forward_joint_deploy", None)):
            raise RuntimeError(
                "checkpoint model does not implement V3 forward_joint_deploy"
            )
        if int(self._algo.config.model.action_dim) != 32:
            raise RuntimeError("V3 checkpoint action_dim is not 32")
        if int(self._algo.config.model.action_horizon) != 30:
            raise RuntimeError("V3 checkpoint action_horizon is not 30")
        auxiliary = (
            self._algo.nets["auxiliary"]
            if "auxiliary" in self._algo.nets
            else None
        )
        if auxiliary is None:
            raise RuntimeError("V3 checkpoint lacks nets.auxiliary")
        if getattr(auxiliary, "contract_name", None) != "egoverse_pi05_expert_action_v3":
            raise RuntimeError(
                "checkpoint auxiliary contract is not Expert-Action V3"
            )
        if int(getattr(auxiliary, "contract_version", -1)) != 3:
            raise RuntimeError("checkpoint auxiliary contract version is not 3")

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
            "architecture": "EgoVerse_pi0.5_Expert_Action_V3",
            "checkpoint_profile": args.checkpoint_profile or "custom",
            "loader": "complete_Lightning_ModelWrapper_strict_assign",
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
            "model_wrapper_state_dict_strict": True,
            "contract_name": auxiliary.contract_name,
            "contract_version": auxiliary.contract_version,
            "joint_residual_scale_deg": float(
                np.degrees(auxiliary.joint_residual_scale_rad)
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
        noise = self._fixed_rollout_noise.clone()
        started = time.perf_counter()
        result = self._algo.forward_joint_deploy(
            front_image=front.unsqueeze(0).to(device),
            wrist_image=wrist.unsqueeze(0).to(device),
            current_raw=current_raw,
            prompt=self._prompt,
            noise=noise,
            num_steps=self.config.num_inference_steps,
        )
        inference_ms = (time.perf_counter() - started) * 1e3
        pred, action32, gripper_logits, normalized_state = (
            validate_v3_deploy_result(result)
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
            EXPERT_ACTION_V3_CHECKPOINT_PROFILES
        )
        args = pi05_legacy.parse_args()
    finally:
        pi05_legacy.PI05_CHECKPOINT_PROFILES = old_profiles
    if not args.list_checkpoint_profiles and args.chunk_smoothing_window > 30:
        raise SystemExit(
            "--chunk-smoothing-window must not exceed the native 30-step horizon"
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

    policy = EgoVersePI05ExpertActionV3JointPolicy(
        args, joint_limits, gripper_stroke_mm
    )
    if args.mode == "checkpoint":
        logger.info(
            "Expert-Action V3 checkpoint loaded without robot/camera/CAN: "
            "sha256=%s profile=%s contract=%s",
            actual_sha256,
            args.checkpoint_profile or "custom",
            policy.checkpoint_report,
        )
        print(
            "PI05_EXPERT_ACTION_V3_CHECKPOINT_OK "
            f"profile={args.checkpoint_profile or 'custom'} "
            f"sha256={actual_sha256} robot_constructed=false "
            "camera_opened=false can_opened=false"
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = legacy.JsonlLogger(
        args.log_dir / f"pi05_expert_action_v3_{args.mode}_{timestamp}.jsonl"
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
        logger.info("wrote Expert-Action V3 audit log to %s", log.path)


if __name__ == "__main__":
    main()
