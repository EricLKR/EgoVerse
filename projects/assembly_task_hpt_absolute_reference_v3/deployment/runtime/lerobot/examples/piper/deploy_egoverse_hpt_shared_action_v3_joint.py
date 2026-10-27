#!/usr/bin/env python
"""Guarded direct-joint deployment for EgoVerse HPT Shared-Action V3.

This additive entry point leaves every previous deployment script unchanged.
It restores the complete training Lightning ``ModelWrapper`` and executes the
same shared HPT trunk and native 30-step V3 decoder used during validation.

The robot consumes only the PIPER projection: six absolute joint angles in
radians plus the independent binary gripper.  The shared 9-D camera-frame pose
is retained in diagnostics but never converted with IK.  Deployment therefore
uses no IK, no FK, no 30<->100 interpolation, and no deployment-only head.

``checkpoint`` mode constructs no robot and opens no camera or CAN interface.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

_TRAINING_ROOT = Path(
    os.environ.get(
        "EGOVERSE_TRAINING_ROOT",
        "/mnt/wd10/kerun/egoverse_piper/EgoVerse",
    )
).expanduser().resolve()
if _TRAINING_ROOT.is_dir() and str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import numpy as np
import torch

import deploy_egoverse_hpt_joint as legacy
import deploy_egoverse_pi05_joint as queue_backend
from lerobot.policies.egoverse_hpt.modeling_egoverse_hpt_joint import (
    adjacent_joint_rms,
    model_indices_for_control_ticks,
    prepare_rgb_chw_image,
    smooth_joint_action_chunk,
)

logger = logging.getLogger("deploy_egoverse_hpt_shared_action_v3_joint")

HPT_SHARED_V3_CHECKPOINT_PROFILES: dict[str, dict[str, str]] = {
    "hpt-v3-trim-val-best-5k": {
        "checkpoint": (
            "/mnt/wd10/kerun/checkpoints/"
            "hpt_shared_v3_human_trim_val_best_step5000.ckpt"
        ),
        "sha256": "e228df6d82154ace89598efc7a52ee09d2d95982f7cb52d7a053c5341d126761",
        "model_action_horizon": 30,
        "description": (
            "HPT Shared-Action V3 human-tail-trim, validation composite-score "
            "selected step 5000"
        ),
    },
    "hpt-v3-trim-joint-candidate-15k": {
        "checkpoint": (
            "/mnt/wd10/kerun/checkpoints/"
            "hpt_shared_v3_human_trim_joint_candidate_step15000.ckpt"
        ),
        "sha256": "72cf7c06723eed7b873e38301b248b7b86a599953a81db60de5721bdde5fc937",
        "model_action_horizon": 30,
        "description": (
            "HPT Shared-Action V3 human-tail-trim, validation joint-only "
            "candidate step 15000"
        ),
    },
}


def validate_deploy_result(
    result: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and unpack the exact native-30 V3 deployment contract."""
    required = {
        "actions",
        "actions_normalized",
        "shared_pose",
        "gripper_logits",
        "normalized_state",
    }
    missing = required.difference(result)
    if missing:
        raise RuntimeError(f"HPT V3 deployment result lacks {sorted(missing)}")
    actions = torch.as_tensor(result["actions"]).detach().float().cpu().numpy()
    actions_normalized = (
        torch.as_tensor(result["actions_normalized"]).detach().float().cpu().numpy()
    )
    shared_pose = (
        torch.as_tensor(result["shared_pose"]).detach().float().cpu().numpy()
    )
    logits = (
        torch.as_tensor(result["gripper_logits"]).detach().float().cpu().numpy()
    )
    state = (
        torch.as_tensor(result["normalized_state"]).detach().float().cpu().numpy()
    )
    expected = {
        "actions": ((1, 30, 7), actions.shape),
        "actions_normalized": ((1, 30, 7), actions_normalized.shape),
        "shared_pose": ((1, 30, 9), shared_pose.shape),
        "gripper_logits": ((1, 30), logits.shape),
        "normalized_state": ((1, 7), state.shape),
    }
    malformed = {
        name: {"expected": wanted, "actual": actual}
        for name, (wanted, actual) in expected.items()
        if tuple(actual) != wanted
    }
    if malformed:
        raise RuntimeError(f"malformed HPT V3 deployment result: {malformed}")
    if not all(
        np.isfinite(value).all()
        for value in (actions, actions_normalized, shared_pose, logits, state)
    ):
        raise RuntimeError("HPT V3 deployment result contains NaN/Inf")
    if not np.isin(actions[..., 6], (0.0, 1.0)).all():
        raise RuntimeError("HPT V3 deployment gripper output is not binary")
    return actions[0], actions_normalized[0], shared_pose[0], logits[0], state[0]


class EgoVerseHPTSharedActionV3JointPolicy(
    queue_backend.EgoVersePI05JointPolicy
):
    """Complete HPT V3 ModelWrapper behind the audited direct-joint queue."""

    name = "egoverse_hpt_shared_action_v3_joint"

    def __init__(
        self,
        args: argparse.Namespace,
        joint_limits_deg: list[tuple[float, float]],
        gripper_stroke_mm: float,
    ) -> None:
        # Reuse only the audited direct-joint queue/filter methods.  The pi0.5
        # model constructor is deliberately bypassed.
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            device=str(args.device),
            chunk_size=30,
            action_horizon_frames=30,
            query_frequency=int(args.query_frequency),
            num_inference_steps=0,
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

        if not _TRAINING_ROOT.is_dir():
            raise FileNotFoundError(
                "full training checkout is unavailable; set EGOVERSE_TRAINING_ROOT "
                f"(resolved root={_TRAINING_ROOT})"
            )
        from egomimic.pl_utils.pl_model import ModelWrapper

        self._model_wrapper = ModelWrapper.load_from_checkpoint(
            str(args.checkpoint), weights_only=False, map_location="cpu"
        ).to(args.device)
        self._model_wrapper.eval()
        self._model_wrapper.model.device = torch.device(args.device)
        self._algo = self._model_wrapper.model
        policy = self._algo.nets["policy"]
        if set(policy.heads) != {"shared"}:
            raise RuntimeError(
                f"HPT V3 checkpoint must contain only shared head, got {list(policy.heads)}"
            )
        head = policy.heads["shared"]
        if getattr(head, "contract_name", None) != "hpt_mixed_shared_action_v3":
            raise RuntimeError("checkpoint is not HPT Shared-Action V3")
        if int(getattr(head, "contract_version", -1)) != 3:
            raise RuntimeError("HPT Shared-Action contract version is not 3")
        if int(head.action_horizon) != 30:
            raise RuntimeError("HPT V3 checkpoint action horizon is not 30")
        if not callable(getattr(self._algo, "forward_joint_deploy", None)):
            raise RuntimeError(
                "training checkout lacks HPT V3 forward_joint_deploy; sync the "
                "matching V3 runtime before deployment"
            )
        self.checkpoint_report = {
            "architecture": "EgoVerse_HPT_Shared_Action_V3",
            "checkpoint_profile": args.checkpoint_profile or "custom",
            "loader": "complete_Lightning_ModelWrapper",
            "deployment_forward": "ModelWrapper.model.forward_joint_deploy",
            "shared_trunk": "HPT_16_blocks_256D",
            "shared_output": "native_30_step_camera_frame_9D_pose",
            "robot_output": "native_30_step_six_absolute_joints_plus_binary_gripper",
            "action_space": "six_absolute_PIPER_joint_radians_plus_binary_gripper",
            "action_horizon": 30,
            "observation_history": 1,
            "normalization": "checkpoint_ModelWrapper_quantile_stats",
            "command_generation": "direct_joint_no_IK_no_FK_no_interpolation",
            "contract_name": head.contract_name,
            "contract_version": head.contract_version,
            "inference_sampling": "deterministic_decoder_no_flow_matching",
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
        current_raw = torch.from_numpy(
            np.concatenate((obs_joint, obs_gripper)).astype(np.float32)
        ).unsqueeze(0).to(device)

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
            current_raw=current_raw,
        )
        inference_ms = (time.perf_counter() - started) * 1e3
        pred, normalized, shared_pose, gripper_logits, normalized_state = (
            validate_deploy_result(result)
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
            "output_head": "full_ModelWrapper.HPT_shared_head.PIPER_projection",
            "joint_only_forward": True,
            "proprio_physical": current_raw[0].float().cpu().tolist(),
            "proprio_model_input": normalized_state.tolist(),
            "model_chunk_shape": [30, 7],
            "shared_pose_shape": [30, 9],
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
            "full_model_chunk_normalized": normalized.tolist(),
            "full_model_chunk_physical": pred.tolist(),
            "full_shared_pose_physical": shared_pose.tolist(),
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
    old_profiles = legacy.CHECKPOINT_PROFILES
    try:
        legacy.CHECKPOINT_PROFILES = HPT_SHARED_V3_CHECKPOINT_PROFILES
        args = legacy.parse_args()
    finally:
        legacy.CHECKPOINT_PROFILES = old_profiles
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

    policy = EgoVerseHPTSharedActionV3JointPolicy(
        args, joint_limits, gripper_stroke_mm
    )
    if args.mode == "checkpoint":
        print(
            "HPT_SHARED_ACTION_V3_CHECKPOINT_OK "
            f"profile={args.checkpoint_profile or 'custom'} "
            f"sha256={actual_sha256} robot_constructed=false "
            "camera_opened=false can_opened=false "
            f"contract={policy.checkpoint_report}"
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = legacy.JsonlLogger(
        args.log_dir / f"hpt_shared_v3_{args.mode}_{timestamp}.jsonl"
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
                "inference_steps": "not_applicable_deterministic_HPT_decoder",
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
        logger.info("wrote HPT V3 deployment audit log to %s", log.path)


if __name__ == "__main__":
    main()
