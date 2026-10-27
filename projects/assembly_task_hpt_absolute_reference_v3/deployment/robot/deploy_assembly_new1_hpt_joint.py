#!/usr/bin/env python
"""Guarded direct joint-space deployment for an EgoVerse PIPER checkpoint.

Modes form a strict escalation ladder:

* ``checkpoint``: load/check checkpoint only; no robot is constructed.
* ``offline``: infer from an NPZ; no robot is constructed.
* ``live-dry-run``: connect cameras and CAN feedback observation-only.
* ``execute``: adopt the all-enabled state left by model-free Home only after
  camera/model warmup and an initial absolute-joint continuity check.  Requires
  ``--allow-motion`` and never quick-stops/re-enables the arm itself.

This script never computes Cartesian poses and never calls FK or IK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.policies.egoverse_hpt.configuration_egoverse_hpt_joint import (
    EgoVerseHPTJointConfig,
)
from lerobot.policies.egoverse_hpt.modeling_egoverse_hpt_joint import (
    EgoVerseHPTJointPolicy,
)
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots.piper.config_piper import PiperConfig
from lerobot.robots.piper.piper import Piper
from lerobot.utils.robot_utils import precise_sleep

from joint_space_adapter import JointSpaceAdapter

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = BUNDLE_ROOT / "assets/piper_follower_calibration.json"
DEFAULT_HARDWARE_CALIBRATION = (
    BUNDLE_ROOT / "assets/piper_original_leader_calibration.json"
)
DEFAULT_ROLE_MAPPING = BUNDLE_ROOT / "assets/reset_control.json"
DEFAULT_JOINT_ZERO_OFFSET = BUNDLE_ROOT / "assets/joint_zero_offset.json"
DEFAULT_LOG_DIR = BUNDLE_ROOT / "logs/hpt_joint"
DEFAULT_INITIAL_POSE = (
    BUNDLE_ROOT / "assets/initial_pose_follower_physical_on_original_leader.json"
)
FRONT_SERIAL = "337322071261"
WRIST_SERIAL = "427622273943"
SIDE_SERIAL = "427622272585"
FRONT_CAMERA_AUTO_WARMUP_S = 5.0
J2_INDEX = 1
J3_INDEX = 2
J5_INDEX = 4

# Formal, audited deployment checkpoints. A profile binds the immutable
# checkpoint path to its SHA-256 so switching models cannot accidentally load
# a partially copied or differently named file. Explicit --checkpoint and
# --checkpoint-sha256 remain available for one-off offline compatibility tests.
CHECKPOINT_PROFILES: dict[str, dict[str, Any]] = {
    "assembly-new1-hpt-mixed-v2-30k": {
        "checkpoint": str(BUNDLE_ROOT / "checkpoints/assembly_new1_hpt_mixed_v2_step30000.ckpt"),
        "sha256": "073da7659afd0fbef074165219b82c4743ef998a5f78252b762b1fdda99ab9a9",
        "model_action_horizon": 100,
        "description": "Assembly New1 three-camera HPT Mixed Clean V2, step 30000",
    },
    "assembly-new1-hpt-expert-action-v3-30k": {
        "checkpoint": str(BUNDLE_ROOT / "checkpoints/assembly_new1_hpt_expert_action_v3_step30000.ckpt"),
        "sha256": "6fc8157d0b8a138c12f557c2c62f761613f90a73175978000e2d1068a36a800d",
        "model_action_horizon": 30,
        "description": "Assembly New1 three-camera HPT Expert-Action V3, step 30000",
    },
    "assembly-new1-hpt-absolute-reference-v3-25k": {
        "checkpoint": str(
            BUNDLE_ROOT
            / "checkpoints/assembly_new1_hpt_absolute_reference_v3_step25000.ckpt"
        ),
        "sha256": "cacec0062862f268479b090cea1bcc68593346d296a8eadf0001a9dc3b68a5ec",
        "model_action_horizon": 100,
        "description": (
            "Assembly New1 three-camera HPT Absolute-Reference V3, step 25000"
        ),
    },
    "assembly-new1-hpt-absolute-reference-v3-30k": {
        "checkpoint": str(
            BUNDLE_ROOT
            / "checkpoints/assembly_new1_hpt_absolute_reference_v3_step30000.ckpt"
        ),
        "sha256": "2c0fb0c724de01a24bdfd8c29e6e22de14d784c64a2bc8de275593035589e992",
        "model_action_horizon": 100,
        "description": (
            "Assembly New1 three-camera HPT Absolute-Reference V3, step 30000"
        ),
    },
}

logger = logging.getLogger("deploy_egoverse_hpt_joint")


@dataclass(frozen=True)
class InitialPose:
    """Follower pose recorded by the data-collection reset controller."""

    joints_deg: tuple[float, ...]
    gripper_mm: float


def load_initial_pose(
    path: Path,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> InitialPose:
    with path.open() as stream:
        payload = json.load(stream)
    joints = tuple(float(value) for value in payload["joint_positions_deg"])
    gripper = float(payload["gripper_opening_mm"])
    if len(joints) != 6 or not all(math.isfinite(value) for value in (*joints, gripper)):
        raise ValueError(f"invalid initial pose in {path}")
    for index, (angle, (low, high)) in enumerate(zip(joints, joint_limits, strict=True)):
        if not low <= angle <= high:
            raise ValueError(
                f"initial pose joint {index}={angle} deg is outside calibration [{low}, {high}]"
            )
    if not 0.0 <= gripper <= gripper_stroke_mm:
        raise ValueError(
            f"initial pose gripper={gripper} mm is outside [0,{gripper_stroke_mm}]"
        )
    return InitialPose(joints, gripper)


def load_hardware_initial_pose(
    path: Path,
    adapter: JointSpaceAdapter,
) -> InitialPose:
    """Map the saved model/follower Home through the full-trajectory adapter."""
    model_pose = load_initial_pose(
        path,
        list(adapter.model_limits_deg),
        adapter.hardware_gripper_stroke_mm,
    )
    hardware_joints = adapter.model_degrees_to_hardware_degrees(
        model_pose.joints_deg
    )
    return InitialPose(
        tuple(float(value) for value in hardware_joints),
        model_pose.gripper_mm,
    )


def physical_pose_to_driver_action(
    joints_deg: tuple[float, ...],
    gripper_mm: float,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> dict[str, float]:
    values = [
        (angle - low) / (high - low) * 200.0 - 100.0
        for angle, (low, high) in zip(joints_deg, joint_limits, strict=True)
    ]
    values.append(gripper_mm / gripper_stroke_mm * 100.0)
    return {f"joint_{index}.pos": float(value) for index, value in enumerate(values)}


def quintic_blend(fraction: float) -> float:
    t = max(0.0, min(1.0, float(fraction)))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def quintic_joint_duration(
    start_joints_deg: tuple[float, ...],
    goal_joints_deg: tuple[float, ...],
    velocity_deg_s: float,
    acceleration_deg_s2: float,
) -> float:
    delta = max(abs(start - goal) for start, goal in zip(start_joints_deg, goal_joints_deg, strict=True))
    return max(
        1.0 / 30.0,
        1.875 * delta / velocity_deg_s,
        math.sqrt(5.774 * delta / acceleration_deg_s2),
    )


def checkpoint_profile_listing() -> str:
    lines = []
    for name, profile in CHECKPOINT_PROFILES.items():
        lines.append(
            f"{name}\n"
            f"  checkpoint: {profile['checkpoint']}\n"
            f"  sha256:     {profile['sha256']}\n"
            f"  model action horizon: {profile['model_action_horizon']}\n"
            f"  description: {profile['description']}"
        )
    return "\n".join(lines)


def resolve_checkpoint_selection(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """Resolve a named immutable profile or a fully explicit path/SHA pair."""
    if args.list_checkpoint_profiles:
        print(checkpoint_profile_listing())
        return args
    if args.checkpoint_profile is not None:
        if args.checkpoint is not None or args.checkpoint_sha256 is not None:
            parser.error(
                "--checkpoint-profile cannot be combined with --checkpoint or "
                "--checkpoint-sha256"
            )
        profile = CHECKPOINT_PROFILES[args.checkpoint_profile]
        args.checkpoint = Path(profile["checkpoint"])
        args.checkpoint_sha256 = profile["sha256"]
        args.checkpoint_description = profile["description"]
        args.model_action_horizon = int(profile["model_action_horizon"])
        return args
    if args.checkpoint is None or args.checkpoint_sha256 is None:
        parser.error(
            "choose --checkpoint-profile, or provide both --checkpoint and "
            "--checkpoint-sha256"
        )
    args.checkpoint_description = "explicit custom checkpoint"
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_calibration(path: Path) -> tuple[list[tuple[float, float]], float]:
    with path.open() as stream:
        calibration = json.load(stream)
    limits = [
        (
            float(calibration[f"joint_{index}"]["range_min"]) / 1000.0,
            float(calibration[f"joint_{index}"]["range_max"]) / 1000.0,
        )
        for index in range(6)
    ]
    gripper_stroke_mm = float(calibration["gripper"]["range_max"]) / 1000.0
    if any(high <= low for low, high in limits):
        raise ValueError(f"invalid joint limits in {path}: {limits}")
    if gripper_stroke_mm <= 0:
        raise ValueError(f"invalid gripper stroke in {path}: {gripper_stroke_mm}")
    return limits, gripper_stroke_mm


def make_policy(
    args: argparse.Namespace, joint_limits: list[tuple[float, float]], gripper_stroke_mm: float
) -> EgoVerseHPTJointPolicy:
    config = EgoVerseHPTJointConfig(
        ckpt_path=str(args.checkpoint),
        joint_limits_deg=joint_limits,
        gripper_stroke_mm=gripper_stroke_mm,
        device=args.device,
        action_horizon_frames=30,
        chunk_size=args.model_action_horizon,
        query_frequency=args.query_frequency,
        num_inference_steps=args.inference_steps,
        inference_seed=args.seed,
        side_camera_key="side",
        require_residual_joint_head=not args.allow_legacy_joint_head,
        expected_image_height=480,
        expected_image_width=640,
        resize_mismatched_images=False,
        gripper_threshold=args.gripper_threshold,
        joint_limit_margin_rad=np.radians(args.joint_limit_margin_deg),
        max_joint_command_step=args.max_joint_command_step,
        max_joint_command_accel=args.max_joint_command_accel,
        temporal_ensemble=args.temporal_ensemble,
        temporal_replan_interval=args.temporal_replan_interval,
        temporal_ensemble_decay=args.temporal_ensemble_decay,
        chunk_smoothing=args.chunk_smoothing,
        chunk_smoothing_window=args.chunk_smoothing_window,
        chunk_smoothing_sigma=args.chunk_smoothing_sigma,
    )
    return EgoVerseHPTJointPolicy(config)


def make_policy_batch(
    state: np.ndarray,
    front_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    side_rgb: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    observation = {
        "observation.state": np.asarray(state),
        "observation.images.435if": np.asarray(front_rgb),
        "observation.images.405": np.asarray(wrist_rgb),
        "observation.images.side": np.asarray(side_rgb),
    }
    return prepare_observation_for_inference(
        observation, device, task=None, robot_type="piper"
    )


def live_observation_to_batch(
    observation: dict[str, Any],
    device: torch.device,
    adapter: JointSpaceAdapter | None = None,
) -> dict[str, torch.Tensor]:
    if adapter is not None:
        observation = adapter.hardware_observation_to_model_observation(observation)
    raw_state = np.asarray(
        [observation[f"joint_{index}.pos"] for index in range(7)], dtype=np.float32
    )
    return make_policy_batch(
        raw_state,
        observation["435if"],
        observation["405"],
        observation["side"],
        device,
    )


def hardware_observation_for_model(
    observation: dict[str, Any],
    j2_command_offset_deg: float,
    j3_command_offset_deg: float = 0.0,
    j5_command_offset_deg: float = 0.0,
) -> dict[str, Any]:
    """Convert raw firmware feedback to the joint conventions used by the model.

    Each command offset is a hardware-minus-model offset. Therefore the inverse
    mapping is used for proprioception: model = firmware - offset. Images and
    every non-offset field are preserved unchanged.
    """

    converted = dict(observation)
    for joint_name, joint_index, requested_offset in (
        ("J2", J2_INDEX, j2_command_offset_deg),
        ("J3", J3_INDEX, j3_command_offset_deg),
        ("J5", J5_INDEX, j5_command_offset_deg),
    ):
        offset = float(requested_offset)
        if not math.isfinite(offset):
            raise ValueError(f"{joint_name} command offset must be finite")
        key = f"joint_{joint_index}.pos"
        firmware_raw = float(observation[key])
        model_raw = firmware_raw - offset * 1000.0
        if not math.isfinite(model_raw):
            raise RuntimeError(
                f"non-finite model-space {joint_name} proprioception"
            )
        converted[key] = model_raw
    return converted


def apply_j2_command_offset_to_hardware_action(
    action: dict[str, float],
    j2_command_offset_deg: float,
    joint_limits: list[tuple[float, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Map a model-space absolute J2 target to the firmware convention."""

    offset = float(j2_command_offset_deg)
    if not math.isfinite(offset):
        raise ValueError("J2 command offset must be finite")
    converted = dict(action)
    key = f"joint_{J2_INDEX}.pos"
    normalized = float(action[key])
    low, high = joint_limits[J2_INDEX]
    model_target_deg = low + (normalized + 100.0) / 200.0 * (high - low)
    firmware_target_deg = model_target_deg + offset
    if not low <= firmware_target_deg <= high:
        raise RuntimeError(
            "J2 command offset would exceed hardware calibration: "
            f"model={model_target_deg:.6f} deg offset={offset:.6f} deg "
            f"firmware={firmware_target_deg:.6f} deg limits=[{low},{high}]"
        )
    converted[key] = (
        (firmware_target_deg - low) / (high - low) * 200.0 - 100.0
    )
    return converted, {
        "j2_command_offset_deg": offset,
        "model_j2_target_deg": model_target_deg,
        "firmware_j2_target_deg": firmware_target_deg,
        "model_j2_driver": normalized,
        "firmware_j2_driver": converted[key],
    }


def apply_j2_command_offset_to_initial_pose(
    pose: InitialPose,
    j2_command_offset_deg: float,
    joint_limits: list[tuple[float, float]],
) -> InitialPose:
    """Map a saved model-space Home pose to its firmware-space target."""

    joints = list(pose.joints_deg)
    joints[J2_INDEX] += float(j2_command_offset_deg)
    low, high = joint_limits[J2_INDEX]
    if not low <= joints[J2_INDEX] <= high:
        raise RuntimeError(
            "shifted Home J2 is outside hardware calibration: "
            f"target={joints[J2_INDEX]:.6f} deg limits=[{low},{high}]"
        )
    return InitialPose(tuple(joints), pose.gripper_mm)


def apply_j3_command_offset_to_hardware_action(
    action: dict[str, float],
    j3_command_offset_deg: float,
    joint_limits: list[tuple[float, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Map a model-space absolute J3 target to the firmware convention."""

    offset = float(j3_command_offset_deg)
    if not math.isfinite(offset):
        raise ValueError("J3 command offset must be finite")
    converted = dict(action)
    key = f"joint_{J3_INDEX}.pos"
    normalized = float(action[key])
    low, high = joint_limits[J3_INDEX]
    model_target_deg = low + (normalized + 100.0) / 200.0 * (high - low)
    firmware_target_deg = model_target_deg + offset
    if not low <= firmware_target_deg <= high:
        raise RuntimeError(
            "J3 command offset would exceed hardware calibration: "
            f"model={model_target_deg:.6f} deg offset={offset:.6f} deg "
            f"firmware={firmware_target_deg:.6f} deg limits=[{low},{high}]"
        )
    converted[key] = (
        (firmware_target_deg - low) / (high - low) * 200.0 - 100.0
    )
    return converted, {
        "j3_command_offset_deg": offset,
        "model_j3_target_deg": model_target_deg,
        "firmware_j3_target_deg": firmware_target_deg,
        "model_j3_driver": normalized,
        "firmware_j3_driver": converted[key],
    }


def apply_j3_command_offset_to_initial_pose(
    pose: InitialPose,
    j3_command_offset_deg: float,
    joint_limits: list[tuple[float, float]],
) -> InitialPose:
    """Map a saved model-space Home pose to its firmware-space J3 target."""

    joints = list(pose.joints_deg)
    joints[J3_INDEX] += float(j3_command_offset_deg)
    low, high = joint_limits[J3_INDEX]
    if not low <= joints[J3_INDEX] <= high:
        raise RuntimeError(
            "shifted Home J3 is outside hardware calibration: "
            f"target={joints[J3_INDEX]:.6f} deg limits=[{low},{high}]"
        )
    return InitialPose(tuple(joints), pose.gripper_mm)


def apply_j5_command_offset_to_hardware_action(
    action: dict[str, float],
    j5_command_offset_deg: float,
    joint_limits: list[tuple[float, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Map a model-space absolute J5 target to the firmware convention."""

    offset = float(j5_command_offset_deg)
    if not math.isfinite(offset):
        raise ValueError("J5 command offset must be finite")
    converted = dict(action)
    key = f"joint_{J5_INDEX}.pos"
    normalized = float(action[key])
    low, high = joint_limits[J5_INDEX]
    model_target_deg = low + (normalized + 100.0) / 200.0 * (high - low)
    firmware_target_deg = model_target_deg + offset
    if not low <= firmware_target_deg <= high:
        raise RuntimeError(
            "J5 command offset would exceed hardware calibration: "
            f"model={model_target_deg:.6f} deg offset={offset:.6f} deg "
            f"firmware={firmware_target_deg:.6f} deg limits=[{low},{high}]"
        )
    converted[key] = (
        (firmware_target_deg - low) / (high - low) * 200.0 - 100.0
    )
    return converted, {
        "j5_command_offset_deg": offset,
        "model_j5_target_deg": model_target_deg,
        "firmware_j5_target_deg": firmware_target_deg,
        "model_j5_driver": normalized,
        "firmware_j5_driver": converted[key],
    }


def apply_j5_command_offset_to_initial_pose(
    pose: InitialPose,
    j5_command_offset_deg: float,
    joint_limits: list[tuple[float, float]],
) -> InitialPose:
    """Map a saved model-space Home pose to its firmware-space J5 target."""

    joints = list(pose.joints_deg)
    joints[J5_INDEX] += float(j5_command_offset_deg)
    low, high = joint_limits[J5_INDEX]
    if not low <= joints[J5_INDEX] <= high:
        raise RuntimeError(
            "shifted Home J5 is outside hardware calibration: "
            f"target={joints[J5_INDEX]:.6f} deg limits=[{low},{high}]"
        )
    return InitialPose(tuple(joints), pose.gripper_mm)


def action_to_robot_dict(action: torch.Tensor) -> dict[str, float]:
    values = action.detach().cpu().numpy().reshape(-1)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise RuntimeError(f"refusing malformed policy action: shape={values.shape} {values}")
    if not np.all((values[:6] >= -100.0001) & (values[:6] <= 100.0001)):
        raise RuntimeError(f"joint command outside driver range: {values.tolist()}")
    if float(values[6]) not in (0.0, 100.0):
        raise RuntimeError(f"gripper command is not binary: {float(values[6])}")
    return {f"joint_{index}.pos": float(value) for index, value in enumerate(values)}


def observation_to_driver_state(
    observation: dict[str, Any],
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> np.ndarray:
    state = np.empty(7, dtype=np.float64)
    for index, (low_deg, high_deg) in enumerate(joint_limits):
        degrees = float(observation[f"joint_{index}.pos"]) / 1000.0
        state[index] = (degrees - low_deg) / (high_deg - low_deg) * 200.0 - 100.0
    gripper_mm = float(observation["joint_6.pos"]) / 1000.0
    state[6] = np.clip(gripper_mm / gripper_stroke_mm * 100.0, 0.0, 100.0)
    if not np.isfinite(state).all():
        raise RuntimeError(f"non-finite normalized feedback: {state.tolist()}")
    return state


def raw_feedback_pose(raw: dict[str, Any]) -> InitialPose:
    joints = tuple(float(raw[f"joint_{index}.pos"]) / 1000.0 for index in range(6))
    gripper = float(raw["joint_6.pos"]) / 1000.0
    if not all(math.isfinite(value) for value in (*joints, gripper)):
        raise RuntimeError(f"non-finite PIPER feedback during homing: {raw}")
    return InitialPose(joints, gripper)


def validate_home_start_feedback(
    raw: dict[str, Any],
    args: argparse.Namespace,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> tuple[InitialPose, dict[str, Any]]:
    measured = raw_feedback_pose(raw)
    recovery_joints = []
    for index, (angle, (low, high)) in enumerate(
        zip(measured.joints_deg, joint_limits, strict=True)
    ):
        allowed_low = low - args.home_start_limit_margin_deg
        allowed_high = high + args.home_start_limit_margin_deg
        if not allowed_low <= angle <= allowed_high:
            raise RuntimeError(
                f"cannot home from joint {index}={angle} deg outside recovery range "
                f"[{allowed_low},{allowed_high}]"
            )
        if not low <= angle <= high:
            recovery_joints.append(index)
    if not (
        -args.home_gripper_tolerance_mm
        <= measured.gripper_mm
        <= gripper_stroke_mm + args.home_gripper_tolerance_mm
    ):
        raise RuntimeError(
            f"cannot home from gripper={measured.gripper_mm} mm outside endpoint margin"
        )
    clamped_gripper = min(gripper_stroke_mm, max(0.0, measured.gripper_mm))
    start = InitialPose(measured.joints_deg, clamped_gripper)
    return start, {
        "measured_joints_deg": list(measured.joints_deg),
        "measured_gripper_mm": measured.gripper_mm,
        "recovery_joints": recovery_joints,
        "gripper_endpoint_clamped": clamped_gripper != measured.gripper_mm,
        "clamped_gripper_mm": clamped_gripper,
    }


def require_deployment_pose_inside_limits(
    validation_report: dict[str, Any],
    *,
    execute: bool,
    home_at_start: bool,
) -> None:
    """Require Home recovery before inference from an out-of-range pose.

    ``validate_home_start_feedback`` deliberately accepts a narrow margin so
    model-free Home can recover a joint just outside its calibrated endpoint.
    Policy inference must not use that recovery-only state as its trajectory
    anchor: normalizing it clips to a model endpoint and later produces a
    misleading hardware-limit error.  Execute mode may proceed only when it
    will actually Home before the first inference.
    """

    recovery_joints = list(validation_report.get("recovery_joints", []))
    if recovery_joints and (not execute or not home_at_start):
        joint_labels = [int(index) + 1 for index in recovery_joints]
        raise RuntimeError(
            "deployment preflight requires model-free Home before policy inference: "
            f"joints outside calibrated hardware limits={joint_labels}. "
            "Run run_home_only.sh in execute mode, then repeat live-dry-run. "
            "No model inference or motion command was sent by this deployment run."
        )


def normalized_present_position(driver_state: np.ndarray) -> dict[str, float]:
    return {f"joint_{index}": float(driver_state[index]) for index in range(7)}


def bind_runtime_calibration(
    robot: Piper,
    raw_feedback: dict[str, Any],
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> dict[str, Any]:
    """Force the SDK and policy to use one calibration for this deployment.

    ``Robot`` normally prefers a calibration cached under its robot id over the
    limits supplied by ``PiperConfig``.  A stale cache therefore used to make
    the policy normalize feedback with the bundle calibration while
    ``send_action`` denormalized the same values with a different calibration.
    Bind the SDK mapping explicitly before motion is adopted and verify the
    result against the *same* raw feedback sample.
    """

    expected_min = [float(low) * 10.0 for low, _ in joint_limits] + [0.0]
    expected_max = [float(high) * 10.0 for _, high in joint_limits] + [100.0]
    before_min = [float(value) for value in robot.sdk.min_pos]
    before_max = [float(value) for value in robot.sdk.max_pos]
    before_gripper_stroke_mm = float(robot.sdk.gripper_stroke_mm)

    robot.sdk.min_pos = expected_min.copy()
    robot.sdk.max_pos = expected_max.copy()
    robot.sdk.gripper_stroke_mm = float(gripper_stroke_mm)

    actual_min = [float(value) for value in robot.sdk.min_pos]
    actual_max = [float(value) for value in robot.sdk.max_pos]
    if not np.allclose(actual_min, expected_min, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"failed to bind deployment joint minima: expected={expected_min}, actual={actual_min}"
        )
    if not np.allclose(actual_max, expected_max, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"failed to bind deployment joint maxima: expected={expected_max}, actual={actual_max}"
        )
    if not math.isclose(
        float(robot.sdk.gripper_stroke_mm),
        float(gripper_stroke_mm),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "failed to bind deployment gripper stroke: "
            f"expected={gripper_stroke_mm}, actual={robot.sdk.gripper_stroke_mm}"
        )

    policy_state = observation_to_driver_state(
        raw_feedback, joint_limits, gripper_stroke_mm
    )
    sdk_state = np.empty(7, dtype=np.float64)
    for index in range(6):
        raw_deg001 = float(raw_feedback[f"joint_{index}.pos"])
        raw_deg01 = raw_deg001 / 100.0
        sdk_state[index] = (
            (raw_deg01 - robot.sdk.min_pos[index])
            / (robot.sdk.max_pos[index] - robot.sdk.min_pos[index])
            * 200.0
            - 100.0
        )
    sdk_state[6] = np.clip(
        float(raw_feedback["joint_6.pos"])
        / (float(robot.sdk.gripper_stroke_mm) * 1000.0)
        * 100.0,
        0.0,
        100.0,
    )
    max_state_error = float(np.max(np.abs(policy_state - sdk_state)))
    if max_state_error > 1e-9:
        raise RuntimeError(
            "policy/SDK calibration consistency check failed before motion: "
            f"max_normalized_error={max_state_error}"
        )

    cache_override_detected = not (
        np.allclose(before_min, expected_min, rtol=0.0, atol=1e-9)
        and np.allclose(before_max, expected_max, rtol=0.0, atol=1e-9)
        and math.isclose(
            before_gripper_stroke_mm,
            float(gripper_stroke_mm),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    return {
        "cache_override_detected": bool(cache_override_detected),
        "sdk_limits_before": {"min": before_min, "max": before_max},
        "sdk_gripper_stroke_mm_before": before_gripper_stroke_mm,
        "sdk_limits_bound": {"min": actual_min, "max": actual_max},
        "sdk_gripper_stroke_mm_bound": float(robot.sdk.gripper_stroke_mm),
        "max_policy_sdk_normalized_error": max_state_error,
        "policy_state": policy_state.tolist(),
        "sdk_state": sdk_state.tolist(),
    }


def verify_homing_health(
    robot: Piper,
    *,
    require_enabled: bool = True,
    check_controller: bool = True,
    context: str = "homing",
) -> list[dict[str, Any]]:
    status = robot.sdk.piper.GetArmStatus().arm_status
    if check_controller and (
        status.err_code or status.arm_status or status.ctrl_mode != 1
    ):
        raise RuntimeError(f"PIPER controller fault during {context}: {status}")
    fault_names = (
        "voltage_too_low",
        "motor_overheating",
        "driver_overcurrent",
        "driver_overheating",
        "collision_protection_triggered",
        "driver_error",
        "stall_protection_triggered",
    )
    motors = robot.sdk.get_motor_faults()
    for index, motor in enumerate(motors):
        active = [name for name in fault_names if motor[name]]
        if active:
            raise RuntimeError(
                f"PIPER motor {index + 1} fault during {context}: {active}"
            )
        if require_enabled and not motor["driver_enabled"]:
            raise RuntimeError(
                f"PIPER motor {index + 1} lost enable during {context}"
            )
    return motors


def require_live_robot_feedback(
    robot: Piper,
    raw_feedback: dict[str, Any],
    *,
    context: str,
) -> list[dict[str, Any]]:
    """Reject an opened SocketCAN interface that has no live arm telemetry.

    The SDK initializes joint, gripper, voltage, and enable caches to zero.  An
    UP SocketCAN interface can therefore look syntactically valid even when no
    PIPER controller is replying.  Treating those cache defaults as a physical
    pose maps J2/J3 to calibration endpoints and can make a dry run fail later
    with a misleading joint-limit error.  Require actual motor-voltage
    telemetry before any Home validation or model inference.
    """

    motors = verify_homing_health(
        robot,
        require_enabled=False,
        check_controller=False,
        context=context,
    )
    if len(motors) != 6:
        raise RuntimeError(
            f"PIPER returned {len(motors)} motor records during {context}; expected 6"
        )
    voltages = np.asarray(
        [float(motor["voltage_v"]) for motor in motors], dtype=np.float64
    )
    if not np.isfinite(voltages).all():
        raise RuntimeError(
            f"non-finite PIPER motor voltage feedback during {context}: "
            f"{voltages.tolist()}"
        )
    if not np.any(voltages > 0.0):
        raw_state = [
            float(raw_feedback[f"joint_{index}.pos"]) for index in range(7)
        ]
        raise RuntimeError(
            "no live PIPER motor feedback during "
            f"{context}: motor_voltage_v={voltages.tolist()}, raw_state={raw_state}. "
            "The CAN interface may be UP while the arm is powered off, disconnected, "
            "or attached to a different CAN port. No model inference or motion was run."
        )
    return motors


def wait_for_stable_homing_health(
    robot: Piper,
    *,
    timeout_s: float = 2.0,
    consecutive_samples: int = 5,
    context: str = "motion enable",
    check_controller: bool = True,
    hold_position: dict[str, float] | None = None,
) -> None:
    """Debounce feedback and optionally sustain an adopted hold target.

    ``hold_position`` is used only during an enabled-arm software handoff.  It
    covers the firmware race where the previous connection's final CAN_CTRL
    sample arrives before the controller later falls back to STANDBY.
    """
    deadline = time.perf_counter() + timeout_s
    healthy = 0
    last_error: Exception | None = None
    while time.perf_counter() <= deadline:
        try:
            verify_homing_health(
                robot,
                context=context,
                check_controller=check_controller and hold_position is None,
            )
            if hold_position is not None:
                status = robot.sdk.piper.GetArmStatus().arm_status
                if status.err_code or status.arm_status:
                    raise RuntimeError(
                        f"PIPER controller fault during {context}: {status}"
                    )
                if status.ctrl_mode == 0:
                    healthy = 0
                    robot.sdk.piper.EnablePiper()
                    robot.sdk.set_joint_positions(
                        [
                            hold_position[f"joint_{index}.pos"]
                            for index in range(7)
                        ]
                    )
                    robot.sdk.piper.MotionCtrl_2(
                        0x01,
                        0x01,
                        int(robot.sdk.speed_percent),
                        0x00,
                    )
                elif status.ctrl_mode == 1:
                    robot.sdk.set_joint_positions(
                        [
                            hold_position[f"joint_{index}.pos"]
                            for index in range(7)
                        ]
                    )
                    healthy += 1
                else:
                    raise RuntimeError(
                        f"PIPER controller entered an unsupported mode during "
                        f"{context}: {status}"
                    )
        except RuntimeError as exc:
            healthy = 0
            last_error = exc
        else:
            if hold_position is not None:
                precise_sleep(1.0 / 30.0)
                continue
            healthy += 1
            if healthy >= consecutive_samples:
                return
        precise_sleep(1.0 / 30.0)
    if hold_position is not None and healthy >= consecutive_samples:
        return
    raise RuntimeError(
        f"PIPER feedback did not remain healthy during {context}; "
        f"last status: {last_error}"
    )


def latch_current_raw_position(robot: Piper) -> dict[str, Any]:
    """Command the measured raw pose without relying on a driver-version helper."""
    sdk_helper = getattr(robot.sdk, "latch_current_raw_position", None)
    if sdk_helper is not None:
        return sdk_helper()

    # The isolated runtime predates ``latch_current_raw_position``.  Convert a
    # single raw feedback sample through the exact SDK calibration fields and
    # send it back as the target.  This preserves torque without quick-stop or
    # DisablePiper and remains compatible with both runtime revisions.
    raw = robot.sdk.get_status()
    positions = []
    for index in range(6):
        raw_deg01 = float(raw[f"joint_{index}.pos"]) / 100.0
        low = float(robot.sdk.min_pos[index])
        high = float(robot.sdk.max_pos[index])
        if high == low:
            raise RuntimeError(f"invalid SDK joint range for joint {index}: [{low},{high}]")
        positions.append((raw_deg01 - low) / (high - low) * 200.0 - 100.0)
    positions.append(
        float(np.clip(
            (float(raw["joint_6.pos"]) / 1000.0)
            / float(robot.sdk.gripper_stroke_mm)
            * 100.0,
            0.0,
            100.0,
        ))
    )
    if not np.isfinite(np.asarray(positions, dtype=np.float64)).all():
        raise RuntimeError(f"non-finite current-pose latch target: {positions}")
    robot.sdk.set_joint_positions(positions)
    return raw


def latch_current_pose_without_disabling_torque(
    robot: Piper,
    log: "JsonlLogger",
    *,
    reason: str,
) -> None:
    """Freeze the measured pose while preserving the controller enable state.

    On the deployment arm ``EmergencyStop(0x01)`` was observed to disable all
    six motors after a short delay.  It must not be used as an automatic
    software-exit action.  Latch raw feedback instead; ``robot.disconnect()``
    is configured to close the connection without disabling torque.
    """
    raw = latch_current_raw_position(robot)
    log.write(
        {
            "event": "current_pose_latched",
            "reason": reason,
            "raw_state": [float(raw[f"joint_{index}.pos"]) for index in range(7)],
            "command_sent": True,
        }
    )


def adopt_already_enabled_motion_hold_current(robot: Piper) -> dict[str, float]:
    """Adopt the enabled pose left by model-free Home without a quick-stop.

    This deployment deliberately connects observation-only so model/camera
    preflight cannot move the arm.  Model-free Home runs in a separate process
    and leaves all six drivers enabled with the measured home pose latched.
    Calling the legacy ``enable_motion_hold_current()`` here is unsafe: that
    method always emits ``EmergencyStop(0x01)`` before re-enabling, and this
    PIPER-L has been observed to disable all motors a few seconds after that
    second quick-stop.  Instead, require the existing hardware enable state,
    opt this *software connection* into command mode, and resend the measured
    pose.  No EmergencyStop, EnablePiper, DisablePiper, or resume command is
    sent by this handoff.
    """
    if robot.sdk.motion_enabled:
        raise RuntimeError(
            "deployment connection is already marked motion-enabled; refusing "
            "an ambiguous second handoff"
        )

    # Give an immediate, actionable error if Home was not run.  All-disabled
    # means torque is absent; partially enabled is an even stronger hard fault.
    # Neither state is repaired automatically by deployment.
    status = robot.sdk.piper.GetArmStatus().arm_status
    if status.err_code or status.arm_status:
        raise RuntimeError(
            "deployment handoff requires NORMAL controller state with error code 0; "
            f"got {status}"
        )
    if status.ctrl_mode not in (0, 1):
        raise RuntimeError(
            "deployment handoff only accepts STANDBY(0) or CAN control(1); "
            f"got ctrl_mode={status.ctrl_mode}"
        )

    motors = verify_homing_health(
        robot,
        require_enabled=False,
        check_controller=False,
        context="deployment handoff precheck",
    )
    enabled = [bool(motor["driver_enabled"]) for motor in motors]
    if not all(enabled):
        raise RuntimeError(
            "deployment requires all six motors to be enabled by model-free Home; "
            f"driver_enabled={enabled}. Run run_home_only.sh execute while the "
            "arm is supported; refusing to quick-stop or re-enable from deployment."
        )
    # Debounce the enabled state and controller mode before the first target.
    wait_for_stable_homing_health(
        robot,
        timeout_s=2.0,
        consecutive_samples=5,
        context="deployment handoff enabled-state precheck",
        check_controller=False,
    )

    raw_hold = robot.sdk.get_status()
    held_position = {
        key: float(value) for key, value in robot.sdk.get_status_normalized().items()
    }
    try:
        # Preload the measured raw target while the controller is still in its
        # inherited mode.  The Home process disconnects without disabling the
        # six drivers, but this firmware returns ctrl_mode to STANDBY(0) when
        # that CAN connection closes.  Changing only MotionCtrl_2 back to
        # CAN/MOVE_J is therefore required; it is not an enable/stop command.
        primer = getattr(robot.sdk, "_prime_measured_raw_target", None)
        if primer is None:
            joints = [int(raw_hold[f"joint_{index}.pos"]) for index in range(6)]
            robot.sdk.piper.JointCtrl(*joints)
            robot.sdk.piper.GripperCtrl(
                int(raw_hold["joint_6.pos"]),
                1000,
                0x01,
                0,
            )
        else:
            primer(raw_hold)
        if status.ctrl_mode == 0:
            robot.sdk.piper.MotionCtrl_2(
                0x01,
                0x01,
                int(robot.sdk.speed_percent),
                0x00,
            )
            precise_sleep(0.1)
            transition_status = robot.sdk.piper.GetArmStatus().arm_status
            if transition_status.ctrl_mode == 0:
                deadline = time.perf_counter() + 2.0
                while time.perf_counter() <= deadline:
                    robot.sdk.piper.EnablePiper()
                    primer = getattr(robot.sdk, "_prime_measured_raw_target", None)
                    if primer is None:
                        robot.sdk.piper.JointCtrl(
                            *[
                                int(raw_hold[f"joint_{index}.pos"])
                                for index in range(6)
                            ]
                        )
                        robot.sdk.piper.GripperCtrl(
                            int(raw_hold["joint_6.pos"]),
                            1000,
                            0x01,
                            0,
                        )
                    else:
                        primer(raw_hold)
                    robot.sdk.piper.MotionCtrl_2(
                        0x01,
                        0x01,
                        int(robot.sdk.speed_percent),
                        0x00,
                    )
                    precise_sleep(0.1)
                    transition_status = robot.sdk.piper.GetArmStatus().arm_status
                    if transition_status.ctrl_mode == 1:
                        break
                    if transition_status.ctrl_mode != 0:
                        raise RuntimeError(
                            "deployment handoff left STANDBY but entered an "
                            f"unsafe control mode: {transition_status}"
                        )
                else:
                    raise RuntimeError(
                        "deployment handoff could not promote STANDBY to CAN "
                        "control within 2 seconds without quick-stop"
                    )
            elif transition_status.ctrl_mode != 1:
                raise RuntimeError(
                    "deployment handoff entered an unsafe control mode after "
                    f"MotionCtrl_2: {transition_status}"
                )

        robot.sdk.motion_enabled = True
        # Resample and latch after the mode transition, so even small gravity
        # motion during the handoff becomes the new hold target.
        latch_current_raw_position(robot)
        verifier = getattr(robot.sdk, "_verify_controller_accepts_motion", None)
        if verifier is not None:
            verifier()
        wait_for_stable_homing_health(
            robot,
            timeout_s=2.0,
            consecutive_samples=5,
            context="deployment handoff after control-mode adoption",
            check_controller=True,
            hold_position=held_position,
        )
    except BaseException:
        # Restore the truthful software state.  Do not send any stop/disable
        # command here: the physical arm may still be enabled and supporting
        # its own weight.
        robot.sdk.motion_enabled = False
        raise
    return held_position


def run_initial_pose_home(
    *,
    robot: Piper,
    goal: InitialPose,
    args: argparse.Namespace,
    log: "JsonlLogger",
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
    phase: str,
) -> None:
    """Reproduce the recorder's follower homing trajectory without leader motion.

    The arm follows a bounded quintic joint trajectory while holding its current
    gripper opening. Only after all six joints are within tolerance does the
    gripper move to the recorded opening. Every command is checked against live
    feedback and the normal deployment relative-target limiter.
    """
    recorded_goal = goal
    goal = apply_j2_command_offset_to_initial_pose(
        recorded_goal,
        args.j2_command_offset_deg,
        joint_limits,
    )
    goal = apply_j3_command_offset_to_initial_pose(
        goal,
        args.j3_command_offset_deg,
        joint_limits,
    )
    goal = apply_j5_command_offset_to_initial_pose(
        goal,
        args.j5_command_offset_deg,
        joint_limits,
    )
    started = time.perf_counter()
    deadline = started + args.home_timeout_s
    start_raw = robot.sdk.get_status()
    start_pose, start_report = validate_home_start_feedback(
        start_raw,
        args,
        joint_limits,
        gripper_stroke_mm,
    )

    joint_duration = quintic_joint_duration(
        start_pose.joints_deg,
        goal.joints_deg,
        args.home_velocity_deg_s,
        args.home_acceleration_deg_s2,
    )
    initial_joint_error = max(
        abs(value - target)
        for value, target in zip(start_pose.joints_deg, goal.joints_deg, strict=True)
    )
    initial_gripper_error = abs(start_pose.gripper_mm - goal.gripper_mm)
    log.write(
        {
            "event": "initial_pose_home_start",
            "phase": phase,
            "start_joints_deg": list(start_pose.joints_deg),
            "recorded_model_goal_joints_deg": list(recorded_goal.joints_deg),
            "firmware_goal_joints_deg": list(goal.joints_deg),
            "goal_joints_deg": list(goal.joints_deg),
            "j2_command_offset_deg": args.j2_command_offset_deg,
            "j3_command_offset_deg": args.j3_command_offset_deg,
            "j5_command_offset_deg": args.j5_command_offset_deg,
            "start_gripper_mm": start_pose.gripper_mm,
            "goal_gripper_mm": goal.gripper_mm,
            "joint_trajectory_duration_s": joint_duration,
            "initial_max_joint_error_deg": initial_joint_error,
            "initial_gripper_error_mm": initial_gripper_error,
            "start_feedback_validation": start_report,
            "command_sent": False,
        }
    )
    wait_for_stable_homing_health(robot)

    if (
        initial_joint_error <= args.home_joint_tolerance_deg
        and initial_gripper_error <= args.home_gripper_tolerance_mm
    ):
        # The guarded motion handoff has already latched this measured pose.
        # Do not switch controller mode or send redundant targets when the arm
        # is already at the recorder pose; some Piper firmware revisions emit
        # transient all-joint communication faults during redundant mode/target
        # changes immediately after enable.
        hold_deadline = time.perf_counter() + args.home_hold_s
        while time.perf_counter() < hold_deadline:
            verify_homing_health(robot, check_controller=False)
            precise_sleep(1.0 / 30.0)
        wait_for_stable_homing_health(robot)
        final_raw = robot.sdk.get_status()
        final_pose = raw_feedback_pose(final_raw)
        log.write(
            {
                "event": "initial_pose_home_complete",
                "phase": phase,
                "already_within_tolerance": True,
                "elapsed_s": time.perf_counter() - started,
                "final_joints_deg": list(final_pose.joints_deg),
                "final_gripper_mm": final_pose.gripper_mm,
                "max_joint_error_deg": initial_joint_error,
                "gripper_error_mm": initial_gripper_error,
                "command_sent": False,
            }
        )
        return

    stage_started = time.perf_counter()
    tick = 0
    while True:
        now = time.perf_counter()
        if now > deadline:
            raise RuntimeError(f"{phase} initial-pose homing timed out during joint return")
        raw = robot.sdk.get_status()
        observed = raw_feedback_pose(raw)
        driver_state = observation_to_driver_state(raw, joint_limits, gripper_stroke_mm)
        verify_homing_health(robot, check_controller=False)
        elapsed = now - stage_started
        blend = quintic_blend(elapsed / joint_duration)
        target_joints = tuple(
            start + (end - start) * blend
            for start, end in zip(start_pose.joints_deg, goal.joints_deg, strict=True)
        )
        command = physical_pose_to_driver_action(
            target_joints,
            start_pose.gripper_mm,
            joint_limits,
            gripper_stroke_mm,
        )
        sent = robot.send_action(
            command,
            present_position=normalized_present_position(driver_state),
        )
        joint_error = max(
            abs(value - target)
            for value, target in zip(observed.joints_deg, goal.joints_deg, strict=True)
        )
        if tick % 30 == 0:
            log.write(
                {
                    "event": "initial_pose_home_progress",
                    "phase": phase,
                    "stage": "joints",
                    "elapsed_s": now - started,
                    "observed_joints_deg": list(observed.joints_deg),
                    "max_joint_error_deg": joint_error,
                    "proposed_action": command,
                    "sent_action": sent,
                    "command_sent": True,
                }
            )
        if elapsed >= joint_duration and joint_error <= args.home_joint_tolerance_deg:
            break
        tick += 1
        precise_sleep(1.0 / 30.0 - (time.perf_counter() - now))

    grip_start = observed.gripper_mm
    gripper_duration = max(
        1.0 / 30.0,
        1.875 * abs(grip_start - goal.gripper_mm) / args.home_gripper_velocity_mm_s,
    )
    stage_started = time.perf_counter()
    tick = 0
    while True:
        now = time.perf_counter()
        if now > deadline:
            raise RuntimeError(f"{phase} initial-pose homing timed out during gripper restore")
        raw = robot.sdk.get_status()
        observed = raw_feedback_pose(raw)
        driver_state = observation_to_driver_state(raw, joint_limits, gripper_stroke_mm)
        verify_homing_health(robot, check_controller=False)
        elapsed = now - stage_started
        blend = quintic_blend(elapsed / gripper_duration)
        target_gripper = grip_start + (goal.gripper_mm - grip_start) * blend
        command = physical_pose_to_driver_action(
            goal.joints_deg,
            target_gripper,
            joint_limits,
            gripper_stroke_mm,
        )
        sent = robot.send_action(
            command,
            present_position=normalized_present_position(driver_state),
        )
        joint_error = max(
            abs(value - target)
            for value, target in zip(observed.joints_deg, goal.joints_deg, strict=True)
        )
        gripper_error = abs(observed.gripper_mm - goal.gripper_mm)
        if tick % 30 == 0:
            log.write(
                {
                    "event": "initial_pose_home_progress",
                    "phase": phase,
                    "stage": "gripper",
                    "elapsed_s": now - started,
                    "max_joint_error_deg": joint_error,
                    "observed_gripper_mm": observed.gripper_mm,
                    "gripper_error_mm": gripper_error,
                    "proposed_action": command,
                    "sent_action": sent,
                    "command_sent": True,
                }
            )
        if (
            elapsed >= gripper_duration
            and joint_error <= args.home_joint_tolerance_deg
            and gripper_error <= args.home_gripper_tolerance_mm
        ):
            break
        tick += 1
        precise_sleep(1.0 / 30.0 - (time.perf_counter() - now))

    hold_started = time.perf_counter()
    goal_action = physical_pose_to_driver_action(
        goal.joints_deg,
        goal.gripper_mm,
        joint_limits,
        gripper_stroke_mm,
    )
    while time.perf_counter() - hold_started < args.home_hold_s:
        tick_started = time.perf_counter()
        raw = robot.sdk.get_status()
        observed = raw_feedback_pose(raw)
        driver_state = observation_to_driver_state(raw, joint_limits, gripper_stroke_mm)
        verify_homing_health(robot, check_controller=False)
        robot.send_action(
            goal_action,
            present_position=normalized_present_position(driver_state),
        )
        precise_sleep(1.0 / 30.0 - (time.perf_counter() - tick_started))

    final_raw = robot.sdk.get_status()
    final_pose = raw_feedback_pose(final_raw)
    final_joint_error = max(
        abs(value - target)
        for value, target in zip(final_pose.joints_deg, goal.joints_deg, strict=True)
    )
    final_gripper_error = abs(final_pose.gripper_mm - goal.gripper_mm)
    wait_for_stable_homing_health(robot)
    if (
        final_joint_error > args.home_joint_tolerance_deg
        or final_gripper_error > args.home_gripper_tolerance_mm
    ):
        raise RuntimeError(
            f"{phase} initial-pose hold drifted outside tolerance: "
            f"joint={final_joint_error:.3f} deg gripper={final_gripper_error:.3f} mm"
        )
    log.write(
        {
            "event": "initial_pose_home_complete",
            "phase": phase,
            "elapsed_s": time.perf_counter() - started,
            "final_joints_deg": list(final_pose.joints_deg),
            "final_gripper_mm": final_pose.gripper_mm,
            "max_joint_error_deg": final_joint_error,
            "gripper_error_mm": final_gripper_error,
            "already_within_tolerance": False,
            "command_sent": True,
        }
    )


class IndependentGripperStateMachine:
    """Latch binary gripper intent independently from the arm action queue.

    The policy may change chunks faster than the physical gripper can travel.
    This controller therefore debounces model intent, latches the selected
    open/close target, and keeps re-sending it while arm actions continue to
    advance.  Feedback is used for reporting completion, never to silently
    reverse the latched target.
    """

    def __init__(
        self,
        *,
        mode: str,
        close_confirm_frames: int,
        open_confirm_frames: int,
        min_close_hold_frames: int,
        close_feedback_threshold: float,
        open_feedback_threshold: float,
        terminal_open_after_release: bool = True,
    ) -> None:
        if mode not in ("direct", "latched"):
            raise ValueError(f"unsupported gripper control mode: {mode}")
        self.control_mode = mode
        self.close_confirm_frames = int(close_confirm_frames)
        self.open_confirm_frames = int(open_confirm_frames)
        self.min_close_hold_frames = int(min_close_hold_frames)
        self.close_feedback_threshold = float(close_feedback_threshold)
        self.open_feedback_threshold = float(open_feedback_threshold)
        self.terminal_open_after_release = bool(terminal_open_after_release)
        self.state = "hold"
        self.state_age_frames = 0
        self.close_request_streak = 0
        self.open_request_streak = 0
        self.has_closed_once = False
        self.release_latched = False
        self.prediction_tick = 0
        self._prediction_horizons: list[tuple[int, int, np.ndarray]] = []

    @staticmethod
    def _validate_prediction_horizon(model_commands_pct: Any) -> np.ndarray:
        commands = np.asarray(model_commands_pct, dtype=np.float64).reshape(-1)
        if commands.size != 30 or not np.isfinite(commands).all():
            raise RuntimeError(
                "full gripper prediction must contain exactly 30 finite commands, "
                f"got shape={commands.shape}"
            )
        if not np.all(np.isin(commands, (0.0, 100.0))):
            raise RuntimeError(
                "full gripper prediction must be binary 0/100, "
                f"got {commands.tolist()}"
            )
        return commands

    def observe_prediction_horizon(
        self, chunk_id: int, model_commands_pct: Any
    ) -> dict[str, Any]:
        """Add one complete 30-tick gripper forecast on an absolute time axis.

        Arm execution may use only the first ``query_frequency`` actions before
        replanning.  Keeping the full gripper forecast here prevents a close
        request at ticks 10--29 from being discarded at every arm replan.
        """
        commands = self._validate_prediction_horizon(model_commands_pct)
        chunk = int(chunk_id)
        if any(existing_chunk == chunk for existing_chunk, _, _ in self._prediction_horizons):
            raise RuntimeError(f"duplicate gripper prediction horizon for chunk {chunk}")
        self._prediction_horizons.append(
            (chunk, int(self.prediction_tick), commands.copy())
        )
        close_offsets = np.flatnonzero(commands == 0.0)
        return {
            "chunk_id": chunk,
            "absolute_start_tick": int(self.prediction_tick),
            "horizon_steps": int(commands.size),
            "close_steps": int(close_offsets.size),
            "first_close_offset": (
                int(close_offsets[0]) if close_offsets.size else None
            ),
            "last_close_offset": (
                int(close_offsets[-1]) if close_offsets.size else None
            ),
            "commands_pct": commands.tolist(),
        }

    def _horizon_command(self, fallback_model_command_pct: float) -> tuple[float, dict[str, Any]]:
        active_votes: list[tuple[int, int, float]] = []
        retained: list[tuple[int, int, np.ndarray]] = []
        for chunk_id, start_tick, commands in self._prediction_horizons:
            offset = int(self.prediction_tick - start_tick)
            if offset < len(commands):
                retained.append((chunk_id, start_tick, commands))
            if 0 <= offset < len(commands):
                active_votes.append((chunk_id, offset, float(commands[offset])))
        self._prediction_horizons = retained
        if active_votes:
            # Execute the oldest forecast that still covers this absolute tick.
            # A newer arm replan must not overwrite either the close phase or
            # the later reopen phase of an already-committed gripper schedule.
            selected_chunk_id, selected_offset, command = active_votes[0]
            source = "oldest_active_full_30_step_horizon"
        else:
            command = float(fallback_model_command_pct)
            source = "selected_action_fallback"
            selected_chunk_id = None
            selected_offset = None
        return command, {
            "prediction_source": source,
            "prediction_tick": int(self.prediction_tick),
            "active_prediction_count": int(len(active_votes)),
            "selected_prediction_chunk_id": selected_chunk_id,
            "selected_prediction_offset": selected_offset,
            "horizon_votes": [
                {"chunk_id": chunk_id, "offset": offset, "command_pct": command_pct}
                for chunk_id, offset, command_pct in active_votes
            ],
        }

    def update_from_prediction_horizons(
        self,
        fallback_model_command_pct: float,
        feedback_pct: float,
    ) -> dict[str, Any]:
        model_command_pct, horizon_report = self._horizon_command(
            fallback_model_command_pct
        )
        report = self.update(model_command_pct, feedback_pct)
        report.update(horizon_report)
        return report

    def advance_prediction_tick(self) -> None:
        self.prediction_tick += 1
        self._prediction_horizons = [
            plan
            for plan in self._prediction_horizons
            if plan[1] + len(plan[2]) > self.prediction_tick
        ]

    @staticmethod
    def _request_name(model_command_pct: float) -> str:
        command = float(model_command_pct)
        if abs(command) <= 1e-6:
            return "close"
        if abs(command - 100.0) <= 1e-6:
            return "open"
        raise RuntimeError(f"gripper model request must be binary 0/100, got {command}")

    def update(self, model_command_pct: float, feedback_pct: float) -> dict[str, Any]:
        feedback = float(np.clip(feedback_pct, 0.0, 100.0))
        model_request = self._request_name(model_command_pct)
        close_request_ignored_after_release = bool(
            self.release_latched and model_request == "close"
        )
        request = "open" if self.release_latched else model_request
        transition = None

        if request == "close":
            self.close_request_streak += 1
            self.open_request_streak = 0
        else:
            self.open_request_streak += 1
            self.close_request_streak = 0

        if self.control_mode == "direct":
            previous_state = self.state
            self.state = request
            self.state_age_frames = 0
            if self.state == "close":
                self.has_closed_once = True
            if (
                self.terminal_open_after_release
                and self.has_closed_once
                and previous_state == "close"
                and self.state == "open"
            ):
                self.release_latched = True
            command = 0.0 if request == "close" else 100.0
        else:
            previous_state = self.state
            if self.state == "hold":
                if self.close_request_streak >= self.close_confirm_frames:
                    self.state = "close"
                elif self.open_request_streak >= self.open_confirm_frames:
                    self.state = "open"
            elif self.state == "open":
                if self.close_request_streak >= self.close_confirm_frames:
                    self.state = "close"
            elif self.state == "close":
                may_release = self.state_age_frames >= self.min_close_hold_frames
                if may_release and self.open_request_streak >= self.open_confirm_frames:
                    self.state = "open"
            else:  # pragma: no cover - protects serialized/runtime corruption.
                raise RuntimeError(f"invalid gripper state: {self.state}")

            if self.state != previous_state:
                transition = f"{previous_state}->{self.state}"
                self.state_age_frames = 0
                if self.state == "close":
                    self.has_closed_once = True
                if (
                    self.terminal_open_after_release
                    and self.has_closed_once
                    and previous_state == "close"
                    and self.state == "open"
                ):
                    self.release_latched = True
            else:
                self.state_age_frames += 1

            if self.state == "hold":
                # Do not move during debounce. Holding measured aperture also
                # makes startup independent of whether the gripper began open,
                # closed, or stopped on an object.
                command = feedback
            else:
                command = 0.0 if self.state == "close" else 100.0

        close_reached = feedback <= self.close_feedback_threshold
        open_reached = feedback >= self.open_feedback_threshold
        target_reached = (
            close_reached
            if self.state == "close"
            else open_reached
            if self.state == "open"
            else True
        )
        return {
            "control_mode": self.control_mode,
            "state": self.state,
            "state_age_frames": int(self.state_age_frames),
            "transition": transition,
            "model_request": model_request,
            "effective_request": request,
            "model_command_pct": float(model_command_pct),
            "command_pct": float(command),
            "feedback_pct": feedback,
            "close_request_streak": int(self.close_request_streak),
            "open_request_streak": int(self.open_request_streak),
            "target_reached": bool(target_reached),
            "close_reached": bool(close_reached),
            "open_reached": bool(open_reached),
            "terminal_open_after_release": self.terminal_open_after_release,
            "has_closed_once": self.has_closed_once,
            "release_latched": self.release_latched,
            "close_request_ignored_after_release": close_request_ignored_after_release,
        }


def observation_to_joint_rad(observation: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(
        [observation[f"joint_{index}.pos"] for index in range(6)], dtype=np.float64
    )
    return raw * 1e-3 * np.pi / 180.0


def full_gripper_horizon_from_report(report: dict[str, Any] | None) -> np.ndarray:
    if report is None or "full_control_gripper_commands_pct" not in report:
        raise RuntimeError(
            "joint policy report lacks the complete 30-step gripper prediction"
        )
    return IndependentGripperStateMachine._validate_prediction_horizon(
        report["full_control_gripper_commands_pct"]
    )


def initial_continuity_report(
    observation: dict[str, Any],
    executable_action: torch.Tensor,
    joint_limits: list[tuple[float, float]],
    max_delta_deg: float,
) -> dict[str, Any]:
    command = executable_action.detach().cpu().numpy().reshape(-1)
    if command.shape != (7,) or not np.isfinite(command).all():
        raise RuntimeError(f"invalid executable warmup action: {command}")
    current = observation_to_joint_rad(observation)
    target_deg = np.asarray(
        [
            low + (float(value) + 100.0) / 200.0 * (high - low)
            for value, (low, high) in zip(command[:6], joint_limits, strict=True)
        ],
        dtype=np.float64,
    )
    target = np.radians(target_deg)
    delta_deg = target_deg - np.degrees(current)
    report = {
        "current_joint_rad": current.tolist(),
        "first_target_joint_rad": target.tolist(),
        "first_target_driver": command[:6].tolist(),
        "joint_delta_deg": delta_deg.tolist(),
        "max_abs_joint_delta_deg": float(np.max(np.abs(delta_deg))),
        "limit_deg": float(max_delta_deg),
    }
    if max_delta_deg > 0 and report["max_abs_joint_delta_deg"] > max_delta_deg:
        raise RuntimeError(
            "initial joint target continuity check failed before motion: "
            f"max_abs_delta={report['max_abs_joint_delta_deg']:.3f}deg exceeds "
            f"limit={max_delta_deg:.3f}deg"
        )
    return report


def configure_front_camera_auto_controls(robot: Piper) -> dict[str, float]:
    import pyrealsense2 as rs

    front_camera = robot.cameras.get("435if")
    if front_camera is None or front_camera.rs_profile is None:
        raise RuntimeError("front RealSense camera is not connected")
    color_sensor = front_camera.rs_profile.get_device().first_color_sensor()
    options = {
        "auto_exposure": rs.option.enable_auto_exposure,
        "exposure": rs.option.exposure,
        "gain": rs.option.gain,
        "auto_white_balance": rs.option.enable_auto_white_balance,
        "white_balance": rs.option.white_balance,
    }
    unsupported = [name for name, option in options.items() if not color_sensor.supports(option)]
    if unsupported:
        raise RuntimeError(f"front camera lacks controls: {unsupported}")
    color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
    color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0)
    deadline = time.monotonic() + FRONT_CAMERA_AUTO_WARMUP_S
    while time.monotonic() < deadline:
        front_camera.read(timeout_ms=1000)
    report = {name: float(color_sensor.get_option(option)) for name, option in options.items()}
    if report["auto_exposure"] != 1.0 or report["auto_white_balance"] != 1.0:
        raise RuntimeError(f"front automatic controls did not remain enabled: {report}")
    return report


class JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("x")

    def write(self, record: dict[str, Any]) -> None:
        self._stream.write(json.dumps(record, allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def save_inference_images(
    log_dir: Path,
    run_timestamp: str,
    chunk_id: int,
    front: np.ndarray,
    wrist: np.ndarray,
    side: np.ndarray,
) -> dict[str, str]:
    from PIL import Image

    image_dir = log_dir / f"images_{run_timestamp}"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, array in (("front", front), ("wrist", wrist), ("side", side)):
        image = np.asarray(array)
        if image.ndim == 3 and image.shape[0] in (3, 4):
            image = np.moveaxis(image[:3], 0, -1)
        image = np.clip(image, 0, 255).astype(np.uint8)
        path = image_dir / f"chunk_{chunk_id:05d}_{name}.png"
        Image.fromarray(image[..., :3], mode="RGB").save(path)
        paths[name] = str(path)
    return paths


def run_offline(
    args: argparse.Namespace,
    policy: EgoVerseHPTJointPolicy,
    log: JsonlLogger,
    device: torch.device,
) -> None:
    if args.offline_npz is None:
        raise ValueError("offline mode requires --offline-npz")
    keys = (
        "observation.state",
        "observation.images.435if",
        "observation.images.405",
        "observation.images.side",
    )
    with np.load(args.offline_npz, allow_pickle=False) as archive:
        missing = [key for key in keys if key not in archive]
        if missing:
            raise KeyError(f"offline NPZ is missing {missing}; required={keys}")
        arrays = {key: archive[key] for key in keys}
    frame_count = min(args.max_steps, *(len(arrays[key]) for key in keys))
    last_reported_chunk = -1
    for frame_index in range(frame_count):
        started = time.perf_counter()
        batch = make_policy_batch(
            arrays[keys[0]][frame_index],
            arrays[keys[1]][frame_index],
            arrays[keys[2]][frame_index],
            arrays[keys[3]][frame_index],
            device,
        )
        action = policy.select_action(batch)
        status = policy.queue_status()
        new_report = None
        if status["chunk_id"] != last_reported_chunk:
            new_report = policy.last_prediction_report
            last_reported_chunk = status["chunk_id"]
        log.write(
            {
                "mode": "offline",
                "frame": frame_index,
                "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                "total_step_ms": (time.perf_counter() - started) * 1e3,
                "action": action.tolist(),
                "queue_status": status,
                "temporal_ensemble_report": policy.last_ensemble_report,
                "new_chunk_report": new_report,
                "command_sent": False,
            }
        )


def make_robot(
    args: argparse.Namespace,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> Piper:
    if args.can_port is None:
        raise ValueError("live modes require an explicitly verified --can-port")
    cameras = {
        "435if": RealSenseCameraConfig(
            serial_number_or_name=args.front_camera_serial,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        ),
        "405": RealSenseCameraConfig(
            serial_number_or_name=args.wrist_camera_serial,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        ),
        "side": RealSenseCameraConfig(
            serial_number_or_name=args.side_camera_serial,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        ),
    }
    return Piper(
        PiperConfig(
            # Do not inherit the unrelated historical my_piper_follower cache.
            # bind_runtime_calibration() below still enforces and verifies the
            # bundle calibration, even if a cache is later created for this id.
            id="assembly_new1_piper_follower",
            port=args.can_port,
            cameras=cameras,
            joint_limits_deg=joint_limits,
            gripper_stroke_mm=gripper_stroke_mm,
            speed_percent=args.speed_percent,
            max_relative_target=split_relative_target_limits(
                args.max_relative_target,
                args.max_gripper_relative_target,
            ),
            enable_motion_on_connect=False,
            home_on_disconnect=False,
            disable_torque_on_disconnect=False,
        )
    )


def split_relative_target_limits(
    arm_limit: float, gripper_limit: float
) -> dict[str, float]:
    """Keep the arm safety cap independent from the gripper aperture cap."""
    return {
        **{f"joint_{index}": float(arm_limit) for index in range(6)},
        "joint_6": float(gripper_limit),
    }


def run_live(
    args: argparse.Namespace,
    policy: EgoVerseHPTJointPolicy,
    log: JsonlLogger,
    device: torch.device,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
    adapter: JointSpaceAdapter,
    run_timestamp: str,
) -> None:
    execute = args.mode == "execute"
    if execute and not args.allow_motion:
        raise RuntimeError("execute mode requires --allow-motion")

    initial_pose = load_hardware_initial_pose(
        args.initial_pose_json,
        adapter,
    )
    firmware_initial_pose = apply_j2_command_offset_to_initial_pose(
        initial_pose,
        args.j2_command_offset_deg,
        list(adapter.hardware_limits_deg),
    )
    firmware_initial_pose = apply_j3_command_offset_to_initial_pose(
        firmware_initial_pose,
        args.j3_command_offset_deg,
        list(adapter.hardware_limits_deg),
    )
    firmware_initial_pose = apply_j5_command_offset_to_initial_pose(
        firmware_initial_pose,
        args.j5_command_offset_deg,
        list(adapter.hardware_limits_deg),
    )
    robot = make_robot(
        args,
        list(adapter.hardware_limits_deg),
        adapter.hardware_gripper_stroke_mm,
    )
    robot.connect(calibrate=False)
    motion_promoted = False
    gripper_controller = IndependentGripperStateMachine(
        mode=args.gripper_control_mode,
        close_confirm_frames=args.gripper_close_confirm_frames,
        open_confirm_frames=args.gripper_open_confirm_frames,
        min_close_hold_frames=args.gripper_min_close_hold_frames,
        close_feedback_threshold=args.gripper_close_feedback_threshold,
        open_feedback_threshold=args.gripper_open_feedback_threshold,
        terminal_open_after_release=args.gripper_terminal_open_after_release,
    )
    last_reported_chunk = -1
    try:
        calibration_report = bind_runtime_calibration(
            robot,
            robot.sdk.get_status(),
            list(adapter.hardware_limits_deg),
            adapter.hardware_gripper_stroke_mm,
        )
        log.write(
            {
                "event": "runtime_calibration_bound",
                **calibration_report,
                "joint_space_adapter": adapter.report(),
                "command_sent": False,
            }
        )
        camera_report = configure_front_camera_auto_controls(robot)
        log.write(
            {
                "event": "camera_config",
                "camera": "435if",
                **camera_report,
                "command_sent": False,
            }
        )

        # Verify all cameras and live joint feedback before adopting the
        # all-enabled state left by model-free Home.
        # Execute mode then reproduces the recorder's follower-only initial-pose
        # return before the policy sees its first observation.
        preflight_observation = robot.get_observation()
        require_live_robot_feedback(
            robot,
            preflight_observation,
            context="deployment preflight",
        )
        preflight_home_start, preflight_home_report = validate_home_start_feedback(
            preflight_observation,
            args,
            list(adapter.hardware_limits_deg),
            adapter.hardware_gripper_stroke_mm,
        )
        require_deployment_pose_inside_limits(
            preflight_home_report,
            execute=execute,
            home_at_start=args.home_at_start,
        )
        preflight_joint_duration = quintic_joint_duration(
            preflight_home_start.joints_deg,
            firmware_initial_pose.joints_deg,
            args.home_velocity_deg_s,
            args.home_acceleration_deg_s2,
        )
        log.write(
            {
                "event": "live_preflight",
                "front_shape": list(np.asarray(preflight_observation["435if"]).shape),
                "wrist_shape": list(np.asarray(preflight_observation["405"]).shape),
                "raw_state": [
                    float(preflight_observation[f"joint_{index}.pos"])
                    for index in range(7)
                ],
                "initial_pose_json": str(args.initial_pose_json),
                "initial_pose_joints_deg": list(initial_pose.joints_deg),
                "firmware_initial_pose_joints_deg": list(
                    firmware_initial_pose.joints_deg
                ),
                "j2_command_offset_deg": args.j2_command_offset_deg,
                "j3_command_offset_deg": args.j3_command_offset_deg,
                "j5_command_offset_deg": args.j5_command_offset_deg,
                "initial_pose_gripper_mm": initial_pose.gripper_mm,
                "home_start_feedback_validation": preflight_home_report,
                "planned_joint_home_duration_s": preflight_joint_duration,
                "command_sent": False,
            }
        )
        if execute:
            held_position = adopt_already_enabled_motion_hold_current(robot)
            motion_promoted = True
            log.write(
                {
                    "event": "motion_enable",
                    "promotion_mode": "adopt_already_enabled_no_quick_stop",
                    "held_position": held_position,
                    "command_sent": True,
                }
            )
            if args.hold_current_only_seconds > 0:
                wait_for_stable_homing_health(robot)
                hold_started = time.perf_counter()
                while time.perf_counter() - hold_started < args.hold_current_only_seconds:
                    tick_started = time.perf_counter()
                    verify_homing_health(robot, check_controller=False)
                    latch_current_raw_position(robot)
                    precise_sleep(1.0 / 30.0 - (time.perf_counter() - tick_started))
                wait_for_stable_homing_health(robot)
                log.write(
                    {
                        "event": "hold_current_only_complete",
                        "duration_s": time.perf_counter() - hold_started,
                        "held_position": held_position,
                        "command_sent": True,
                    }
                )
                latch_current_pose_without_disabling_torque(
                    robot, log, reason="hold_current_only_complete"
                )
                return
            if args.home_at_start:
                run_initial_pose_home(
                    robot=robot,
                    goal=initial_pose,
                    args=args,
                    log=log,
                    joint_limits=list(adapter.hardware_limits_deg),
                    gripper_stroke_mm=adapter.hardware_gripper_stroke_mm,
                    phase="before_policy",
                )
            if args.home_only:
                log.write(
                    {
                        "event": "home_only_complete",
                        "command_sent": True,
                    }
                )
                latch_current_pose_without_disabling_torque(
                    robot, log, reason="home_only_complete"
                )
                return

        warmup_observation = robot.get_observation()
        warmup_model_input_observation = hardware_observation_for_model(
            warmup_observation,
            args.j2_command_offset_deg,
            args.j3_command_offset_deg,
            args.j5_command_offset_deg,
        )
        warmup_started = time.perf_counter()
        # Populate the queue without consuming it.  The continuity gate below
        # must validate the exact first action that execute mode will send;
        # re-sampling a diffusion chunk after warmup would invalidate that
        # guarantee even when the observation is unchanged.
        warmup_action = policy.peek_action(
            live_observation_to_batch(
                warmup_model_input_observation, device, adapter
            )
        )
        warmup_model_observation = adapter.hardware_observation_to_model_observation(
            warmup_model_input_observation
        )
        continuity = initial_continuity_report(
            warmup_model_observation,
            warmup_action,
            joint_limits,
            args.max_initial_joint_delta_deg,
        )
        log.write(
            {
                "event": "warmup",
                "mode": args.mode,
                "total_step_ms": (time.perf_counter() - warmup_started) * 1e3,
                "discarded_action": warmup_action.tolist(),
                "initial_continuity": continuity,
                "chunk_report": policy.last_prediction_report,
                "temporal_ensemble_report": policy.last_ensemble_report,
                "command_sent": False,
            }
        )
        started_run = time.perf_counter()
        for frame_index in range(args.max_steps):
            loop_started = time.perf_counter()
            observation = robot.get_observation()
            model_input_observation = hardware_observation_for_model(
                observation,
                args.j2_command_offset_deg,
                args.j3_command_offset_deg,
                args.j5_command_offset_deg,
            )
            batch = live_observation_to_batch(
                model_input_observation, device, adapter
            )
            action = policy.peek_action(batch)
            queue_status_before = policy.queue_status()
            new_report = None
            image_paths = None
            gripper_horizon_update = None
            if queue_status_before["chunk_id"] != last_reported_chunk:
                new_report = policy.last_prediction_report
                last_reported_chunk = queue_status_before["chunk_id"]
                if args.gripper_prediction_source == "full-horizon":
                    gripper_horizon_update = gripper_controller.observe_prediction_horizon(
                        queue_status_before["chunk_id"],
                        full_gripper_horizon_from_report(new_report),
                    )
                if args.save_inference_images:
                    image_paths = save_inference_images(
                        args.log_dir,
                        run_timestamp,
                        queue_status_before["chunk_id"],
                        observation["435if"],
                        observation["405"],
                        observation["side"],
                    )

            model_action = action_to_robot_dict(action)
            driver_state = observation_to_driver_state(
                observation,
                list(adapter.hardware_limits_deg),
                adapter.hardware_gripper_stroke_mm,
            )
            if args.gripper_prediction_source == "full-horizon":
                gripper_state = gripper_controller.update_from_prediction_horizons(
                    model_action["joint_6.pos"], driver_state[6]
                )
            else:
                gripper_state = gripper_controller.update(
                    model_action["joint_6.pos"], driver_state[6]
                )
            unshifted_robot_action = (
                adapter.model_driver_action_to_hardware_action(model_action)
            )
            robot_action, j2_command_mapping = (
                apply_j2_command_offset_to_hardware_action(
                    unshifted_robot_action,
                    args.j2_command_offset_deg,
                    list(adapter.hardware_limits_deg),
                )
            )
            robot_action, j3_command_mapping = (
                apply_j3_command_offset_to_hardware_action(
                    robot_action,
                    args.j3_command_offset_deg,
                    list(adapter.hardware_limits_deg),
                )
            )
            robot_action, j5_command_mapping = (
                apply_j5_command_offset_to_hardware_action(
                    robot_action,
                    args.j5_command_offset_deg,
                    list(adapter.hardware_limits_deg),
                )
            )
            robot_action["joint_6.pos"] = gripper_state["command_pct"]
            if execute:
                verify_homing_health(
                    robot,
                    context="deployment control loop",
                )
                sent_action = robot.send_action(
                    robot_action,
                    present_position=normalized_present_position(driver_state),
                )
            else:
                sent_action = None
            arm_command_was_clipped = None
            gripper_command_was_clipped = None
            if sent_action is not None:
                arm_command_was_clipped = any(
                    abs(robot_action[f"joint_{index}.pos"] - sent_action[f"joint_{index}.pos"])
                    > 1e-6
                    for index in range(6)
                )
                gripper_command_was_clipped = (
                    abs(robot_action["joint_6.pos"] - sent_action["joint_6.pos"]) > 1e-6
                )
            command_was_clipped = (
                bool(arm_command_was_clipped or gripper_command_was_clipped)
                if sent_action is not None
                else None
            )
            # Dry-run advances on the control clock.  Execute advances only
            # after the arm command survives the driver's relative-target
            # clamp.  A clipped target is held and retried against fresh
            # feedback instead of silently skipping to a different action.
            action_advanced = not execute or not arm_command_was_clipped
            if action_advanced:
                policy.advance_action()
                gripper_controller.advance_prediction_tick()
            queue_status = policy.queue_status()
            log.write(
                {
                    "mode": args.mode,
                    "frame": frame_index,
                    "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": time.perf_counter() - started_run,
                    "total_step_ms": (time.perf_counter() - loop_started) * 1e3,
                    "front_shape": list(np.asarray(observation["435if"]).shape),
                    "wrist_shape": list(np.asarray(observation["405"]).shape),
                    "raw_state": [
                        float(observation[f"joint_{index}.pos"]) for index in range(7)
                    ],
                    "model_input_state": [
                        float(model_input_observation[f"joint_{index}.pos"])
                        for index in range(7)
                    ],
                    "driver_state": driver_state.tolist(),
                    "model_action": model_action,
                    "unshifted_hardware_action": unshifted_robot_action,
                    "j2_command_mapping": j2_command_mapping,
                    "j3_command_mapping": j3_command_mapping,
                    "j5_command_mapping": j5_command_mapping,
                    "proposed_action": robot_action,
                    "sent_action": sent_action,
                    "command_was_clipped": command_was_clipped,
                    "arm_command_was_clipped": arm_command_was_clipped,
                    "gripper_command_was_clipped": gripper_command_was_clipped,
                    "gripper_state_machine": gripper_state,
                    "gripper_horizon_update": gripper_horizon_update,
                    "action_advanced": action_advanced,
                    "queue_status": queue_status,
                    "temporal_ensemble_report": policy.last_ensemble_report,
                    "new_chunk_report": new_report,
                    "inference_images": image_paths,
                    "command_sent": execute,
                }
            )
            precise_sleep(1.0 / 30.0 - (time.perf_counter() - loop_started))

        # Only a normally completed rollout returns home. Any exception,
        # hardware fault, Ctrl+C, or emergency stop skips autonomous recovery
        # so the original failure cannot trigger an additional motion.
        if execute and args.home_at_end:
            run_initial_pose_home(
                robot=robot,
                goal=initial_pose,
                args=args,
                log=log,
                joint_limits=list(adapter.hardware_limits_deg),
                gripper_stroke_mm=adapter.hardware_gripper_stroke_mm,
                phase="after_policy",
            )
        if execute:
            latch_current_pose_without_disabling_torque(
                robot, log, reason="normal_completion"
            )
    except BaseException:
        if execute and motion_promoted:
            try:
                latch_current_pose_without_disabling_torque(
                    robot, log, reason="exception_abort"
                )
            except BaseException as hold_error:
                logger.error("failed to latch current pose after abort: %s", hold_error)
        raise
    finally:
        robot.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("checkpoint", "offline", "live-dry-run", "execute"), default="checkpoint"
    )
    parser.add_argument(
        "--checkpoint-profile",
        choices=tuple(CHECKPOINT_PROFILES),
        help=(
            "select an audited checkpoint and its SHA-256 by name; mutually "
            "exclusive with explicit --checkpoint/--checkpoint-sha256"
        ),
    )
    parser.add_argument(
        "--list-checkpoint-profiles",
        action="store_true",
        help="print available checkpoint profiles and exit without loading a model",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="recorded follower/model-space calibration",
    )
    parser.add_argument(
        "--hardware-calibration-json",
        type=Path,
        default=DEFAULT_HARDWARE_CALIBRATION,
        help="original leader calibration used by the physical deployment arm",
    )
    parser.add_argument(
        "--role-mapping-json",
        type=Path,
        default=DEFAULT_ROLE_MAPPING,
        help="collection metadata containing the paired leader/follower mapping",
    )
    parser.add_argument(
        "--joint-mapping-mode",
        choices=("physical-identity", "recorded-percentile-affine"),
        default="physical-identity",
        help=(
            "model/hardware joint-space conversion; the stable default keeps "
            "physical angles, while recorded-percentile-affine reconstructs "
            "collection-time leader commands and applies the exact inverse to proprio"
        ),
    )
    parser.add_argument(
        "--joint-zero-offset-json",
        type=Path,
        default=DEFAULT_JOINT_ZERO_OFFSET,
        help=(
            "audited full-trajectory hardware-minus-model joint zero offset; "
            "non-zero values require approved multi-pose validation"
        ),
    )
    parser.add_argument(
        "--allow-experimental-joint-mapping",
        action="store_true",
        help=(
            "explicitly allow either a joint-zero-offset file marked "
            "experimental_unvalidated or the rejected recorded-percentile-affine "
            "mapping in execute mode; never enabled by the stable launcher"
        ),
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--initial-pose-json",
        type=Path,
        default=DEFAULT_INITIAL_POSE,
        help="follower physical camera Home replayed on the original-leader hardware",
    )
    parser.add_argument(
        "--home-at-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="in execute mode, return to the recorded data-collection pose before policy inference",
    )
    parser.add_argument(
        "--home-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after normal execute completion, return to the recorded data-collection pose",
    )
    parser.add_argument(
        "--home-only",
        action="store_true",
        help="execute the guarded start homing sequence and exit without sending any policy action",
    )
    parser.add_argument(
        "--hold-current-only-seconds",
        type=float,
        default=0.0,
        help=(
            "adopt the already-enabled Home state and hold the measured pose "
            "for N seconds, then latch it and exit without policy actions"
        ),
    )
    parser.add_argument("--home-timeout-s", type=float, default=45.0)
    parser.add_argument("--home-velocity-deg-s", type=float, default=10.0)
    parser.add_argument("--home-acceleration-deg-s2", type=float, default=20.0)
    parser.add_argument("--home-gripper-velocity-mm-s", type=float, default=15.0)
    parser.add_argument("--home-joint-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--home-gripper-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--home-start-limit-margin-deg", type=float, default=10.0)
    parser.add_argument("--home-hold-s", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-frequency", type=int, choices=range(1, 31), default=10)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument(
        "--allow-legacy-joint-head",
        action="store_true",
        help=(
            "allow a pre-residual joint Flow-Matching checkpoint; disabled by "
            "default so formal deployment only accepts ResidualJointPolicy"
        ),
    )
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument(
        "--gripper-prediction-source",
        choices=("full-horizon", "selected-action"),
        default="full-horizon",
        help=(
            "full-horizon keeps every 30-step gripper forecast on an absolute "
            "time axis; selected-action preserves the legacy qf-sampled input"
        ),
    )
    parser.add_argument(
        "--gripper-control-mode",
        choices=("direct", "latched"),
        default="latched",
        help="direct preserves legacy per-frame switching; latched uses feedback-aware hysteresis",
    )
    parser.add_argument("--gripper-close-confirm-frames", type=int, default=3)
    parser.add_argument("--gripper-open-confirm-frames", type=int, default=15)
    parser.add_argument("--gripper-min-close-hold-frames", type=int, default=60)
    parser.add_argument("--gripper-close-feedback-threshold", type=float, default=10.0)
    parser.add_argument("--gripper-open-feedback-threshold", type=float, default=90.0)
    parser.add_argument(
        "--gripper-terminal-open-after-release",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "after a confirmed close->open release, permanently command open "
            "and log but ignore later close predictions for this rollout"
        ),
    )
    parser.add_argument("--joint-limit-margin-deg", type=float, default=0.0)
    parser.add_argument("--max-joint-command-step", type=float, default=0.50)
    parser.add_argument("--max-joint-command-accel", type=float, default=0.10)
    parser.add_argument("--temporal-ensemble", action="store_true")
    parser.add_argument("--temporal-replan-interval", type=int, default=5)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.10)
    parser.add_argument("--chunk-smoothing", action="store_true")
    parser.add_argument("--chunk-smoothing-window", type=int, default=11)
    parser.add_argument("--chunk-smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--offline-npz", type=Path)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--can-port")
    parser.add_argument(
        "--front-camera-serial",
        default=FRONT_SERIAL,
        help="RealSense serial used for the global/front RGB camera",
    )
    parser.add_argument(
        "--wrist-camera-serial",
        default=WRIST_SERIAL,
        help="RealSense serial used for the wrist RGB camera",
    )
    parser.add_argument(
        "--side-camera-serial",
        default=SIDE_SERIAL,
        help="RealSense serial used for the fixed side RGB camera",
    )
    parser.add_argument(
        "--model-action-horizon",
        type=int,
        choices=(30, 100),
        default=100,
        help="custom checkpoint output horizon; audited profiles set it automatically",
    )
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--max-relative-target", type=float, default=0.5)
    parser.add_argument(
        "--max-gripper-relative-target",
        type=float,
        default=100.0,
        help=(
            "maximum per-command gripper aperture change in normalized percent; "
            "100 sends binary 0/100 targets without deployment-side clipping"
        ),
    )
    parser.add_argument("--max-initial-joint-delta-deg", type=float, default=20.0)
    parser.add_argument(
        "--j2-command-offset-deg",
        type=float,
        default=0.0,
        help=(
            "hardware-minus-model J2 zero offset in degrees: model proprio "
            "uses firmware J2 minus this value, while model actions and the "
            "saved Home J2 are sent to firmware plus this value"
        ),
    )
    parser.add_argument(
        "--j3-command-offset-deg",
        type=float,
        default=0.0,
        help=(
            "hardware-minus-model J3 zero offset in degrees: model proprio "
            "uses firmware J3 minus this value, while model actions and the "
            "saved Home J3 are sent to firmware plus this value"
        ),
    )
    parser.add_argument(
        "--j5-command-offset-deg",
        type=float,
        default=0.0,
        help=(
            "hardware-minus-model J5 zero offset in degrees: model proprio "
            "uses firmware J5 minus this value, while model actions and the "
            "saved Home J5 are sent to firmware plus this value"
        ),
    )
    parser.add_argument("--save-inference-images", action="store_true")
    parser.add_argument("--allow-motion", action="store_true")
    args = parser.parse_args()
    args = resolve_checkpoint_selection(parser, args)
    if args.list_checkpoint_profiles:
        return args
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    camera_serials = (
        str(args.front_camera_serial).strip(),
        str(args.wrist_camera_serial).strip(),
        str(args.side_camera_serial).strip(),
    )
    if not all(camera_serials) or len(set(camera_serials)) != 3:
        parser.error("front, wrist and side camera serials must be non-empty and distinct")
    if args.inference_steps <= 0:
        parser.error("--inference-steps must be positive")
    if not args.initial_pose_json.is_file():
        parser.error(f"--initial-pose-json does not exist: {args.initial_pose_json}")
    if not args.calibration_json.is_file():
        parser.error(f"--calibration-json does not exist: {args.calibration_json}")
    if not args.hardware_calibration_json.is_file():
        parser.error(
            "--hardware-calibration-json does not exist: "
            f"{args.hardware_calibration_json}"
        )
    if not args.role_mapping_json.is_file():
        parser.error(f"--role-mapping-json does not exist: {args.role_mapping_json}")
    if not args.joint_zero_offset_json.is_file():
        parser.error(
            "--joint-zero-offset-json does not exist: "
            f"{args.joint_zero_offset_json}"
        )
    for option in (
        "home_timeout_s",
        "home_velocity_deg_s",
        "home_acceleration_deg_s2",
        "home_gripper_velocity_mm_s",
        "home_joint_tolerance_deg",
        "home_gripper_tolerance_mm",
        "home_start_limit_margin_deg",
        "home_hold_s",
    ):
        value = getattr(args, option)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{option.replace('_', '-')} must be finite and positive")
    if not 0 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [0,100]")
    if args.max_relative_target <= 0:
        parser.error("--max-relative-target must be positive")
    if not 0 < args.max_gripper_relative_target <= 100:
        parser.error("--max-gripper-relative-target must be in (0,100]")
    if args.max_initial_joint_delta_deg < 0:
        parser.error("--max-initial-joint-delta-deg must be non-negative")
    if (
        not math.isfinite(args.j2_command_offset_deg)
        or abs(args.j2_command_offset_deg) > 15.0
    ):
        parser.error("--j2-command-offset-deg must be finite and within [-15,15]")
    if (
        not math.isfinite(args.j3_command_offset_deg)
        or abs(args.j3_command_offset_deg) > 15.0
    ):
        parser.error("--j3-command-offset-deg must be finite and within [-15,15]")
    if (
        not math.isfinite(args.j5_command_offset_deg)
        or abs(args.j5_command_offset_deg) > 15.0
    ):
        parser.error("--j5-command-offset-deg must be finite and within [-15,15]")
    if args.joint_limit_margin_deg < 0:
        parser.error("--joint-limit-margin-deg must be non-negative")
    if args.max_joint_command_step <= 0:
        parser.error("--max-joint-command-step must be positive")
    if not 0 < args.max_joint_command_accel <= args.max_joint_command_step:
        parser.error(
            "--max-joint-command-accel must be in (0,--max-joint-command-step]"
        )
    if not 0 < args.temporal_replan_interval <= 30:
        parser.error("--temporal-replan-interval must be in [1,30]")
    if args.temporal_ensemble_decay < 0:
        parser.error("--temporal-ensemble-decay must be non-negative")
    if (
        args.chunk_smoothing_window <= 0
        or args.chunk_smoothing_window > 100
        or args.chunk_smoothing_window % 2 == 0
    ):
        parser.error("--chunk-smoothing-window must be an odd value in [1,100]")
    if args.chunk_smoothing_sigma <= 0:
        parser.error("--chunk-smoothing-sigma must be positive")
    if args.max_joint_command_step > args.max_relative_target:
        parser.error(
            "--max-joint-command-step must not exceed --max-relative-target"
        )
    if not 0 <= args.gripper_threshold <= 1:
        parser.error("--gripper-threshold must be in [0,1]")
    if args.gripper_close_confirm_frames <= 0:
        parser.error("--gripper-close-confirm-frames must be positive")
    if args.gripper_open_confirm_frames <= 0:
        parser.error("--gripper-open-confirm-frames must be positive")
    if args.gripper_min_close_hold_frames < 0:
        parser.error("--gripper-min-close-hold-frames must be non-negative")
    if not (
        0.0 <= args.gripper_close_feedback_threshold
        < args.gripper_open_feedback_threshold
        <= 100.0
    ):
        parser.error(
            "gripper feedback thresholds must satisfy 0 <= close < open <= 100"
        )
    if args.mode != "execute" and args.allow_motion:
        parser.error("--allow-motion is only valid with --mode execute")
    if (
        args.mode == "execute"
        and args.joint_mapping_mode != "physical-identity"
        and not args.allow_experimental_joint_mapping
    ):
        parser.error(
            "execute mode requires --joint-mapping-mode physical-identity; "
            "the recorded-percentile-affine A/B mapping was rejected by "
            "multi-view replay validation and requires the explicit "
            "--allow-experimental-joint-mapping opt-in"
        )
    if args.home_only and args.mode != "execute":
        parser.error("--home-only is only valid with --mode execute")
    if args.home_only and not args.home_at_start:
        parser.error("--home-only requires --home-at-start")
    if not math.isfinite(args.hold_current_only_seconds) or args.hold_current_only_seconds < 0:
        parser.error("--hold-current-only-seconds must be finite and non-negative")
    if args.hold_current_only_seconds > 0 and args.mode != "execute":
        parser.error("--hold-current-only-seconds is only valid with --mode execute")
    if args.hold_current_only_seconds > 0 and args.home_only:
        parser.error("--hold-current-only-seconds cannot be combined with --home-only")
    if args.mode in ("live-dry-run", "execute") and not args.can_port:
        parser.error("live modes require --can-port")
    if args.mode == "execute" and args.save_inference_images:
        parser.error(
            "execute mode forbids synchronous --save-inference-images; use live-dry-run for images"
        )
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.list_checkpoint_profiles:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    actual_sha256 = sha256_file(args.checkpoint)
    if actual_sha256 != args.checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {args.checkpoint_sha256}, got {actual_sha256}"
        )
    joint_limits, gripper_stroke_mm = load_calibration(args.calibration_json)
    adapter = JointSpaceAdapter.from_files(
        mapping_path=args.role_mapping_json,
        model_calibration_path=args.calibration_json,
        hardware_calibration_path=args.hardware_calibration_json,
        joint_zero_offset_path=args.joint_zero_offset_json,
        allow_experimental_joint_mapping=args.allow_experimental_joint_mapping,
        mapping_mode=args.joint_mapping_mode,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    policy = make_policy(args, joint_limits, gripper_stroke_mm)
    if args.mode == "checkpoint":
        logger.info(
            "Joint checkpoint loaded without constructing a robot: sha256=%s "
            "profile=%s qf=%d steps=%d contract=%s",
            actual_sha256,
            args.checkpoint_profile or "custom",
            args.query_frequency,
            args.inference_steps,
            policy.checkpoint_report,
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = JsonlLogger(args.log_dir / f"{args.mode}_{timestamp}.jsonl")
    try:
        log.write(
            {
                "event": "start",
                "mode": args.mode,
                "checkpoint_profile": args.checkpoint_profile or "custom",
                "checkpoint_description": args.checkpoint_description,
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": actual_sha256,
                "policy_space": "absolute_joint_radians_plus_binary_gripper",
                "checkpoint_contract": policy.checkpoint_report,
                "query_frequency": args.query_frequency,
                "effective_prediction_horizon_ticks": (
                    30 if args.temporal_ensemble else args.query_frequency
                ),
                "inference_steps": args.inference_steps,
                "seed": args.seed,
                "speed_percent": args.speed_percent,
                "max_relative_target": args.max_relative_target,
                "max_gripper_relative_target": args.max_gripper_relative_target,
                "max_initial_joint_delta_deg": args.max_initial_joint_delta_deg,
                "j2_command_offset_deg": args.j2_command_offset_deg,
                "j2_mapping": {
                    "model_proprio": "firmware_feedback_minus_offset",
                    "firmware_action": "model_action_plus_offset",
                    "firmware_home": "recorded_home_plus_offset",
                },
                "j3_command_offset_deg": args.j3_command_offset_deg,
                "j3_mapping": {
                    "model_proprio": "firmware_feedback_minus_offset",
                    "firmware_action": "model_action_plus_offset",
                    "firmware_home": "recorded_home_plus_offset",
                },
                "j5_command_offset_deg": args.j5_command_offset_deg,
                "j5_mapping": {
                    "model_proprio": "firmware_feedback_minus_offset",
                    "firmware_action": "model_action_plus_offset",
                    "firmware_home": "recorded_home_plus_offset",
                },
                "initial_pose_json": str(args.initial_pose_json),
                "model_calibration_json": str(args.calibration_json),
                "hardware_calibration_json": str(args.hardware_calibration_json),
                "role_mapping_json": str(args.role_mapping_json),
                "joint_zero_offset_json": str(args.joint_zero_offset_json),
                "allow_experimental_joint_mapping": (
                    args.allow_experimental_joint_mapping
                ),
                "joint_space_adapter": adapter.report(),
                "home_at_start": args.home_at_start,
                "home_at_end": args.home_at_end,
                "home_only": args.home_only,
                "hold_current_only_seconds": args.hold_current_only_seconds,
                "home_timeout_s": args.home_timeout_s,
                "home_velocity_deg_s": args.home_velocity_deg_s,
                "home_acceleration_deg_s2": args.home_acceleration_deg_s2,
                "home_gripper_velocity_mm_s": args.home_gripper_velocity_mm_s,
                "home_joint_tolerance_deg": args.home_joint_tolerance_deg,
                "home_gripper_tolerance_mm": args.home_gripper_tolerance_mm,
                "home_start_limit_margin_deg": args.home_start_limit_margin_deg,
                "home_hold_s": args.home_hold_s,
                "gripper_control_mode": args.gripper_control_mode,
                "gripper_prediction_source": args.gripper_prediction_source,
                "gripper_close_confirm_frames": args.gripper_close_confirm_frames,
                "gripper_open_confirm_frames": args.gripper_open_confirm_frames,
                "gripper_min_close_hold_frames": args.gripper_min_close_hold_frames,
                "gripper_close_feedback_threshold": args.gripper_close_feedback_threshold,
                "gripper_open_feedback_threshold": args.gripper_open_feedback_threshold,
                "gripper_terminal_open_after_release": (
                    args.gripper_terminal_open_after_release
                ),
                "joint_limit_margin_deg": args.joint_limit_margin_deg,
                "max_joint_command_step": args.max_joint_command_step,
                "max_joint_command_accel": args.max_joint_command_accel,
                "temporal_ensemble": args.temporal_ensemble,
                "temporal_replan_interval": args.temporal_replan_interval,
                "temporal_ensemble_decay": args.temporal_ensemble_decay,
                "chunk_smoothing": args.chunk_smoothing,
                "chunk_smoothing_window": args.chunk_smoothing_window,
                "chunk_smoothing_sigma": args.chunk_smoothing_sigma,
                "deterministic_inference_seed": args.seed,
                "command_sent": False,
            }
        )
        if args.mode == "offline":
            run_offline(args, policy, log, device)
        else:
            run_live(
                args,
                policy,
                log,
                device,
                joint_limits,
                gripper_stroke_mm,
                adapter,
                timestamp,
            )
    finally:
        log.close()
        logger.info("wrote joint deployment audit log to %s", log.path)


if __name__ == "__main__":
    main()
