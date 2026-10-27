#!/usr/bin/env python
"""Guarded deployment entry point for the best Piper EgoVerse checkpoint.

The default ``checkpoint`` mode never constructs a robot or opens CAN. The
``offline`` mode consumes a prepared NPZ and is also hardware-free.
``live-dry-run`` opens cameras/CAN feedback through Piper's observation-only
mode but cannot send motion. ``execute`` is deliberately double-gated and
requires explicit workspace bounds; it must only be used after the offline and
live dry-run reports have been reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.policies.egoverse_hpt.configuration_egoverse_hpt import EgoVerseHPTConfig
from lerobot.policies.egoverse_hpt.modeling_egoverse_hpt import EgoVerseHPTPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.robots.piper.config_piper import PiperConfig
from lerobot.robots.piper.piper import Piper
from lerobot.utils.robot_utils import precise_sleep

DEFAULT_CHECKPOINT = Path(
    "/mnt/wd10/kerun/checkpoints/piper_finetune_geodesic_2726084_step8000.ckpt"
)
DEFAULT_CHECKPOINT_SHA256 = "c4f57023c804b4a2543fdf539fc14b6c51266b0a670a412d03517f6751b9233f"
DEFAULT_CALIBRATION = Path(
    "/mnt/wd10/kerun/calibration/my_piper_follower.json"
)
DEFAULT_LOG_DIR = Path("/mnt/wd10/kerun/deploy_logs")
FRONT_SERIAL = "327122075831"  # D435IF, confirmed front camera
WRIST_SERIAL = "427622273943"  # D405, confirmed wrist camera
FRONT_CAMERA_AUTO_WARMUP_S = 5.0

logger = logging.getLogger("deploy_egoverse_hpt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_xyz(value: str | None) -> list[float] | None:
    if value is None:
        return None
    result = [float(item) for item in value.split(",")]
    if len(result) != 3:
        raise argparse.ArgumentTypeError("XYZ bounds must be comma-separated x,y,z triples")
    return result


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


def make_policy(args: argparse.Namespace, joint_limits: list[tuple[float, float]]) -> EgoVerseHPTPolicy:
    config = EgoVerseHPTConfig(
        ckpt_path=str(args.checkpoint),
        joint_limits_deg=joint_limits,
        device=args.device,
        action_frame="absolute",
        action_horizon_frames=30,
        query_frequency=args.query_frequency,
        num_inference_steps=10,
        expected_image_height=480,
        expected_image_width=640,
        resize_mismatched_images=False,
        gripper_threshold=0.5,
        normalize_proprio_input=args.proprio_normalization == "on",
        max_consecutive_ik_failures=args.max_consecutive_ik_failures,
        workspace_xyz_min=args.workspace_min,
        workspace_xyz_max=args.workspace_max,
    )
    return EgoVerseHPTPolicy(config)


def make_policy_batch(
    state: np.ndarray,
    front_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    observation = {
        "observation.state": np.asarray(state),
        "observation.images.435if": np.asarray(front_rgb),
        "observation.images.405": np.asarray(wrist_rgb),
    }
    return prepare_observation_for_inference(observation, device, task=None, robot_type="piper")


def live_observation_to_batch(observation: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    state = np.asarray([observation[f"joint_{index}.pos"] for index in range(7)], dtype=np.float32)
    return make_policy_batch(state, observation["435if"], observation["405"], device)


def action_to_robot_dict(action: torch.Tensor) -> dict[str, float]:
    values = action.detach().cpu().numpy().reshape(-1)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise RuntimeError(f"refusing malformed policy action with shape/value {values}")
    return {f"joint_{index}.pos": float(value) for index, value in enumerate(values)}


def observation_to_normalized_state(
    observation: dict[str, Any],
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> np.ndarray:
    """Convert Piper raw feedback to the command coordinate system.

    Joints 0-5 are raw 0.001-degree values mapped to [-100, 100]. The
    gripper is raw 0.001-mm mapped to [0, 100]. This mirrors
    PiperSDKInterface.get_status_normalized without issuing a second CAN read,
    so queue progress is evaluated against the exact observation used by the
    policy on this control cycle.
    """
    normalized = np.empty(7, dtype=np.float64)
    for index, (low_deg, high_deg) in enumerate(joint_limits):
        if high_deg <= low_deg:
            raise ValueError(f"invalid joint {index} limits: {(low_deg, high_deg)}")
        raw_deg = float(observation[f"joint_{index}.pos"]) / 1000.0
        normalized[index] = (raw_deg - low_deg) / (high_deg - low_deg) * 200.0 - 100.0
    if gripper_stroke_mm <= 0:
        raise ValueError(f"invalid gripper stroke: {gripper_stroke_mm}")
    gripper_mm = float(observation["joint_6.pos"]) / 1000.0
    normalized[6] = np.clip(gripper_mm / gripper_stroke_mm * 100.0, 0.0, 100.0)
    if not np.isfinite(normalized).all():
        raise RuntimeError(f"non-finite normalized robot feedback: {normalized.tolist()}")
    return normalized


class FeedbackQueueTracker:
    """Decide whether measured arm progress permits consuming a waypoint.

    The gripper is deliberately excluded from the reach criterion: its
    physical fully-open feedback can remain a few normalized units below 100,
    which must not permanently block arm trajectory progress.
    """

    def __init__(
        self,
        *,
        joint_tolerance: float,
        min_progress: float,
        stall_steps: int,
        max_hold_steps: int,
    ) -> None:
        self.joint_tolerance = joint_tolerance
        self.min_progress = min_progress
        self.stall_steps_limit = stall_steps
        self.max_hold_steps = max_hold_steps
        self._target_key: tuple[int, int] | None = None
        self._best_rms_error = float("inf")
        self._stall_steps = 0
        self._hold_steps = 0

    def observe(
        self, target_key: tuple[int, int], target: np.ndarray, actual: np.ndarray
    ) -> dict[str, Any]:
        arm_error = np.asarray(target[:6], dtype=np.float64) - np.asarray(
            actual[:6], dtype=np.float64
        )
        if arm_error.shape != (6,) or not np.isfinite(arm_error).all():
            raise RuntimeError(f"invalid arm tracking error: {arm_error}")
        max_abs_error = float(np.max(np.abs(arm_error)))
        rms_error = float(np.sqrt(np.mean(np.square(arm_error))))
        new_target = target_key != self._target_key
        if new_target:
            self._target_key = target_key
            self._best_rms_error = rms_error
            self._stall_steps = 0
            self._hold_steps = 0

        reached = max_abs_error <= self.joint_tolerance
        if not reached:
            self._hold_steps += 1
            if rms_error <= self._best_rms_error - self.min_progress:
                self._best_rms_error = rms_error
                self._stall_steps = 0
            elif not new_target:
                self._stall_steps += 1

            if self._stall_steps >= self.stall_steps_limit:
                raise RuntimeError(
                    "feedback-aware queue stopped: arm tracking error made no sufficient "
                    f"progress for {self._stall_steps} control cycles "
                    f"(target={target_key}, max_error={max_abs_error:.3f}, "
                    f"rms_error={rms_error:.3f}, best_rms={self._best_rms_error:.3f})"
                )
            if self._hold_steps >= self.max_hold_steps:
                raise RuntimeError(
                    "feedback-aware queue stopped: target was not reached before the hold "
                    f"limit of {self._hold_steps} cycles "
                    f"(target={target_key}, max_error={max_abs_error:.3f}, "
                    f"rms_error={rms_error:.3f})"
                )

        return {
            "target_key": list(target_key),
            "new_target": new_target,
            "reached": reached,
            "arm_error": arm_error.tolist(),
            "max_abs_arm_error": max_abs_error,
            "rms_arm_error": rms_error,
            "best_rms_arm_error": self._best_rms_error,
            "stall_steps": self._stall_steps,
            "hold_steps": self._hold_steps,
        }


class PhaseAlignedTemporalAggregator:
    """Blend overlapping action chunks on an absolute policy-step timeline.

    A fresh HPT chunk is *not* assumed to start at the robot's current task
    phase.  Before adding it to the timeline, its prefix is searched for the
    waypoint that best matches the already-planned waypoint at the current
    absolute policy step.  Cartesian position and arm-joint continuity are
    both mandatory safety gates.  The matching suffix is then timestamped on
    the same absolute timeline and blended with every older prediction for
    each future step.

    Timeline progress is advanced only by :meth:`advance_action`, which the
    execute loop calls after measured joint feedback reaches the current
    target.  Holding a clipped target therefore never advances time and never
    causes the temporal ensemble to skip actions.
    """

    def __init__(
        self,
        policy: EgoVerseHPTPolicy,
        *,
        replan_interval: int,
        ensemble_decay: float,
        max_alignment_xyz_m: float,
        max_alignment_joint_rms: float,
        max_output_joint_rms: float,
    ) -> None:
        if replan_interval <= 0:
            raise ValueError("temporal replan_interval must be positive")
        if ensemble_decay < 0:
            raise ValueError("temporal ensemble_decay must be non-negative")
        if max_alignment_xyz_m <= 0:
            raise ValueError("temporal max_alignment_xyz_m must be positive")
        if max_alignment_joint_rms <= 0:
            raise ValueError("temporal max_alignment_joint_rms must be positive")
        if max_output_joint_rms <= 0:
            raise ValueError("temporal max_output_joint_rms must be positive")
        self.policy = policy
        self.replan_interval = int(replan_interval)
        self.ensemble_decay = float(ensemble_decay)
        self.max_alignment_xyz_m = float(max_alignment_xyz_m)
        self.max_alignment_joint_rms = float(max_alignment_joint_rms)
        self.max_output_joint_rms = float(max_output_joint_rms)
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self._timeline: dict[int, list[dict[str, Any]]] = {}
        self._absolute_index = 0
        self._prediction_id = 0
        self._plan_revision = 0
        self._next_replan_index = 0
        self._last_replan_index: int | None = None
        self._last_action: np.ndarray | None = None
        self._last_xyz: np.ndarray | None = None
        self._last_alignment: dict[str, Any] | None = None
        self.last_prediction_report: dict[str, Any] | None = None

    @staticmethod
    def _valid_xyz(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        xyz = np.asarray(value, dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            return None
        return xyz

    def _aggregate_slot(self, absolute_index: int) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        samples = self._timeline.get(absolute_index, [])
        if not samples:
            raise RuntimeError(
                "phase-aligned temporal aggregation has no prediction for absolute "
                f"policy step {absolute_index}"
            )
        newest_id = max(int(sample["prediction_id"]) for sample in samples)
        weights = np.asarray(
            [
                np.exp(-self.ensemble_decay * (newest_id - int(sample["prediction_id"])))
                for sample in samples
            ],
            dtype=np.float64,
        )
        weights /= np.sum(weights)
        actions = np.stack([np.asarray(sample["action"], dtype=np.float64) for sample in samples])
        action = np.sum(actions * weights[:, None], axis=0)
        # Never send a fractional gripper command.  Temporal aggregation is a
        # weighted vote over the binary per-chunk decisions.
        action[6] = 100.0 if action[6] >= 50.0 else 0.0

        xyz_samples = [self._valid_xyz(sample.get("xyz")) for sample in samples]
        valid_xyz = [(xyz, weight) for xyz, weight in zip(xyz_samples, weights) if xyz is not None]
        xyz = None
        if valid_xyz:
            xyz_weights = np.asarray([weight for _, weight in valid_xyz], dtype=np.float64)
            xyz_weights /= np.sum(xyz_weights)
            xyz = np.sum(
                np.stack([value for value, _ in valid_xyz]) * xyz_weights[:, None], axis=0
            )
        metadata = {
            "absolute_index": int(absolute_index),
            "source_prediction_ids": [int(sample["prediction_id"]) for sample in samples],
            "source_local_offsets": [int(sample["local_offset"]) for sample in samples],
            "weights": weights.tolist(),
        }
        return action, xyz, metadata

    def _alignment_reference(self) -> tuple[np.ndarray, np.ndarray | None, str]:
        if self._timeline.get(self._absolute_index):
            action, xyz, _ = self._aggregate_slot(self._absolute_index)
            return action, xyz, "existing_timeline"
        if self._last_action is not None:
            return self._last_action.copy(), None if self._last_xyz is None else self._last_xyz.copy(), "last_executed"
        raise RuntimeError("temporal alignment has neither a timeline nor a last executed action")

    def _infer_and_merge(self, batch: dict[str, torch.Tensor], **kwargs) -> None:
        chunk = self.policy.predict_action_chunk(batch, **kwargs).detach().cpu().numpy()
        report = self.policy.last_prediction_report
        if report is None:
            raise RuntimeError("HPT inference returned no prediction report for temporal alignment")
        steps = report.get("steps")
        if chunk.ndim != 2 or chunk.shape[1] != 7 or not np.isfinite(chunk).all():
            raise RuntimeError(f"invalid temporal action chunk: shape={chunk.shape}")
        if not isinstance(steps, list) or len(steps) != len(chunk):
            raise RuntimeError(
                "temporal alignment requires one Cartesian step report per action: "
                f"actions={len(chunk)}, reports={None if not isinstance(steps, list) else len(steps)}"
            )

        reference_action = None
        reference_xyz = None
        reference_source = "initial_chunk"
        output_joint_rms = None
        candidate_diagnostics: list[dict[str, Any]] = []
        if self._prediction_id == 0:
            aligned_offset = 0
        else:
            reference_action, reference_xyz, reference_source = self._alignment_reference()
            for local_offset, (action, step) in enumerate(zip(chunk, steps)):
                joint_rms = float(
                    np.sqrt(np.mean(np.square(action[:6] - reference_action[:6])))
                )
                candidate_xyz = self._valid_xyz(step.get("predicted_xyz_base_m"))
                xyz_distance_m = (
                    None
                    if reference_xyz is None or candidate_xyz is None
                    else float(np.linalg.norm(candidate_xyz - reference_xyz))
                )
                xyz_safe = xyz_distance_m is None or xyz_distance_m <= self.max_alignment_xyz_m
                joint_safe = joint_rms <= self.max_alignment_joint_rms
                # Cartesian position is the main task-phase signal.  Joint
                # distance is a secondary term that rejects IK branch jumps.
                score = joint_rms / self.max_alignment_joint_rms
                if xyz_distance_m is not None:
                    score += xyz_distance_m / self.max_alignment_xyz_m
                candidate_diagnostics.append(
                    {
                        "local_offset": int(local_offset),
                        "score": float(score),
                        "joint_rms": joint_rms,
                        "xyz_distance_m": xyz_distance_m,
                        "safe": bool(xyz_safe and joint_safe),
                    }
                )
            safe_candidates = [item for item in candidate_diagnostics if item["safe"]]
            if not safe_candidates:
                best = min(candidate_diagnostics, key=lambda item: item["score"])
                self.last_prediction_report = dict(report)
                self.last_prediction_report["temporal_alignment_rejection"] = {
                    "absolute_index": int(self._absolute_index),
                    "reason": "no_safe_phase_offset",
                    "best_candidate": best,
                    "max_alignment_joint_rms": self.max_alignment_joint_rms,
                    "max_alignment_xyz_m": self.max_alignment_xyz_m,
                }
                raise RuntimeError(
                    "phase-aligned temporal aggregation rejected a discontinuous new chunk: "
                    f"best_offset={best['local_offset']}, joint_rms={best['joint_rms']:.3f} "
                    f"(limit={self.max_alignment_joint_rms:.3f}), "
                    f"xyz_distance_m={best['xyz_distance_m']} "
                    f"(limit={self.max_alignment_xyz_m:.3f})"
                )
            aligned_offset = min(safe_candidates, key=lambda item: item["score"])["local_offset"]

            # Check the command that would actually result after temporal
            # blending, not only the raw candidate.  A moderately different
            # IK solution may be safe to contribute at low weight, while an
            # excessive change in the final arm target must still abort.
            existing_samples = list(self._timeline.get(self._absolute_index, []))
            trial_prediction_id = self._prediction_id + 1
            trial_samples = existing_samples + [
                {
                    "prediction_id": trial_prediction_id,
                    "local_offset": int(aligned_offset),
                    "action": chunk[int(aligned_offset)].astype(np.float64),
                    "xyz": steps[int(aligned_offset)].get("predicted_xyz_base_m"),
                }
            ]
            weights = np.asarray(
                [
                    np.exp(
                        -self.ensemble_decay
                        * (trial_prediction_id - int(sample["prediction_id"]))
                    )
                    for sample in trial_samples
                ],
                dtype=np.float64,
            )
            weights /= np.sum(weights)
            blended_action = np.sum(
                np.stack([np.asarray(sample["action"], dtype=np.float64) for sample in trial_samples])
                * weights[:, None],
                axis=0,
            )
            output_joint_rms = float(
                np.sqrt(np.mean(np.square(blended_action[:6] - reference_action[:6])))
            )
            if output_joint_rms > self.max_output_joint_rms:
                self.last_prediction_report = dict(report)
                self.last_prediction_report["temporal_alignment_rejection"] = {
                    "absolute_index": int(self._absolute_index),
                    "reason": "blended_output_discontinuity",
                    "aligned_local_offset": int(aligned_offset),
                    "candidate_joint_rms": next(
                        item["joint_rms"]
                        for item in candidate_diagnostics
                        if item["local_offset"] == aligned_offset
                    ),
                    "output_joint_rms": output_joint_rms,
                    "max_output_joint_rms": self.max_output_joint_rms,
                    "weights": weights.tolist(),
                }
                raise RuntimeError(
                    "phase-aligned temporal aggregation rejected the blended arm target: "
                    f"output_joint_rms={output_joint_rms:.3f} "
                    f"(limit={self.max_output_joint_rms:.3f})"
                )

        self._prediction_id += 1
        prediction_id = self._prediction_id
        for local_offset in range(int(aligned_offset), len(chunk)):
            absolute_index = self._absolute_index + local_offset - int(aligned_offset)
            self._timeline.setdefault(absolute_index, []).append(
                {
                    "prediction_id": prediction_id,
                    "local_offset": int(local_offset),
                    "action": chunk[local_offset].astype(np.float64),
                    "xyz": steps[local_offset].get("predicted_xyz_base_m"),
                }
            )

        self._plan_revision += 1
        self._last_replan_index = self._absolute_index
        self._next_replan_index = self._absolute_index + self.replan_interval
        aligned_candidate = (
            None
            if not candidate_diagnostics
            else next(
                item for item in candidate_diagnostics if item["local_offset"] == aligned_offset
            )
        )
        self._last_alignment = {
            "prediction_id": prediction_id,
            "absolute_index": int(self._absolute_index),
            "reference_source": reference_source,
            "aligned_local_offset": int(aligned_offset),
            "skipped_prefix_actions": int(aligned_offset),
            "suffix_actions_added": int(len(chunk) - aligned_offset),
            "joint_rms": None if aligned_candidate is None else aligned_candidate["joint_rms"],
            "xyz_distance_m": (
                None if aligned_candidate is None else aligned_candidate["xyz_distance_m"]
            ),
            "max_alignment_joint_rms": self.max_alignment_joint_rms,
            "max_alignment_xyz_m": self.max_alignment_xyz_m,
            "max_output_joint_rms": self.max_output_joint_rms,
        }
        self.last_prediction_report = dict(report)
        self.last_prediction_report["temporal_alignment"] = self._last_alignment

    def _should_replan(self) -> bool:
        no_current_prediction = not self._timeline.get(self._absolute_index)
        scheduled = self._absolute_index >= self._next_replan_index
        return (no_current_prediction or scheduled) and self._last_replan_index != self._absolute_index

    @torch.no_grad()
    def peek_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        self.last_prediction_report = None
        if self._should_replan():
            self._infer_and_merge(batch, **kwargs)
        action, _, _ = self._aggregate_slot(self._absolute_index)
        return torch.from_numpy(action.astype(np.float32))

    def advance_action(self) -> torch.Tensor:
        action, xyz, _ = self._aggregate_slot(self._absolute_index)
        self._last_action = action.copy()
        self._last_xyz = None if xyz is None else xyz.copy()
        del self._timeline[self._absolute_index]
        self._absolute_index += 1
        return torch.from_numpy(action.astype(np.float32))

    def queue_status(self) -> dict[str, Any]:
        slots = sorted(index for index, samples in self._timeline.items() if samples)
        _, _, ensemble = self._aggregate_slot(self._absolute_index) if slots else (None, None, None)
        return {
            # Keep these four names compatible with FeedbackQueueTracker and
            # existing deployment-log analysis.
            "chunk_id": int(self._plan_revision),
            "chunk_size": 0 if not slots else int(slots[-1] + 1),
            "action_offset": int(self._absolute_index),
            "pending_actions": int(len(slots)),
            "temporal_mode": True,
            "timeline_end": None if not slots else int(slots[-1]),
            "prediction_count": int(self._prediction_id),
            "next_replan_index": int(self._next_replan_index),
            "last_alignment": self._last_alignment,
            "current_ensemble": ensemble,
        }


def configure_front_camera_auto_controls(robot: Piper) -> dict[str, float]:
    """Enable D435IF auto photometric controls and let them settle before inference."""
    import pyrealsense2 as rs

    front_camera = robot.cameras.get("435if")
    if front_camera is None or front_camera.rs_profile is None:
        raise RuntimeError("front RealSense camera is not connected")

    color_sensor = front_camera.rs_profile.get_device().first_color_sensor()
    required_options = {
        "auto_exposure": rs.option.enable_auto_exposure,
        "exposure": rs.option.exposure,
        "gain": rs.option.gain,
        "auto_white_balance": rs.option.enable_auto_white_balance,
        "white_balance": rs.option.white_balance,
    }
    unsupported = [
        name for name, option in required_options.items() if not color_sensor.supports(option)
    ]
    if unsupported:
        raise RuntimeError(f"front camera does not support required controls: {unsupported}")

    color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
    color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0)

    deadline = time.monotonic() + FRONT_CAMERA_AUTO_WARMUP_S
    while time.monotonic() < deadline:
        front_camera.read(timeout_ms=1000)

    report = {
        name: float(color_sensor.get_option(option)) for name, option in required_options.items()
    }
    if report["auto_exposure"] != 1.0 or report["auto_white_balance"] != 1.0:
        raise RuntimeError(f"front camera automatic controls did not remain enabled: {report}")
    logger.info("Front camera automatic controls settled: %s", report)
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


def run_offline(
    args: argparse.Namespace, policy: EgoVerseHPTPolicy, log: JsonlLogger, device: torch.device
) -> None:
    if args.offline_npz is None:
        raise ValueError("offline mode requires --offline-npz")
    required = ("observation.state", "observation.images.435if", "observation.images.405")
    with np.load(args.offline_npz, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"offline NPZ is missing {missing}; required keys are {required}")
        # Materialize compressed members once. Indexing NPZ members in the
        # loop would decompress entire image arrays every frame and corrupt
        # the deployment latency measurement by hundreds of milliseconds.
        arrays = {key: archive[key] for key in required}
    lengths = [len(arrays[key]) for key in required]
    frame_count = min(min(lengths), args.max_steps)
    temporal = (
        PhaseAlignedTemporalAggregator(
            policy,
            replan_interval=args.temporal_replan_interval,
            ensemble_decay=args.temporal_ensemble_decay,
            max_alignment_xyz_m=args.temporal_max_alignment_xyz_m,
            max_alignment_joint_rms=args.temporal_max_alignment_joint_rms,
            max_output_joint_rms=args.temporal_max_output_joint_rms,
        )
        if args.queue_mode == "temporal"
        else None
    )
    for frame_index in range(frame_count):
        started = time.perf_counter()
        batch = make_policy_batch(
            arrays[required[0]][frame_index],
            arrays[required[1]][frame_index],
            arrays[required[2]][frame_index],
            device,
        )
        if temporal is not None:
            try:
                action = temporal.peek_action(batch)
                new_chunk_report = temporal.last_prediction_report
                queue_status = temporal.queue_status()
                temporal.advance_action()
            except RuntimeError as exc:
                log.write(
                    {
                        "event": "temporal_abort",
                        "mode": "offline",
                        "frame": frame_index,
                        "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                        "reason": str(exc),
                        "queue_status": temporal.queue_status(),
                        "new_chunk_report": (
                            temporal.last_prediction_report or policy.last_prediction_report
                        ),
                        "command_sent": False,
                    }
                )
                raise
        else:
            action = policy.select_action(batch)
            new_chunk_report = policy.last_prediction_report
            queue_status = policy.queue_status()
        log.write(
            {
                "mode": "offline",
                "frame": frame_index,
                "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                "total_step_ms": (time.perf_counter() - started) * 1e3,
                "action": action.detach().cpu().tolist(),
                "queue_mode": args.queue_mode,
                "queue_status": queue_status,
                "new_chunk_report": new_chunk_report,
                "command_sent": False,
            }
        )


def make_robot(
    args: argparse.Namespace,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
    enable_motion: bool,
) -> Piper:
    if args.can_port is None:
        raise ValueError("live modes require an explicitly verified --can-port; no default is allowed")
    cameras = {
        "435if": RealSenseCameraConfig(
            serial_number_or_name=FRONT_SERIAL,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        ),
        "405": RealSenseCameraConfig(
            serial_number_or_name=WRIST_SERIAL,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
        ),
    }
    return Piper(
        PiperConfig(
            id="my_piper_follower",
            port=args.can_port,
            cameras=cameras,
            joint_limits_deg=joint_limits,
            gripper_stroke_mm=gripper_stroke_mm,
            speed_percent=args.speed_percent,
            max_relative_target=args.max_relative_target,
            enable_motion_on_connect=enable_motion,
            home_on_disconnect=False,
            disable_torque_on_disconnect=False,
        )
    )


def run_live(
    args: argparse.Namespace,
    policy: EgoVerseHPTPolicy,
    log: JsonlLogger,
    device: torch.device,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> None:
    execute = args.mode == "execute"
    if execute:
        if not args.allow_motion:
            raise RuntimeError("execute mode requires --allow-motion")
        if args.workspace_min is None or args.workspace_max is None:
            raise RuntimeError("execute mode requires explicit --workspace-min and --workspace-max")
    # Always connect observation-only first.  Camera/model/IK warmup can fail,
    # and none of those checks require an enabled arm.  Execute mode is
    # promoted only after warmup succeeds, at which point the SDK resumes an
    # emergency-stopped controller and immediately latches measured feedback
    # as the first target.
    robot = make_robot(args, joint_limits, gripper_stroke_mm, enable_motion=False)
    robot.connect(calibrate=False)
    try:
        front_camera_report = configure_front_camera_auto_controls(robot)
        log.write(
            {
                "event": "camera_config",
                "mode": args.mode,
                "camera": "435if",
                "serial": FRONT_SERIAL,
                **front_camera_report,
                "command_sent": False,
            }
        )
        # Warm CUDA kernels and IK before the timed loop. The proposed action
        # is deliberately discarded; even execute mode sends nothing here.
        warmup_started = time.perf_counter()
        warmup_observation = robot.get_observation()
        warmup_action = policy.select_action(live_observation_to_batch(warmup_observation, device))
        log.write(
            {
                "event": "warmup",
                "mode": args.mode,
                "total_step_ms": (time.perf_counter() - warmup_started) * 1e3,
                "discarded_action": warmup_action.detach().cpu().tolist(),
                "chunk_report": policy.last_prediction_report,
                "command_sent": False,
            }
        )
        policy.reset()
        if execute:
            held_position = robot.enable_motion_hold_current()
            log.write(
                {
                    "event": "motion_enable",
                    "mode": args.mode,
                    "held_position": held_position,
                    "command_sent": True,
                }
            )
        temporal = (
            PhaseAlignedTemporalAggregator(
                policy,
                replan_interval=args.temporal_replan_interval,
                ensemble_decay=args.temporal_ensemble_decay,
                max_alignment_xyz_m=args.temporal_max_alignment_xyz_m,
                max_alignment_joint_rms=args.temporal_max_alignment_joint_rms,
                max_output_joint_rms=args.temporal_max_output_joint_rms,
            )
            if execute and args.queue_mode == "temporal"
            else None
        )
        action_provider = temporal if temporal is not None else policy
        use_feedback_queue = execute and args.queue_mode in ("feedback", "temporal")
        queue_tracker = (
            FeedbackQueueTracker(
                joint_tolerance=args.queue_joint_tolerance,
                min_progress=args.queue_min_progress,
                stall_steps=args.queue_stall_steps,
                max_hold_steps=args.queue_max_hold_steps,
            )
            if use_feedback_queue
            else None
        )
        started_run = time.perf_counter()
        for frame_index in range(args.max_steps):
            loop_started = time.perf_counter()
            observation = robot.get_observation()
            batch = live_observation_to_batch(observation, device)
            normalized_state = observation_to_normalized_state(
                observation, joint_limits, gripper_stroke_mm
            )
            new_chunk_report = None
            queue_feedback = None
            queue_actions_advanced = 0

            if use_feedback_queue:
                evaluated_status = None
                evaluated_target = None
                try:
                    # Peek first: consuming a waypoint is now conditional on
                    # measured feedback. Temporal-mode inference/alignment is
                    # inside this guarded section so a rejected new chunk is
                    # logged before any discontinuous command can be sent.
                    action = action_provider.peek_action(batch)
                    if action_provider.last_prediction_report is not None:
                        new_chunk_report = action_provider.last_prediction_report
                    evaluated_status = action_provider.queue_status()
                    evaluated_target = action.detach().cpu().numpy().reshape(-1)
                    target_key = (
                        evaluated_status["chunk_id"],
                        evaluated_status["action_offset"],
                    )
                    queue_feedback = queue_tracker.observe(
                        target_key, evaluated_target, normalized_state
                    )

                    if queue_feedback["reached"]:
                        # Advance at most one planned action per real control
                        # tick. In temporal mode this increments the absolute
                        # policy-step clock and may trigger one overlapping
                        # replan for the newly-current step.
                        action_provider.advance_action()
                        queue_actions_advanced = 1
                        action = action_provider.peek_action(batch)
                        if action_provider.last_prediction_report is not None:
                            new_chunk_report = action_provider.last_prediction_report
                except RuntimeError as exc:
                    failure_report = (
                        action_provider.last_prediction_report
                        if action_provider.last_prediction_report is not None
                        else policy.last_prediction_report
                    )
                    try:
                        failure_status = action_provider.queue_status()
                    except RuntimeError:
                        failure_status = evaluated_status
                    log.write(
                        {
                            "event": "queue_abort",
                            "mode": args.mode,
                            "frame": frame_index,
                            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                            "reason": str(exc),
                            "normalized_state": normalized_state.tolist(),
                            "evaluated_queue_status": failure_status,
                            "evaluated_target": (
                                None if evaluated_target is None else evaluated_target.tolist()
                            ),
                            "new_chunk_report": failure_report,
                            "command_sent": False,
                        }
                    )
                    raise
            else:
                action = policy.select_action(batch)
                new_chunk_report = policy.last_prediction_report

            robot_action = action_to_robot_dict(action)
            sent_action = robot.send_action(robot_action) if execute else None
            command_was_clipped = (
                any(
                    abs(robot_action[key] - sent_action[key]) > 1e-6
                    for key in robot_action
                )
                if sent_action is not None
                else None
            )
            command_target = np.asarray(
                [robot_action[f"joint_{index}.pos"] for index in range(7)], dtype=np.float64
            )
            log.write(
                {
                    "mode": args.mode,
                    "frame": frame_index,
                    "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": time.perf_counter() - started_run,
                    "total_step_ms": (time.perf_counter() - loop_started) * 1e3,
                    "front_shape": list(np.asarray(observation["435if"]).shape),
                    "wrist_shape": list(np.asarray(observation["405"]).shape),
                    "raw_state": [float(observation[f"joint_{i}.pos"]) for i in range(7)],
                    "normalized_state": normalized_state.tolist(),
                    "proposed_action": robot_action,
                    "sent_action": sent_action,
                    "command_was_clipped": command_was_clipped,
                    "queue_mode": args.queue_mode if execute else "legacy",
                    "queue_status": action_provider.queue_status(),
                    "queue_feedback": queue_feedback,
                    "queue_actions_advanced": queue_actions_advanced,
                    "command_target_max_abs_arm_error": float(
                        np.max(np.abs(command_target[:6] - normalized_state[:6]))
                    ),
                    "new_chunk_report": new_chunk_report,
                    "command_sent": execute,
                }
            )
            precise_sleep(1.0 / 30.0 - (time.perf_counter() - loop_started))
    finally:
        # Observation-only disconnect only closes CAN/cameras. Execute mode is
        # configured to leave torque enabled rather than unexpectedly dropping
        # the arm; the operator must follow the physical shutdown procedure.
        robot.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("checkpoint", "offline", "live-dry-run", "execute"), default="checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256)
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--proprio-normalization",
        choices=("on", "off"),
        default="on",
        help=(
            "Apply the checkpoint's quantile norm_stats to observations.state.ee_pose "
            "before HPT. Use 'off' to pass physical proprio values through unchanged."
        ),
    )
    parser.add_argument("--query-frequency", type=int, choices=range(1, 31), default=5)
    parser.add_argument("--offline-npz", type=Path)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--can-port", help="Must be explicitly verified; intentionally has no default")
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--max-relative-target", type=float, default=2.0)
    parser.add_argument(
        "--max-consecutive-ik-failures",
        type=int,
        default=0,
        help=(
            "Abort after this many consecutive IK failures. Set to 0 to disable this abort; "
            "each failed point still holds the previous joint target."
        ),
    )
    parser.add_argument(
        "--queue-mode",
        choices=("temporal", "feedback", "legacy"),
        default="feedback",
        help=(
            "'temporal' adds phase-aligned temporal aggregation on top of feedback-aware "
            "waypoint consumption. Execute mode otherwise defaults to feedback-aware consumption. "
            "'legacy' consumes one queued action per loop regardless of tracking/clipping."
        ),
    )
    parser.add_argument(
        "--queue-joint-tolerance",
        type=float,
        default=1.0,
        help="Maximum normalized arm-joint error for advancing one queued action.",
    )
    parser.add_argument(
        "--queue-min-progress",
        type=float,
        default=0.05,
        help="Required reduction in normalized RMS arm error before the stall counter resets.",
    )
    parser.add_argument(
        "--queue-stall-steps",
        type=int,
        default=45,
        help="Abort after this many cycles without sufficient tracking-error reduction.",
    )
    parser.add_argument(
        "--queue-max-hold-steps",
        type=int,
        default=150,
        help="Abort if one queued target is not reached within this many control cycles.",
    )
    parser.add_argument(
        "--temporal-replan-interval",
        type=int,
        default=10,
        help=(
            "For --queue-mode temporal, infer and merge a new overlapping chunk after this "
            "many feedback-accepted policy waypoints (not wall-clock control cycles)."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.3,
        help=(
            "Exponential age decay across overlapping predictions; 0 gives equal weights and "
            "larger values favor the newest phase-aligned chunk."
        ),
    )
    parser.add_argument(
        "--temporal-max-alignment-xyz-m",
        type=float,
        default=0.03,
        help="Reject a new chunk if no phase offset is within this Cartesian distance.",
    )
    parser.add_argument(
        "--temporal-max-alignment-joint-rms",
        type=float,
        default=15.0,
        help="Reject a new chunk if no phase offset is within this normalized arm-joint RMS.",
    )
    parser.add_argument(
        "--temporal-max-output-joint-rms",
        type=float,
        default=8.0,
        help=(
            "Reject temporal merging if the final blended arm target changes by more than this "
            "normalized joint RMS from the existing absolute-timeline target."
        ),
    )
    parser.add_argument("--workspace-min", type=parse_xyz)
    parser.add_argument("--workspace-max", type=parse_xyz)
    parser.add_argument("--allow-motion", action="store_true")
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.mode != "execute" and args.allow_motion:
        parser.error("--allow-motion is only valid with --mode execute")
    if not 0 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [0,100]")
    if args.max_relative_target <= 0:
        parser.error("--max-relative-target must be positive")
    if args.max_consecutive_ik_failures < 0:
        parser.error("--max-consecutive-ik-failures must be non-negative")
    if args.queue_joint_tolerance <= 0:
        parser.error("--queue-joint-tolerance must be positive")
    if args.queue_min_progress < 0:
        parser.error("--queue-min-progress must be non-negative")
    if args.queue_stall_steps <= 0:
        parser.error("--queue-stall-steps must be positive")
    if args.queue_max_hold_steps <= 0:
        parser.error("--queue-max-hold-steps must be positive")
    if args.temporal_replan_interval <= 0:
        parser.error("--temporal-replan-interval must be positive")
    if args.queue_mode == "temporal" and args.temporal_replan_interval >= args.query_frequency:
        parser.error("--temporal-replan-interval must be smaller than --query-frequency")
    if args.temporal_ensemble_decay < 0:
        parser.error("--temporal-ensemble-decay must be non-negative")
    if args.temporal_max_alignment_xyz_m <= 0:
        parser.error("--temporal-max-alignment-xyz-m must be positive")
    if args.temporal_max_alignment_joint_rms <= 0:
        parser.error("--temporal-max-alignment-joint-rms must be positive")
    if args.temporal_max_output_joint_rms <= 0:
        parser.error("--temporal-max-output-joint-rms must be positive")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    actual_sha256 = sha256_file(args.checkpoint)
    if actual_sha256 != args.checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {args.checkpoint_sha256}, got {actual_sha256}"
        )
    joint_limits, gripper_stroke_mm = load_calibration(args.calibration_json)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    policy = make_policy(args, joint_limits)
    if args.mode == "checkpoint":
        logger.info(
            "Checkpoint loaded without constructing a robot: sha256=%s, action_frame=absolute, "
            "qf=%d, inference_steps=10, model_chunk=100x10",
            actual_sha256,
            args.query_frequency,
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = JsonlLogger(args.log_dir / f"{args.mode}_{timestamp}.jsonl")
    try:
        log.write(
            {
                "event": "start",
                "mode": args.mode,
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": actual_sha256,
                "action_frame": "absolute",
                "proprio_normalization": args.proprio_normalization,
                "query_frequency": args.query_frequency,
                "num_inference_steps": 10,
                "speed_percent": args.speed_percent,
                "max_relative_target": args.max_relative_target,
                "max_consecutive_ik_failures": args.max_consecutive_ik_failures,
                "workspace_min": args.workspace_min,
                "workspace_max": args.workspace_max,
                "queue_mode": args.queue_mode if args.mode == "execute" else "legacy",
                "queue_joint_tolerance": args.queue_joint_tolerance,
                "queue_min_progress": args.queue_min_progress,
                "queue_stall_steps": args.queue_stall_steps,
                "queue_max_hold_steps": args.queue_max_hold_steps,
                "temporal_replan_interval": args.temporal_replan_interval,
                "temporal_ensemble_decay": args.temporal_ensemble_decay,
                "temporal_max_alignment_xyz_m": args.temporal_max_alignment_xyz_m,
                "temporal_max_alignment_joint_rms": args.temporal_max_alignment_joint_rms,
                "temporal_max_output_joint_rms": args.temporal_max_output_joint_rms,
                "command_sent": False,
            }
        )
        if args.mode == "offline":
            run_offline(args, policy, log, device)
        else:
            run_live(args, policy, log, device, joint_limits, gripper_stroke_mm)
    finally:
        log.close()
        logger.info("Wrote deployment audit log to %s", log.path)


if __name__ == "__main__":
    main()
