#!/usr/bin/env python3
"""Strict preflight for the trimmed three-camera Assembly New1 Zarr dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from PIL import Image


EXPECTED_SPLITS = {"train": 90, "valid": 5, "test": 5}
HUMAN_EXPECTED_SPLITS = {"train": 54, "valid": 3, "test": 3}
EXPECTED_TOTAL_FRAMES = 186065
NUMERIC = {
    "obs_joint": 6,
    "cmd_joint": 6,
    "obs_ee_pose": 7,
    "cmd_ee_pose": 7,
    "obs_gripper": 1,
    "cmd_gripper": 1,
}
IMAGES = ("images.front_1", "images.wrist_1", "images.side_1")
HUMAN_NUMERIC = {
    "right.obs_ee_pose": 7,
    "right.obs_wrist_pose": 7,
    "right.obs_aria_keypoints": 63,
    "right.obs_keypoints": 63,
    "obs_head_pose": 7,
}


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_bytes(value: Any) -> bytes:
    while isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(type(value).__name__)
    return bytes(value)


def check_image(array: Any, index: int, label: str, audit: Audit) -> None:
    try:
        payload = scalar_bytes(array[index])
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            audit.require(image.size == (640, 480), f"{label}: size={image.size}")
            audit.require(image.mode == "RGB", f"{label}: mode={image.mode}")
    except Exception as exc:
        audit.errors.append(f"{label}: JPEG decode failed: {exc}")


def calibration_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    low = np.deg2rad(
        [float(payload[f"joint_{i}"]["range_min"]) / 1000.0 for i in range(6)]
    )
    high = np.deg2rad(
        [float(payload[f"joint_{i}"]["range_max"]) / 1000.0 for i in range(6)]
    )
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-root", type=Path, required=True)
    parser.add_argument("--human-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--human-split-json", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--camera-contract", type=Path, required=True)
    parser.add_argument("--extrinsics-json", type=Path, required=True)
    parser.add_argument("--formal-acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = Audit()
    split_manifest = json.loads(args.split_json.read_text())
    camera_contract = json.loads(args.camera_contract.read_text())
    extrinsics = json.loads(args.extrinsics_json.read_text())
    formal_acceptance = json.loads(args.formal_acceptance.read_text())
    expected_by_index = split_manifest["split_by_episode"]
    human_split_manifest = json.loads(args.human_split_json.read_text())
    human_expected_by_index = human_split_manifest["split_by_episode"]
    cal_sha = sha256(args.calibration_json)
    low, high = calibration_bounds(args.calibration_json)

    audit.require(
        camera_contract.get("excluded_source_keys") == ["observation.images.aria"],
        "camera contract must exclude only observation.images.aria",
    )
    audit.require(camera_contract.get("phase_sampling") is False, "phase sampling must be disabled")
    # Use the immutable calibration snapshot shipped with this bundle.  The
    # split manifest deliberately records the original collection path, which
    # is a local workstation path and is not expected to exist on HPC.
    source_calib = args.extrinsics_json.parent / "head_cam_to_follower_base_recording.calib"
    audit.require(source_calib.exists(), f"missing bundled source calibration {source_calib}")
    if source_calib.exists():
        audit.require(
            sha256(source_calib) == extrinsics.get("source_sha256"),
            "extrinsics source SHA does not match raw dataset",
        )
    audit.require(
        extrinsics.get("matrix_convention") == "base_world_from_camera_4x4",
        "unexpected extrinsics matrix convention",
    )
    audit.require(
        formal_acceptance.get("extrinsics_sha256") == sha256(args.extrinsics_json),
        "formal acceptance refers to a different extrinsics file",
    )
    matrix = np.asarray(extrinsics.get("matrix"), dtype=np.float64)
    audit.require(matrix.shape == (4, 4), "extrinsics matrix is not 4x4")
    if matrix.shape == (4, 4):
        rotation = matrix[:3, :3]
        audit.require(np.all(np.isfinite(matrix)), "extrinsics contains NaN/Inf")
        audit.require(
            np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6),
            "extrinsics rotation is not orthonormal",
        )
        audit.require(
            np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6),
            "extrinsics rotation determinant is not +1",
        )
    formal_extrinsics_accepted = formal_acceptance.get("accepted") is True
    audit.warn(
        formal_extrinsics_accepted,
        "head extrinsics came with the recording but lack held-out acceptance; formal mixed training remains gated",
    )

    paths = sorted(args.robot_root.glob("*.zarr"))
    audit.require(len(paths) == 100, f"expected 100 Zarr episodes, found {len(paths)}")
    splits: Counter[str] = Counter()
    indices: list[int] = []
    total_frames = 0
    removed: list[int] = []
    transitions: list[int] = []
    grip_counts: Counter[int] = Counter()
    train_grip_counts: Counter[int] = Counter()
    clip_counts = np.zeros(6, dtype=np.int64)
    cmd_min = np.full(6, np.inf)
    cmd_max = np.full(6, -np.inf)
    obs_min = np.full(6, np.inf)
    obs_max = np.full(6, -np.inf)
    residual_max_deg = np.zeros(6, dtype=np.float64)
    hpt_residual_over_limit = np.zeros(6, dtype=np.int64)
    pi05_residual_scale_deg = np.asarray([100.0, 135.0, 135.0, 160.0, 135.0, 150.0])
    pi05_residual_over_limit = np.zeros(6, dtype=np.int64)

    for path in paths:
        label = path.name
        try:
            group = zarr.open_group(str(path), mode="r")
            frames = int(group.attrs["total_frames"])
            source_frames = int(group.attrs["source_frames"])
            index = int(group.attrs["source_episode_index"])
            split = str(group.attrs["dataset_split"])
        except Exception as exc:
            audit.errors.append(f"{label}: cannot read attrs: {exc}")
            continue
        indices.append(index)
        splits[split] += 1
        total_frames += frames
        removed.append(source_frames - frames)
        audit.require(split == expected_by_index.get(str(index)), f"{label}: split mismatch")
        audit.require(frames > 30, f"{label}: length {frames} <= horizon 30")
        audit.require(31 <= source_frames - frames <= 44, f"{label}: unexpected trim count")
        trim = dict(group.attrs.get("trim", {}))
        audit.require(trim.get("retain_after_last_motion_frames") == 5, f"{label}: wrong tail retain")
        audit.require(group.attrs.get("data_contract") == "assembly_task_new1_3cam_v2", f"{label}: wrong contract")
        audit.require(group.attrs.get("calibration_sha256") == cal_sha, f"{label}: calibration SHA mismatch")
        audit.require(group.attrs.get("phase_sampling") is False, f"{label}: phase sampling enabled")
        audit.require(group.attrs.get("excluded_source_cameras") == ["observation.images.aria"], f"{label}: aria exclusion mismatch")
        audit.require("images.aria" not in group, f"{label}: aria image unexpectedly present")

        values: dict[str, np.ndarray] = {}
        for key, width in NUMERIC.items():
            if key not in group:
                audit.errors.append(f"{label}: missing {key}")
                continue
            value = np.asarray(group[key][:frames])
            values[key] = value
            audit.require(value.shape == (frames, width), f"{label}: {key} shape {value.shape}")
            audit.require(np.all(np.isfinite(value)), f"{label}: {key} has NaN/Inf")
        for key in IMAGES:
            if key not in group:
                audit.errors.append(f"{label}: missing {key}")
                continue
            audit.require(group[key].shape[0] >= frames, f"{label}: {key} shorter than frames")
            check_image(group[key], 0, f"{label}/{key}[0]", audit)
            check_image(group[key], frames - 1, f"{label}/{key}[-1]", audit)
        if len(values) != len(NUMERIC):
            continue

        cmd = values["cmd_joint"]
        obs = values["obs_joint"]
        cmd_min = np.minimum(cmd_min, cmd.min(axis=0))
        cmd_max = np.maximum(cmd_max, cmd.max(axis=0))
        obs_min = np.minimum(obs_min, obs.min(axis=0))
        obs_max = np.maximum(obs_max, obs.max(axis=0))
        audit.require(np.all(cmd >= low - 1e-8) and np.all(cmd <= high + 1e-8), f"{label}: command outside calibration")
        tol = math.radians(1.0)
        audit.require(np.all(obs >= low - tol) and np.all(obs <= high + tol), f"{label}: feedback >1deg outside calibration")
        if frames >= 30:
            horizon_indices = np.arange(frames - 29)[:, None] + np.arange(30)[None, :]
            residual_deg = np.abs(
                np.rad2deg(cmd[horizon_indices] - obs[: frames - 29, None, :])
            )
            residual_max_deg = np.maximum(
                residual_max_deg, residual_deg.max(axis=(0, 1))
            )
            hpt_residual_over_limit += np.count_nonzero(
                residual_deg > 110.0, axis=(0, 1)
            )
            pi05_residual_over_limit += np.count_nonzero(
                residual_deg > pi05_residual_scale_deg[None, None, :], axis=(0, 1)
            )
        for pose_key in ("obs_ee_pose", "cmd_ee_pose"):
            norms = np.linalg.norm(values[pose_key][:, 3:7], axis=1)
            audit.require(float(np.max(np.abs(norms - 1.0))) <= 1e-3, f"{label}: {pose_key} quaternion norm")
        grip = values["cmd_gripper"][:, 0]
        audit.require(np.all(np.isin(grip, [0.0, 1.0])), f"{label}: non-binary gripper")
        n_transitions = int(np.count_nonzero(np.diff(grip)))
        transitions.append(n_transitions)
        audit.require(n_transitions == 6, f"{label}: gripper transitions={n_transitions}, expected 6")
        counts = Counter(int(v) for v in grip)
        grip_counts.update(counts)
        if split == "train":
            train_grip_counts.update(counts)
        clip = np.asarray(group.attrs.get("joint_percent_clip_counts"), dtype=np.int64)
        audit.require(clip.shape == (6,), f"{label}: invalid clip counts")
        if clip.shape == (6,):
            clip_counts += clip

    audit.require(sorted(indices) == list(range(100)), "source episode indices are not 0..99")
    audit.require(dict(splits) == EXPECTED_SPLITS, f"split counts {dict(splits)} != {EXPECTED_SPLITS}")
    audit.require(total_frames == EXPECTED_TOTAL_FRAMES, f"total frames {total_frames} != {EXPECTED_TOTAL_FRAMES}")
    audit.require(clip_counts[:4].sum() == 0 and clip_counts[5] == 0, f"unexpected clipping outside J5: {clip_counts.tolist()}")
    audit.require(int(clip_counts[4]) == 76, f"expected 76 clipped J5 samples, got {clip_counts[4]}")
    audit.require(
        int(hpt_residual_over_limit.sum()) == 0,
        f"HPT 110deg residual head cannot represent labels: {hpt_residual_over_limit.tolist()}",
    )
    audit.require(
        int(pi05_residual_over_limit.sum()) == 0,
        f"pi0.5 per-joint residual head cannot represent labels: {pi05_residual_over_limit.tolist()}",
    )

    human_paths = sorted(args.human_root.glob("*.zarr"))
    audit.require(len(human_paths) == 60, f"expected 60 human Zarr episodes, found {len(human_paths)}")
    human_splits: Counter[str] = Counter()
    human_source_indices: list[int] = []
    human_frames = 0
    human_max_timestamp_gap_ms = 0.0
    human_sentinel_gaze_frames = 0
    for path in human_paths:
        label = path.name
        try:
            group = zarr.open_group(str(path), mode="r")
            frames = int(group.attrs["total_frames"])
            split = str(group.attrs["dataset_split"])
            source_index = int(group.attrs["source_episode_index"])
        except Exception as exc:
            audit.errors.append(f"human/{label}: cannot read attrs: {exc}")
            continue
        human_splits[split] += 1
        human_source_indices.append(source_index)
        human_frames += frames
        audit.require(
            split == human_expected_by_index.get(str(source_index)),
            f"human/{label}: split mismatch for source episode {source_index}",
        )
        audit.require(
            group.attrs.get("embodiment") == "human_right_arm",
            f"human/{label}: embodiment is not human_right_arm",
        )
        audit.require(
            group.attrs.get("data_contract") == "aria_assembly_new1_right_gen2_v1",
            f"human/{label}: wrong data contract",
        )
        audit.require(
            group.attrs.get("trajectory_source") == "vrs_stream_371-4_vio_high_frequency",
            f"human/{label}: wrong trajectory source",
        )
        audit.require(
            group.attrs.get("trajectory_reference_frame") == "per_episode_odometry",
            f"human/{label}: wrong trajectory frame",
        )
        source_vrs_sha256 = str(group.attrs.get("source_vrs_sha256", ""))
        audit.require(
            len(source_vrs_sha256) == 64
            and all(character in "0123456789abcdef" for character in source_vrs_sha256),
            f"human/{label}: invalid source VRS SHA-256",
        )
        mano = dict(group.attrs.get("mano", {}))
        audit.require(
            mano.get("enabled") is True
            and mano.get("iters") == 2400
            and math.isclose(float(mano.get("lr", float("nan"))), 0.05)
            and math.isclose(float(mano.get("beta_reg", float("nan"))), 0.01),
            f"human/{label}: unexpected MANO conversion settings {mano}",
        )
        audit.require(frames > 30, f"human/{label}: length {frames} <= horizon 30")
        audit.require(
            not any(key.startswith("left.") for key in group.array_keys()),
            f"human/{label}: unexpected left-hand arrays",
        )
        human_values: dict[str, np.ndarray] = {}
        for key, width in HUMAN_NUMERIC.items():
            if key not in group:
                audit.errors.append(f"human/{label}: missing {key}")
                continue
            value = np.asarray(group[key][:frames])
            human_values[key] = value
            audit.require(value.shape == (frames, width), f"human/{label}: {key} shape {value.shape}")
            audit.require(np.all(np.isfinite(value)), f"human/{label}: {key} has NaN/Inf")
        if "obs_rgb_timestamps_ns" not in group:
            audit.errors.append(f"human/{label}: missing obs_rgb_timestamps_ns")
        else:
            timestamps = np.asarray(group["obs_rgb_timestamps_ns"][:frames], dtype=np.int64)
            audit.require(timestamps.shape == (frames,), f"human/{label}: timestamp shape {timestamps.shape}")
            gaps = np.diff(timestamps)
            audit.require(np.all(gaps > 0), f"human/{label}: RGB timestamps not strictly increasing")
            if gaps.size:
                gap_ms = float(gaps.max() / 1e6)
                human_max_timestamp_gap_ms = max(human_max_timestamp_gap_ms, gap_ms)
                audit.require(gap_ms <= 50.0, f"human/{label}: timestamp gap {gap_ms:.3f} ms > 50 ms")
        if "images.front_1" not in group:
            audit.errors.append(f"human/{label}: missing images.front_1")
        else:
            check_image(group["images.front_1"], 0, f"human/{label}/front[0]", audit)
            check_image(group["images.front_1"], frames - 1, f"human/{label}/front[-1]", audit)
        for pose_key in ("right.obs_ee_pose", "right.obs_wrist_pose", "obs_head_pose"):
            if pose_key in human_values:
                norms = np.linalg.norm(human_values[pose_key][:, 3:7], axis=1)
                audit.require(
                    float(np.max(np.abs(norms - 1.0))) <= 1e-3,
                    f"human/{label}: {pose_key} quaternion norm",
                )
        if "obs_eye_gaze" in human_values:
            human_sentinel_gaze_frames += int(
                np.count_nonzero(np.any(np.abs(human_values["obs_eye_gaze"]) >= 1e8, axis=1))
            )
    audit.require(
        dict(human_splits) == HUMAN_EXPECTED_SPLITS,
        f"human split counts {dict(human_splits)} != {HUMAN_EXPECTED_SPLITS}",
    )
    audit.require(
        sorted(human_source_indices) == list(range(1, 61)),
        "human source episode indices are not exactly 1..60",
    )
    if human_sentinel_gaze_frames:
        audit.warnings.append(
            f"human eye-gaze sentinel appears in {human_sentinel_gaze_frames} frames; current configs do not consume eye gaze"
        )

    closed = train_grip_counts[0]
    opened = train_grip_counts[1]
    train_total = closed + opened
    train_open_prior = opened / train_total if train_total else float("nan")
    class_weights = {
        "closed_weight": train_total / (2.0 * closed) if closed else float("inf"),
        "open_weight": train_total / (2.0 * opened) if opened else float("inf"),
        "open_prior": train_open_prior,
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_root": str(args.robot_root.resolve()),
        "human_root": str(args.human_root.resolve()),
        "ready_for_dataloader_smoke": not audit.errors,
        "formal_mixed_training_ready": not audit.errors and formal_extrinsics_accepted,
        "errors": audit.errors,
        "warnings": audit.warnings,
        "episodes": len(paths),
        "splits": dict(splits),
        "total_frames": total_frames,
        "trimmed_frames": {
            "total_removed": int(sum(removed)),
            "min": int(min(removed)) if removed else None,
            "median": float(np.median(removed)) if removed else None,
            "max": int(max(removed)) if removed else None,
        },
        "gripper_transition_counts": dict(Counter(transitions)),
        "gripper_counts_all": {"closed_0": grip_counts[0], "open_1": grip_counts[1]},
        "gripper_counts_train": {"closed_0": closed, "open_1": opened},
        "train_gripper_class_weights": class_weights,
        "joint_percent_clip_counts": clip_counts.tolist(),
        "cmd_joint_min_deg": np.rad2deg(cmd_min).tolist(),
        "cmd_joint_max_deg": np.rad2deg(cmd_max).tolist(),
        "obs_joint_min_deg": np.rad2deg(obs_min).tolist(),
        "obs_joint_max_deg": np.rad2deg(obs_max).tolist(),
        "native_30_step_residual_audit": {
            "max_abs_delta_deg_by_joint": residual_max_deg.tolist(),
            "hpt_scalar_scale_deg": 110.0,
            "hpt_count_over_scale_by_joint": hpt_residual_over_limit.tolist(),
            "pi05_scale_deg_by_joint": pi05_residual_scale_deg.tolist(),
            "pi05_count_over_scale_by_joint": pi05_residual_over_limit.tolist(),
        },
        "camera_keys": list(IMAGES),
        "aria_excluded": True,
        "phase_sampling": False,
        "extrinsics_status": extrinsics.get("status"),
        "formal_extrinsics_accepted": formal_extrinsics_accepted,
        "human": {
            "episodes": len(human_paths),
            "splits": dict(human_splits),
            "total_frames": human_frames,
            "max_rgb_timestamp_gap_ms": human_max_timestamp_gap_ms,
            "eye_gaze_sentinel_frames": human_sentinel_gaze_frames,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if audit.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
