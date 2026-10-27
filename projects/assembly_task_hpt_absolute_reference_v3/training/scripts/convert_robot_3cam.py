#!/usr/bin/env python3
"""Convert the new 100-episode LeRobot recording into trimmed 3-camera Zarr.

The converter is task-specific and intentionally excludes the robot-recorded
Aria stream.  It preserves the raw dataset, clips only the converted joint
commands to the calibration percentage contract, and records every alteration
in episode attributes plus a conversion manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import simplejpeg
import zarr
from PIL import Image

from egomimic.robot.piper.gripper_utils import binarize_gripper_commands
from egomimic.robot.piper.piper_kinematics import (
    PiperKinematicsSolver,
    raw_gripper_to_frac,
    raw_joint_to_rad,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter


FPS = 30
CAMERAS = {
    "observation.images.head": "images.front_1",
    "observation.images.wrist": "images.wrist_1",
    "observation.images.side": "images.side_1",
}
REQUIRED_ARRAYS = {
    "obs_joint": 6,
    "cmd_joint": 6,
    "obs_ee_pose": 7,
    "cmd_ee_pose": 7,
    "obs_gripper": 1,
    "cmd_gripper": 1,
    "images.front_1": None,
    "images.wrist_1": None,
    "images.side_1": None,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_episode_table(root: Path):
    paths = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError("no episode parquet files")
    return pa.concat_tables([pq.read_table(path) for path in paths]).to_pandas()


def load_joint_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    low_deg = np.asarray(
        [float(payload[f"joint_{index}"]["range_min"]) / 1000.0 for index in range(6)]
    )
    high_deg = np.asarray(
        [float(payload[f"joint_{index}"]["range_max"]) / 1000.0 for index in range(6)]
    )
    if np.any(high_deg <= low_deg):
        raise ValueError("invalid calibration joint ranges")
    return np.deg2rad(low_deg), np.deg2rad(high_deg)


def percent_to_joint_rad(
    percent: np.ndarray, low_rad: np.ndarray, high_rad: np.ndarray
) -> np.ndarray:
    return low_rad[None] + (high_rad - low_rad)[None] * (percent + 100.0) / 200.0


def trim_length(
    cmd_joint: np.ndarray,
    gripper_mm: np.ndarray,
    *,
    window: int = 5,
    joint_threshold_deg: float = 1.0,
    gripper_threshold_mm: float = 3.0,
    retain_after: int = 5,
) -> tuple[int, dict[str, Any]]:
    frames = len(cmd_joint)
    if frames <= window:
        return frames, {"last_active_index": frames - 1, "removed_frames": 0}
    joint_change = np.zeros(frames, dtype=np.float64)
    grip_change = np.zeros(frames, dtype=np.float64)
    joint_change[window:] = np.max(
        np.abs(np.rad2deg(cmd_joint[window:] - cmd_joint[:-window])), axis=1
    )
    grip_change[window:] = np.abs(gripper_mm[window:] - gripper_mm[:-window])
    active = (joint_change >= joint_threshold_deg) | (
        grip_change >= gripper_threshold_mm
    )
    indices = np.flatnonzero(active)
    last_active = int(indices[-1]) if len(indices) else frames - 1
    keep = min(frames, last_active + 1 + retain_after)
    return keep, {
        "window_frames": window,
        "joint_threshold_deg": joint_threshold_deg,
        "gripper_threshold_mm": gripper_threshold_mm,
        "retain_after_last_motion_frames": retain_after,
        "last_active_index": last_active,
        "removed_frames": frames - keep,
    }


def decode_range_to_jpeg(
    video_path: Path,
    *,
    from_timestamp: float,
    expected_frames: int,
    jpeg_quality: int,
) -> tuple[np.ndarray, list[int], dict[str, float]]:
    """Seek near an episode and return exactly expected_frames JPEG payloads."""
    encoded = np.empty((expected_frames,), dtype=object)
    accepted_times: list[float] = []
    cursor = 0
    tolerance = 0.5 / FPS + 1e-4
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        seek_s = max(0.0, float(from_timestamp) - 2.0)
        container.seek(int(seek_s / float(stream.time_base)), stream=stream, backward=True)
        image_shape: list[int] | None = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            if timestamp + tolerance < from_timestamp:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            if image_shape is None:
                image_shape = list(rgb.shape)
            elif list(rgb.shape) != image_shape:
                raise ValueError(f"video frame shape changed in {video_path}")
            encoded[cursor] = simplejpeg.encode_jpeg(
                rgb, quality=jpeg_quality, colorspace="RGB"
            )
            accepted_times.append(timestamp)
            cursor += 1
            if cursor == expected_frames:
                break
    if cursor != expected_frames:
        raise ValueError(
            f"{video_path}: decoded {cursor} frames, expected {expected_frames} "
            f"from timestamp {from_timestamp}"
        )
    if image_shape != [480, 640, 3]:
        raise ValueError(f"{video_path}: image shape {image_shape}, expected [480,640,3]")
    expected_last = from_timestamp + (expected_frames - 1) / FPS
    if abs(accepted_times[0] - from_timestamp) > 1.5 / FPS:
        raise ValueError(
            f"{video_path}: first decoded timestamp {accepted_times[0]} does not match "
            f"episode start {from_timestamp}"
        )
    if abs(accepted_times[-1] - expected_last) > 1.5 / FPS:
        raise ValueError(
            f"{video_path}: last decoded timestamp {accepted_times[-1]} does not match "
            f"expected {expected_last}"
        )
    return encoded, image_shape, {
        "first_timestamp": accepted_times[0],
        "last_timestamp": accepted_times[-1],
    }


def valid_existing(path: Path, source_index: int, split: str) -> bool:
    if not path.exists():
        return False
    try:
        group = zarr.open_group(str(path), mode="r")
        frames = int(group.attrs["total_frames"])
        if int(group.attrs.get("source_episode_index", -1)) != source_index:
            return False
        if group.attrs.get("dataset_split") != split:
            return False
        if group.attrs.get("data_contract") != "assembly_task_new1_3cam_v2":
            return False
        for key, width in REQUIRED_ARRAYS.items():
            if key not in group or group[key].shape[0] < frames:
                return False
            if width is not None and group[key].shape[-1] != width:
                return False
        return "images.aria" not in group
    except Exception:
        return False


def verify_jpeg(payload: object, label: str) -> None:
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"{label}: not bytes")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.size != (640, 480) or image.mode != "RGB":
            raise ValueError(f"{label}: decoded {image.size}/{image.mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--intrinsics-json", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    split_payload = json.loads(args.split_json.read_text())
    split_by_episode = split_payload["split_by_episode"]
    intrinsics = json.loads(args.intrinsics_json.read_text())
    if set(intrinsics) != {"front_1", "wrist_1"}:
        raise ValueError("intrinsics JSON must contain exactly front_1 and wrist_1")
    low_rad, high_rad = load_joint_calibration(args.calibration_json)
    calibration_sha = sha256(args.calibration_json)

    data_paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    data_df = pa.concat_tables([pq.read_table(path) for path in data_paths]).to_pandas()
    episodes = load_episode_table(root)
    if len(episodes) != 100:
        raise ValueError(f"expected 100 episodes, found {len(episodes)}")
    if args.episodes is not None:
        episodes = episodes[episodes["episode_index"].isin(args.episodes)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fk = PiperKinematicsSolver()
    converted: list[dict[str, Any]] = []
    skipped: list[str] = []

    for _, row in episodes.sort_values("episode_index").iterrows():
        episode_index = int(row["episode_index"])
        split = str(split_by_episode[str(episode_index)])
        name = f"piper-assembly-new1-{split}-ep{episode_index:04d}.zarr"
        output = args.out_dir / name
        if valid_existing(output, episode_index, split) and not args.overwrite:
            skipped.append(name)
            continue
        if output.exists():
            if not args.overwrite:
                raise FileExistsError(f"incomplete/incompatible output exists: {output}")
            shutil.rmtree(output)

        start = int(row["dataset_from_index"])
        stop = int(row["dataset_to_index"])
        episode_rows = data_df.iloc[start:stop]
        state = np.stack(episode_rows["observation.state"].to_numpy()).astype(np.float64)
        raw_action = np.stack(episode_rows["action"].to_numpy()).astype(np.float64)
        if state.shape != raw_action.shape or state.shape[1] != 7:
            raise ValueError(f"episode {episode_index}: invalid state/action shapes")

        unclipped_joint_percent = raw_action[:, :6]
        joint_percent = np.clip(unclipped_joint_percent, -100.0, 100.0)
        gripper_mm = np.clip(raw_action[:, 6], 0.0, 100.0)
        cmd_joint = percent_to_joint_rad(joint_percent, low_rad, high_rad)
        keep, trim = trim_length(cmd_joint, gripper_mm)
        if keep <= 30:
            raise ValueError(f"episode {episode_index}: trimmed length {keep} <= horizon 30")

        obs_joint = raw_joint_to_rad(state[:keep, :6]).astype(np.float64)
        obs_gripper = raw_gripper_to_frac(state[:keep, 6:7]).astype(np.float64)
        cmd_joint = cmd_joint[:keep].astype(np.float64)
        cmd_gripper = binarize_gripper_commands(
            np.clip(raw_action[:keep, 6:7] / 100.0, 0.0, 1.0)
        ).astype(np.float32)
        obs_ee_pose = fk.fk(obs_joint).astype(np.float64)
        cmd_ee_pose = fk.fk(cmd_joint).astype(np.float64)

        pre_encoded: dict[str, tuple[np.ndarray, list[int]]] = {}
        video_audit: dict[str, Any] = {}
        for lerobot_key, zarr_key in CAMERAS.items():
            prefix = f"videos/{lerobot_key}"
            chunk = int(row[f"{prefix}/chunk_index"])
            file_index = int(row[f"{prefix}/file_index"])
            from_timestamp = float(row[f"{prefix}/from_timestamp"])
            video_path = (
                root
                / "videos"
                / lerobot_key
                / f"chunk-{chunk:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            encoded, image_shape, timing = decode_range_to_jpeg(
                video_path,
                from_timestamp=from_timestamp,
                expected_frames=keep,
                jpeg_quality=args.jpeg_quality,
            )
            verify_jpeg(encoded[0], f"episode {episode_index}/{zarr_key}[0]")
            verify_jpeg(encoded[-1], f"episode {episode_index}/{zarr_key}[-1]")
            pre_encoded[zarr_key] = (encoded, image_shape)
            video_audit[zarr_key] = {
                "source_key": lerobot_key,
                "source_video": str(video_path.relative_to(root)),
                **timing,
            }

        numeric = {
            "obs_joint": obs_joint,
            "cmd_joint": cmd_joint,
            "obs_ee_pose": obs_ee_pose,
            "cmd_ee_pose": cmd_ee_pose,
            "obs_gripper": obs_gripper,
            "cmd_gripper": cmd_gripper,
        }
        ZarrWriter.create_and_write(
            episode_path=output,
            numeric_data=numeric,
            pre_encoded_image_data=pre_encoded,
            embodiment="piper_left_arm",
            fps=FPS,
            task_name=f"assembly_task_new1_{split}",
            task_description="Complete the assembly task.",
            annotations=[],
            intrinsics=intrinsics,
            extrinsics=None,
        )
        group = zarr.open_group(str(output), mode="r+")
        clip_counts = np.count_nonzero(
            unclipped_joint_percent != joint_percent, axis=0
        ).astype(int)
        group.attrs.update(
            {
                "dataset_split": split,
                "source_dataset": str(root),
                "source_episode_index": episode_index,
                "source_frames": int(len(state)),
                "total_frames": int(keep),
                "data_contract": "assembly_task_new1_3cam_v2",
                "joint_space_encoding": "six_absolute_joint_angles_radians_plus_binary_gripper",
                "obs_gripper_encoding": "normalized_continuous_aperture_0_closed_1_open",
                "cmd_gripper_encoding": "0_closed_1_open_hysteresis_0.2_0.8",
                "joint_percent_clip_range": [-100.0, 100.0],
                "joint_percent_clip_counts": clip_counts.tolist(),
                "calibration_sha256": calibration_sha,
                "camera_mapping": CAMERAS,
                "excluded_source_cameras": ["observation.images.aria"],
                "side_intrinsics_status": "unavailable_not_consumed_by_image_model",
                "phase_sampling": False,
                "trim": trim,
                "video_decode_audit": video_audit,
                "converted_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        converted.append(
            {
                "episode_index": episode_index,
                "output": name,
                "split": split,
                "source_frames": int(len(state)),
                "kept_frames": int(keep),
                "removed_frames": int(len(state) - keep),
                "joint_percent_clip_counts": clip_counts.tolist(),
            }
        )
        print(
            f"episode {episode_index:03d} {split}: {len(state)} -> {keep} frames, "
            f"clipped={clip_counts.tolist()}"
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(root),
        "source_info_sha256": sha256(root / "meta" / "info.json"),
        "split_manifest_sha256": sha256(args.split_json),
        "calibration_sha256": calibration_sha,
        "intrinsics_sha256": sha256(args.intrinsics_json),
        "camera_mapping": CAMERAS,
        "excluded_source_cameras": ["observation.images.aria"],
        "side_intrinsics_status": "unavailable_not_consumed_by_image_model",
        "phase_sampling": False,
        "converted": converted,
        "skipped_complete": skipped,
    }
    (args.out_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"converted": len(converted), "skipped": len(skipped)}, indent=2))


if __name__ == "__main__":
    main()
