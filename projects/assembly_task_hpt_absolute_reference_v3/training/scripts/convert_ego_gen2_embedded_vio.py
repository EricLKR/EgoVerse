#!/usr/bin/env python3
"""Convert the 60 right-hand Aria Gen2 episodes with embedded VIO to Zarr.

This is intentionally separate from the legacy cloud-MPS converter.  It reads
the Gen2 ``vio_high_frequency`` stream (371-4) directly from each VRS, uses the
per-episode odometry frame as the local world frame, and records complete
provenance in every output episode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId

from egomimic.rldb.embodiment.human import ARIA_INTRINSICS
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.aria_process.aria_utils import (
    AriaVRSExtractor,
    convert_aria_keypoints_to_mano,
    mano_keypoints_to_cartesian,
    slam_to_rgb,
)


RGB_STREAM = StreamId("214-1")
VIO_STREAM = StreamId("371-4")
SLAM_LEFT_STREAM = StreamId("1201-1")
SLAM_RIGHT_STREAM = StreamId("1201-2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _deduplicated_hand_csv(source: Path, temp_dir: Path) -> tuple[Path, int]:
    """Stable timestamp de-duplication; keep highest right-hand confidence."""
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "tracking_timestamp_us" not in fieldnames:
            raise ValueError(f"unexpected hand CSV schema: {source}")
        rows = list(reader)

    best: dict[int, tuple[int, dict[str, str]]] = {}
    for order, row in enumerate(rows):
        timestamp = int(row["tracking_timestamp_us"])
        incumbent = best.get(timestamp)
        confidence = float(row.get("right_tracking_confidence", "-1"))
        if incumbent is None:
            best[timestamp] = (order, row)
        else:
            old_confidence = float(incumbent[1].get("right_tracking_confidence", "-1"))
            if confidence > old_confidence:
                best[timestamp] = (order, row)

    duplicate_count = len(rows) - len(best)
    if duplicate_count == 0:
        return source, 0

    output = temp_dir / source.name
    selected = [item[1] for item in sorted(best.values(), key=lambda item: int(item[1]["tracking_timestamp_us"]))]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return output, duplicate_count


def _load_embedded_vio(provider) -> tuple[list, dict[str, float | int]]:
    timestamps_ns = provider.get_timestamps_ns(VIO_STREAM, TimeDomain.DEVICE_TIME)
    if not timestamps_ns:
        raise RuntimeError("VRS contains no embedded high-frequency VIO samples on stream 371-4")
    trajectory = []
    quality = []
    for index in range(len(timestamps_ns)):
        pose = provider.get_vio_high_freq_data_by_index(VIO_STREAM, index)
        if pose is None:
            continue
        trajectory.append(pose)
        quality.append(float(pose.quality_score))
    if len(trajectory) < 2:
        raise RuntimeError(f"only {len(trajectory)} usable embedded VIO samples")
    pose_timestamps = np.asarray(
        [int(pose.tracking_timestamp.total_seconds() * 1e9) for pose in trajectory],
        dtype=np.int64,
    )
    if np.any(np.diff(pose_timestamps) <= 0):
        raise RuntimeError("embedded VIO timestamps are not strictly increasing")
    return trajectory, {
        "samples": len(trajectory),
        "quality_min": float(np.min(quality)),
        "quality_median": float(np.median(quality)),
        "quality_max": float(np.max(quality)),
    }


def extract_features(
    vrs_path: Path,
    hand_csv_path: Path,
    *,
    mano_iters: int,
    mano_lr: float,
    mano_beta_reg: float,
    mano_chunk_size: int,
    mano_device: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"failed to open VRS: {vrs_path}")

    with tempfile.TemporaryDirectory(prefix="aria-hand-dedup-") as temp_name:
        effective_hand_csv, duplicates = _deduplicated_hand_csv(hand_csv_path, Path(temp_name))
        hand_tracking = mps.hand_tracking.read_hand_tracking_results(str(effective_hand_csv))
        trajectory, vio_stats = _load_embedded_vio(provider)

        stream_ids = {
            "rgb": RGB_STREAM,
            "slam-left": SLAM_LEFT_STREAM,
            "slam-right": SLAM_RIGHT_STREAM,
        }
        rgb_timestamps = provider.get_timestamps_ns(RGB_STREAM, TimeDomain.DEVICE_TIME)
        if not rgb_timestamps:
            raise RuntimeError("VRS contains no RGB frames on stream 214-1")
        stream_timestamps = {"rgb": rgb_timestamps}
        device_rgb = slam_to_rgb(provider, height=480, width=640).inverse()

        cartesian = AriaVRSExtractor.get_ee_pose(
            world_device_T=trajectory,
            stream_timestamps_ns=stream_timestamps,
            hand_tracking_results=hand_tracking,
            arm="right",
        )
        keypoints = AriaVRSExtractor.get_hand_keypoints(
            world_device_T=trajectory,
            stream_timestamps_ns=stream_timestamps,
            hand_tracking_results=hand_tracking,
            arm="right",
        )
        head_pose = AriaVRSExtractor.get_head_pose(
            world_device_T=trajectory,
            device_rgb_T=device_rgb,
            stream_timestamps_ns=stream_timestamps,
        )
        images = AriaVRSExtractor.get_images(
            vrs_reader=provider,
            stream_ids=stream_ids,
            stream_timestamps_ns=stream_timestamps,
            height=480,
            width=640,
        )

    (filtered, passthrough) = AriaVRSExtractor.clean_data(
        filter_mask_data=[cartesian, keypoints, head_pose],
        mask_data=[images, np.asarray(rgb_timestamps, dtype=np.int64)],
    )
    cartesian, keypoints, head_pose = filtered
    images, rgb_timestamps = passthrough
    if len(images) <= 30:
        raise RuntimeError(f"only {len(images)} valid RGB/hand/VIO frames remain")

    raw_keypoints = keypoints[:, 7 : 7 + 63]
    mano_keypoints = convert_aria_keypoints_to_mano(
        raw_keypoints,
        is_rhand=True,
        device=mano_device,
        n_iters=mano_iters,
        lr=mano_lr,
        beta_reg=mano_beta_reg,
        chunk_size=mano_chunk_size,
    )
    mano_ee_pose = mano_keypoints_to_cartesian(mano_keypoints, is_rhand=True)
    valid_mano = np.isfinite(mano_ee_pose).all(axis=1) & np.all(np.abs(mano_ee_pose) < 1e8, axis=1)
    if not np.all(valid_mano):
        cartesian = cartesian[valid_mano]
        keypoints = keypoints[valid_mano]
        head_pose = head_pose[valid_mano]
        images = images[valid_mano]
        rgb_timestamps = rgb_timestamps[valid_mano]
        raw_keypoints = raw_keypoints[valid_mano]
        mano_keypoints = mano_keypoints[valid_mano]
        mano_ee_pose = mano_ee_pose[valid_mano]
    if len(images) <= 30:
        raise RuntimeError(f"only {len(images)} valid frames remain after MANO fit")

    features = {
        "right.obs_ee_pose": mano_ee_pose.astype(np.float64),
        "right.obs_wrist_pose": keypoints[:, :7].astype(np.float64),
        "right.obs_aria_keypoints": raw_keypoints.astype(np.float64),
        "right.obs_keypoints": mano_keypoints.astype(np.float64),
        "obs_head_pose": head_pose.astype(np.float64),
        "obs_rgb_timestamps_ns": np.asarray(rgb_timestamps, dtype=np.int64),
        "images.front_1": np.asarray(images, dtype=np.uint8),
    }
    stats: dict[str, object] = {
        "raw_rgb_frames": len(provider.get_timestamps_ns(RGB_STREAM, TimeDomain.DEVICE_TIME)),
        "valid_frames": len(images),
        "dropped_frames": len(provider.get_timestamps_ns(RGB_STREAM, TimeDomain.DEVICE_TIME)) - len(images),
        "deduplicated_hand_rows": duplicates,
        "embedded_vio": vio_stats,
    }
    return features, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, action="append")
    parser.add_argument("--mano-iters", type=int, default=2400)
    parser.add_argument("--mano-lr", type=float, default=0.05)
    parser.add_argument("--mano-beta-reg", type=float, default=0.01)
    parser.add_argument("--mano-chunk-size", type=int, default=128)
    parser.add_argument("--mano-device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(newline="")))
    split_payload = json.loads(args.split_json.read_text())
    selected = set(args.episode or [int(row["episode"]) for row in rows])
    if not selected.issubset(set(range(1, 61))):
        raise ValueError(f"invalid episode selection: {sorted(selected)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "conversion_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "episodes": {},
    }

    for row in rows:
        episode = int(row["episode"])
        if episode not in selected:
            continue
        vrs_path = args.raw_root / row["vrs_file"]
        hand_path = args.raw_root / row["mps_hand_file"]
        output = args.output_root / f"aria-assembly-new1-right-ep{episode:04d}.zarr"
        if output.exists() and not args.overwrite:
            group = zarr.open_group(str(output), mode="r")
            if (
                group.attrs.get("source_vrs_sha256") == row["vrs_sha256"]
                and group.attrs.get("data_contract") == "aria_assembly_new1_right_gen2_v1"
            ):
                print(f"[skip] episode {episode:03d}: {output.name}")
                continue
            raise RuntimeError(f"existing output fails provenance check: {output}")
        actual_sha = sha256(vrs_path)
        if actual_sha != row["vrs_sha256"]:
            raise RuntimeError(f"episode {episode}: VRS SHA mismatch")

        print(f"[convert] episode {episode:03d}: {vrs_path.name}", flush=True)
        started = time.monotonic()
        features, stats = extract_features(
            vrs_path,
            hand_path,
            mano_iters=args.mano_iters,
            mano_lr=args.mano_lr,
            mano_beta_reg=args.mano_beta_reg,
            mano_chunk_size=args.mano_chunk_size,
            mano_device=args.mano_device,
        )
        split = split_payload["split_by_episode"][str(episode)]
        numeric = {key: value for key, value in features.items() if not key.startswith("images.")}
        images = {key: value for key, value in features.items() if key.startswith("images.")}
        ZarrWriter.create_and_write(
            episode_path=output,
            numeric_data=numeric,
            image_data=images,
            embodiment="human_right_arm",
            fps=30,
            task_name="assembly_task_new1",
            task_description="Complete the assembly task.",
            chunk_timesteps=100,
            intrinsics={"front_1": ARIA_INTRINSICS},
            metadata_override={
                "dataset_split": split,
                "data_contract": "aria_assembly_new1_right_gen2_v1",
                "source_episode_index": episode,
                "source_uuid": row["uuid"],
                "source_vrs_file": row["vrs_file"],
                "source_vrs_sha256": actual_sha,
                "source_hand_file": row["mps_hand_file"],
                "trajectory_source": "vrs_stream_371-4_vio_high_frequency",
                "trajectory_reference_frame": "per_episode_odometry",
                "hand_side": "right",
                "mano": {
                    "enabled": True,
                    "iters": args.mano_iters,
                    "lr": args.mano_lr,
                    "beta_reg": args.mano_beta_reg,
                    "chunk_size": args.mano_chunk_size,
                },
                "conversion_stats": stats,
            },
        )
        elapsed = round(time.monotonic() - started, 3)
        status["episodes"][str(episode)] = {
            "state": "complete",
            "output": str(output),
            "split": split,
            "elapsed_seconds": elapsed,
            "stats": stats,
        }
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"[done] episode {episode:03d}: {stats['valid_frames']} frames in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
