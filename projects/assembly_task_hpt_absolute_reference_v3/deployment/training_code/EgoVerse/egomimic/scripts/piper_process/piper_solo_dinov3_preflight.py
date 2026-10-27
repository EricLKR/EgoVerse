#!/usr/bin/env python3
"""Read-only raw-data readiness check for online Piper-only DINOv3 runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import simplejpeg
import zarr


SPLIT_TASKS = {
    "train_valid": {f"teddy_0912_{i}" for i in range(1, 7)},
    "test": {"teddy_0912_7"},
    "all": {f"teddy_0912_{i}" for i in range(1, 8)},
}
EXPECTED = {
    "train_valid": {"episodes": 60, "frames": 21493},
    "test": {"episodes": 10, "frames": 3294},
    "all": {"episodes": 70, "frames": 24787},
}
IMAGE_KEYS = ("images.front_1", "images.wrist_1")
NUMERIC_KEYS = ("obs_ee_pose", "obs_gripper", "cmd_gripper")


def _episode_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory is missing: {root}")
    paths = sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith(".zarr"))
    if not paths:
        raise RuntimeError(f"No .zarr episodes found in {root}")
    return paths


def _probes(frames: int) -> list[int]:
    return sorted({0, frames // 2, frames - 1})


def _check_image(store: zarr.Group, path: Path, key: str, frames: int) -> int:
    if key not in store:
        raise KeyError(f"{path.name}: missing {key}")
    arr = store[key]
    length = int(arr.shape[0])
    if length < frames:
        raise ValueError(f"{path.name}:{key} length {length} < total_frames {frames}")
    for index in _probes(frames):
        try:
            # Match MultiDataset's slice-based read for Zarr variable-length
            # JPEG arrays. Direct scalar indexing can expose the encoded
            # scalar wrapper rather than its payload under Zarr v3.
            payload = arr[index : index + 1][0]
            image = simplejpeg.decode_jpeg(bytes(payload), colorspace="RGB")
        except Exception as exc:
            raise ValueError(f"{path.name}:{key}[{index}] JPEG decode failed") from exc
        if image.ndim != 3 or image.shape[-1] != 3 or image.size == 0:
            raise ValueError(
                f"{path.name}:{key}[{index}] expected nonempty HWC RGB, got {image.shape}"
            )
    return length - frames


def _check_numeric(store: zarr.Group, path: Path, key: str, frames: int) -> int:
    if key not in store:
        raise KeyError(f"{path.name}: missing {key}")
    arr = store[key]
    length = int(arr.shape[0])
    if length < frames:
        raise ValueError(f"{path.name}:{key} length {length} < total_frames {frames}")
    values = np.asarray(arr[_probes(frames)], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}:{key} contains NaN/Inf in sampled rows")
    return length - frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("dinov3_front", "dinov3_both"), required=True)
    parser.add_argument("--robot-root", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(SPLIT_TASKS), default="train_valid")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    selected_tasks = SPLIT_TASKS[args.split]
    task_counts: Counter[str] = Counter()
    episodes = 0
    frames = 0
    decoded_images = 0
    padded_rows: Counter[str] = Counter()
    for path in _episode_dirs(args.robot_root):
        store = zarr.open_group(str(path), mode="r")
        task_name = str(store.attrs.get("task_name", ""))
        task_counts[task_name] += 1
        if task_name not in selected_tasks:
            continue
        episode_frames = int(store.attrs.get("total_frames", 0))
        if episode_frames <= 0:
            raise ValueError(f"{path.name}: invalid total_frames={episode_frames}")
        for key in IMAGE_KEYS:
            padded_rows[key] += _check_image(store, path, key, episode_frames)
            decoded_images += len(_probes(episode_frames))
        for key in NUMERIC_KEYS:
            padded_rows[key] += _check_numeric(store, path, key, episode_frames)
        episodes += 1
        frames += episode_frames

    missing_tasks = selected_tasks - set(task_counts)
    if missing_tasks:
        raise RuntimeError(f"Missing required task_name values: {sorted(missing_tasks)}")
    expected = EXPECTED[args.split]
    if episodes != expected["episodes"] or frames != expected["frames"]:
        raise RuntimeError(
            f"Unexpected {args.split} coverage: episodes={episodes}, frames={frames}; "
            f"expected episodes={expected['episodes']}, frames={expected['frames']}"
        )

    report = {
        "status": "PASS",
        "variant": args.variant,
        "encoder_mode": "online_frozen_dinov3",
        "split": args.split,
        "selection_policy": "session 6 validation only; session 7 held out",
        "robot_root": str(args.robot_root.resolve()),
        "selected_tasks": sorted(selected_tasks),
        "selected_episodes": episodes,
        "selected_frames": frames,
        "required_raw_images": list(IMAGE_KEYS),
        "required_numeric_arrays": list(NUMERIC_KEYS),
        "required_embeddings": [],
        "sampled_jpegs_decoded": decoded_images,
        "ignored_padding_rows": dict(sorted(padded_rows.items())),
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n")


if __name__ == "__main__":
    main()
