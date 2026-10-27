#!/usr/bin/env python3
"""Read-only readiness check for the three Piper/human vision variants."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import zarr


ROBOT_TRAIN = {f"teddy_0912_{i}" for i in range(1, 6)}
ROBOT_VALID = {"teddy_0912_6"}
HUMAN_TRAIN = {"put_bear_0919_train"}
HUMAN_VALID = {"put_bear_0919_valid"}


def _episode_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory is missing: {root}")
    paths = sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith(".zarr"))
    if not paths:
        raise RuntimeError(f"No .zarr episodes found in {root}")
    return paths


def _task_name(store: zarr.Group) -> str:
    return str(store.attrs.get("task_name", ""))


def _check_embedding(store: zarr.Group, path: Path, key: str, total_frames: int) -> None:
    if key not in store:
        raise KeyError(f"{path.name}: missing {key}")
    arr = store[key]
    if arr.ndim != 3 or tuple(arr.shape[1:]) != (196, 1024):
        raise ValueError(
            f"{path.name}:{key} expected (T,196,1024), got {tuple(arr.shape)}"
        )
    if arr.shape[0] < total_frames:
        raise ValueError(
            f"{path.name}:{key} has {arr.shape[0]} rows but total_frames={total_frames}"
        )
    if np.dtype(arr.dtype) not in (np.dtype("float16"), np.dtype("float32")):
        raise TypeError(f"{path.name}:{key} expected float16/float32, got {arr.dtype}")
    probes = sorted({0, max(0, total_frames // 2), max(0, total_frames - 1)})
    values = np.asarray(arr[probes], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}:{key} contains NaN/Inf in sampled rows {probes}")


def _check_root(
    root: Path,
    *,
    allowed_tasks: set[str],
    required_embeddings: tuple[str, ...],
) -> dict:
    counts: Counter[str] = Counter()
    selected = 0
    frames = 0
    dtypes: Counter[str] = Counter()
    for path in _episode_dirs(root):
        store = zarr.open_group(str(path), mode="r")
        task_name = _task_name(store)
        counts[task_name] += 1
        if task_name not in allowed_tasks:
            continue
        selected += 1
        total_frames = int(store.attrs.get("total_frames", 0))
        if total_frames <= 0:
            raise ValueError(f"{path.name}: invalid total_frames={total_frames}")
        frames += total_frames
        for key in required_embeddings:
            _check_embedding(store, path, key, total_frames)
            dtypes[str(store[key].dtype)] += 1

    missing_tasks = allowed_tasks - set(counts)
    if missing_tasks:
        raise RuntimeError(f"{root}: missing required task_name values {sorted(missing_tasks)}")
    if selected == 0:
        raise RuntimeError(f"{root}: no train/validation episodes selected")
    return {
        "root": str(root.resolve()),
        "selected_episodes": selected,
        "selected_frames": frames,
        "task_counts": dict(sorted(counts.items())),
        "embedding_dtypes": dict(sorted(dtypes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("resnet18", "dinov3_front", "dinov3_both"), required=True
    )
    parser.add_argument("--robot-root", type=Path, required=True)
    parser.add_argument("--human-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.variant == "resnet18":
        robot_embeddings: tuple[str, ...] = ()
        human_embeddings: tuple[str, ...] = ()
    elif args.variant == "dinov3_front":
        robot_embeddings = ("dino.front_1",)
        human_embeddings = ("dino.front_1",)
    else:
        robot_embeddings = ("dino.front_1", "dino.wrist_1")
        human_embeddings = ("dino.front_1",)

    report = {
        "variant": args.variant,
        "selection_policy": "train + validation only; held-out test excluded",
        "robot": _check_root(
            args.robot_root,
            allowed_tasks=ROBOT_TRAIN | ROBOT_VALID,
            required_embeddings=robot_embeddings,
        ),
        "human": _check_root(
            args.human_root,
            allowed_tasks=HUMAN_TRAIN | HUMAN_VALID,
            required_embeddings=human_embeddings,
        ),
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n")


if __name__ == "__main__":
    main()
