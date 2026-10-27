#!/usr/bin/env python3
"""Plot and audit right-hand trajectories in the training camera frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.spatial.transform import Rotation


def camera_frame_pose(hand: np.ndarray, head: np.ndarray) -> tuple[np.ndarray, Rotation]:
    head_rotation = Rotation.from_quat(head[:, 3:7])
    hand_rotation = Rotation.from_quat(hand[:, 3:7])
    position = head_rotation.inv().apply(hand[:, :3] - head[:, :3])
    rotation = head_rotation.inv() * hand_rotation
    return position, rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-root", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[1, 7, 10, 20, 25, 30, 40, 50, 55, 60],
    )
    args = parser.parse_args()

    paths = sorted(args.human_root.glob("*.zarr"))
    if len(paths) != 60:
        raise ValueError(f"expected 60 episodes, found {len(paths)}")
    selected = set(args.episodes)
    figure, axes = plt.subplots(5, 2, figsize=(15, 16), constrained_layout=True)
    axes_by_episode = dict(zip(args.episodes, axes.flat, strict=True))
    audit: dict[str, dict] = {}

    for path in paths:
        group = zarr.open_group(str(path), mode="r")
        episode = int(group.attrs["source_episode_index"])
        frames = int(group.attrs["total_frames"])
        hand = np.asarray(group["right.obs_ee_pose"][:frames], dtype=np.float64)
        head = np.asarray(group["obs_head_pose"][:frames], dtype=np.float64)
        position, rotation = camera_frame_pose(hand, head)

        window = 5
        translation_mm = np.linalg.norm(position[window:] - position[:-window], axis=1) * 1000.0
        rotation_deg = (rotation[:-window].inv() * rotation[window:]).magnitude()
        rotation_deg = np.rad2deg(rotation_deg)
        active = (translation_mm >= 3.0) | (rotation_deg >= 1.0)
        active_indices = np.flatnonzero(active)
        first_active = int(active_indices[0]) if active_indices.size else None
        last_active = int(active_indices[-1] + window) if active_indices.size else None
        audit[str(episode)] = {
            "split": str(group.attrs["dataset_split"]),
            "frames": frames,
            "duration_s": frames / 30.0,
            "prefix_before_motion_frames": first_active,
            "tail_after_motion_frames": None if last_active is None else frames - 1 - last_active,
            "median_5frame_translation_mm": float(np.median(translation_mm)),
            "median_5frame_rotation_deg": float(np.median(rotation_deg)),
        }

        if episode in selected:
            axis = axes_by_episode[episode]
            time_s = np.arange(frames) / 30.0
            delta_mm = (position - position[0]) * 1000.0
            axis.plot(time_s, delta_mm[:, 0], label="x", linewidth=1.1)
            axis.plot(time_s, delta_mm[:, 1], label="y", linewidth=1.1)
            axis.plot(time_s, delta_mm[:, 2], label="z", linewidth=1.1)
            axis.set_title(
                f"episode {episode:02d} | {group.attrs['dataset_split']} | {frames} frames"
            )
            axis.set_xlabel("time (s)")
            axis.set_ylabel("camera-frame hand displacement (mm)")
            axis.grid(alpha=0.25)
            axis.legend(ncol=3, fontsize=8)

    missing = selected - set(map(int, audit))
    if missing:
        raise ValueError(f"requested episodes missing: {sorted(missing)}")
    prefix = [item["prefix_before_motion_frames"] for item in audit.values()]
    tail = [item["tail_after_motion_frames"] for item in audit.values()]
    summary = {
        "criterion": "active if camera-frame right-hand changes >=3mm or >=1deg over 5 frames",
        "episodes": audit,
        "summary": {
            "max_prefix_before_motion_frames": max(value for value in prefix if value is not None),
            "median_prefix_before_motion_frames": float(np.median(prefix)),
            "max_tail_after_motion_frames": max(value for value in tail if value is not None),
            "median_tail_after_motion_frames": float(np.median(tail)),
        },
    }
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle("Assembly New1 right-hand trajectories in the model camera frame", fontsize=16)
    figure.savefig(args.output_png, dpi=170)
    plt.close(figure)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
