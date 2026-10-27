#!/usr/bin/env python3
"""Plot ten deterministic train episodes after trimming for visual QA."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = []
    for path in sorted(args.robot_root.glob("*.zarr")):
        group = zarr.open_group(str(path), mode="r")
        if group.attrs.get("dataset_split") == "train":
            train.append(path)
    if len(train) != 90:
        raise ValueError(f"expected 90 train episodes, found {len(train)}")
    chosen = random.Random(args.seed).sample(train, 10)

    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(5, 2, figsize=(17, 18))
    for axis, path in zip(axes.flat, chosen, strict=True):
        group = zarr.open_group(str(path), mode="r")
        frames = int(group.attrs["total_frames"])
        joints = np.rad2deg(np.asarray(group["cmd_joint"][:frames]))
        relative = joints - joints[0]
        time_s = np.arange(frames) / float(group.attrs["fps"])
        for joint in range(6):
            axis.plot(
                time_s,
                relative[:, joint],
                color=colors[joint],
                linewidth=1.0,
                label=f"J{joint + 1}",
            )
        twin = axis.twinx()
        gripper = np.asarray(group["cmd_gripper"][:frames, 0])
        twin.step(time_s, gripper, where="post", color="black", linewidth=1.1, alpha=0.7)
        twin.set_ylim(-0.08, 1.08)
        twin.set_yticks([0, 1], ["close", "open"])
        removed = int(group.attrs["source_frames"]) - frames
        axis.set_title(
            f"ep {int(group.attrs['source_episode_index']):02d} | "
            f"{frames} frames | tail removed {removed}"
        )
        axis.set_xlabel("time (s)")
        axis.set_ylabel("joint displacement from episode start (deg)")
        axis.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.966),
        ncol=6,
        frameon=False,
    )
    figure.suptitle(
        "Assembly New1: 10 deterministic random train episodes after tail trim\n"
        "No head trim; 5 frames retained after last detected motion; phase sampling disabled",
        fontsize=15,
        y=0.995,
    )
    figure.subplots_adjust(top=0.93, bottom=0.04, hspace=0.42, wspace=0.28)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
