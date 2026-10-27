#!/usr/bin/env python
"""Build a first-grasp joint-pose safety profile from training episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def first_close_index(commands: np.ndarray) -> int:
    values = np.asarray(commands, dtype=np.float64).reshape(-1)
    transitions = np.flatnonzero((values[:-1] > 0.5) & (values[1:] <= 0.5)) + 1
    if transitions.size == 0:
        raise RuntimeError("episode has no open-to-close gripper transition")
    return int(transitions[0])


def nearest_neighbor_rms(samples: np.ndarray) -> np.ndarray:
    delta = samples[:, None, :] - samples[None, :, :]
    rms = np.sqrt(np.mean(np.square(delta), axis=-1))
    np.fill_diagonal(rms, np.inf)
    return np.min(rms, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glob", default="*train*.zarr")
    parser.add_argument("--max-nearest-rms-deg", type=float, default=3.0)
    parser.add_argument("--envelope-margin-deg", type=float, default=1.0)
    args = parser.parse_args()

    episodes = sorted(args.dataset_root.glob(args.glob))
    if not episodes:
        raise RuntimeError(f"no training episodes matched {args.dataset_root / args.glob}")

    records: list[dict[str, object]] = []
    samples: list[np.ndarray] = []
    for episode in episodes:
        group = zarr.open_group(episode, mode="r")
        index = first_close_index(group["cmd_gripper"][:])
        joints_deg = np.rad2deg(np.asarray(group["obs_joint"][index], dtype=np.float64))
        if joints_deg.shape != (6,) or not np.isfinite(joints_deg).all():
            raise RuntimeError(f"invalid obs_joint at {episode.name} frame {index}")
        samples.append(joints_deg)
        records.append(
            {
                "episode": episode.name,
                "first_close_frame": index,
                "obs_joint_deg": joints_deg.tolist(),
            }
        )

    pose_samples = np.stack(samples, axis=0)
    nn_rms = nearest_neighbor_rms(pose_samples)
    percentiles = {
        name: np.percentile(pose_samples, percentile, axis=0).tolist()
        for name, percentile in (
            ("p01", 1),
            ("p05", 5),
            ("median", 50),
            ("p95", 95),
            ("p99", 99),
        )
    }
    profile = {
        "schema": "assembly_first_grasp_pose_profile_v1",
        "source_dataset_root": str(args.dataset_root.resolve()),
        "joint_source": "obs_joint at first cmd_gripper open-to-close transition",
        "joint_units": "degrees",
        "episode_count": len(records),
        "joint_names": [f"J{index}" for index in range(1, 7)],
        "samples_deg": pose_samples.tolist(),
        "percentiles_deg": percentiles,
        "nearest_neighbor_rms_deg": {
            "median": float(np.percentile(nn_rms, 50)),
            "p90": float(np.percentile(nn_rms, 90)),
            "p95": float(np.percentile(nn_rms, 95)),
            "p99": float(np.percentile(nn_rms, 99)),
            "max": float(np.max(nn_rms)),
        },
        "recommended_gate": {
            "max_nearest_rms_deg": float(args.max_nearest_rms_deg),
            "envelope_percentiles": [1, 99],
            "envelope_margin_deg": float(args.envelope_margin_deg),
        },
        "episodes": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episode_count": len(records),
                "p01_deg": percentiles["p01"],
                "p99_deg": percentiles["p99"],
                "nearest_neighbor_rms_deg": profile["nearest_neighbor_rms_deg"],
                "recommended_gate": profile["recommended_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
