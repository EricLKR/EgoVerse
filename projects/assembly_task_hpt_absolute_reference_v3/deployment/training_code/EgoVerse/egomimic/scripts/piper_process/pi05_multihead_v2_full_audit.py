#!/usr/bin/env python3
"""Full-data, read-only preflight audit for EgoVerse pi0.5 MultiHead V2.

Unlike the fast smoke preflight, this script visits every transformed window
in train and validation.  It reports the exact fraction of windows that the
training dataset would replace because of quantile-bound rejection, verifies
finite values and tensor contracts, checks split disjointness, and measures
the binary PIPER gripper label balance.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


EXPECTED = {
    "piper_left_arm": {
        "observations.images.front_img_1": (3, 480, 640),
        "observations.images.wrist_img": (3, 480, 640),
        "observations.state.joints": (7,),
        "actions_cartesian_pose": (100, 9),
        "actions_joints": (100, 7),
    },
    "human_left_arm": {
        "observations.images.front_img_1": (3, 480, 640),
        "observations.state.ee_pose": (9,),
        "actions_cartesian_pose": (100, 9),
    },
}


def _load_stats(path: Path, train_datasets: dict[str, MultiDataset]) -> MultiDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    descriptor = MultiDataset(state={"norm_mode": "quantile"})
    descriptor.populate_from_datasets(train_datasets)
    descriptor.norm_stats = {
        int(embodiment): {
            key: {
                name: np.asarray(value, dtype=np.float32)
                for name, value in stats.items()
            }
            for key, stats in keys.items()
        }
        for embodiment, keys in payload["stats"].items()
    }
    return descriptor


def _bound_counts(value: torch.Tensor, stats: dict) -> tuple[int, int, int]:
    value = value.float()
    low = torch.as_tensor(stats["quantile_0_01"], dtype=torch.float32)
    high = torch.as_tensor(stats["quantile_99_99"], dtype=torch.float32)
    low = torch.broadcast_to(low, value.shape)
    high = torch.broadcast_to(high, value.shape)
    below = int((value < low).sum().item())
    above = int((value > high).sum().item())
    return below, above, value.numel()


def _scan_dataset(
    name: str,
    dataset: MultiDataset,
    descriptor: MultiDataset,
) -> dict:
    expected = EXPECTED[name]
    per_key_windows = defaultdict(int)
    per_key_below = defaultdict(int)
    per_key_above = defaultdict(int)
    per_key_values = defaultdict(int)
    windows_outside_any = 0
    frames = 0
    gripper_open = 0
    gripper_closed = 0
    gripper_transitions = 0
    episodes: list[str] = []

    for episode_name, leaf in sorted(dataset.datasets.items()):
        episodes.append(episode_name)
        for local_index in range(len(leaf)):
            sample = leaf[local_index]
            frames += 1
            missing = sorted(set(expected) - set(sample))
            if missing:
                raise KeyError(f"{name}:{episode_name}[{local_index}] missing {missing}")
            for key, shape in expected.items():
                value = sample[key]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{name}.{key} must be a tensor")
                if tuple(value.shape) != shape:
                    raise ValueError(
                        f"{name}:{episode_name}[{local_index}] {key} has "
                        f"shape {tuple(value.shape)}, expected {shape}"
                    )
                if not torch.isfinite(value).all():
                    raise ValueError(
                        f"{name}:{episode_name}[{local_index}] {key} contains NaN/Inf"
                    )

            embodiment = int(sample["embodiment"])
            outside_this_window = False
            for stat_key, stat in descriptor.norm_stats[embodiment].items():
                data_key = descriptor.zarr_keys[embodiment].get(stat_key)
                if data_key is None or data_key not in sample:
                    continue
                below, above, values = _bound_counts(sample[data_key], stat)
                per_key_below[data_key] += below
                per_key_above[data_key] += above
                per_key_values[data_key] += values
                if below or above:
                    per_key_windows[data_key] += 1
                    outside_this_window = True
            windows_outside_any += int(outside_this_window)

            if name == "piper_left_arm":
                gripper = sample["actions_joints"][..., 6]
                distance = torch.minimum(gripper.abs(), (gripper - 1.0).abs())
                if distance.max().item() > 2e-4:
                    raise ValueError(
                        f"non-binary gripper label in {episode_name}[{local_index}]"
                    )
                is_open = gripper >= 0.5
                gripper_open += int(is_open.sum().item())
                gripper_closed += int((~is_open).sum().item())
                gripper_transitions += int(
                    (is_open[1:] != is_open[:-1]).sum().item()
                )

    if frames != len(dataset):
        raise AssertionError(f"{name}: scanned {frames}, dataset reports {len(dataset)}")
    keys = {}
    for key in sorted(per_key_values):
        keys[key] = {
            "windows_outside": per_key_windows[key],
            "window_fraction_outside": per_key_windows[key] / frames,
            "values_below": per_key_below[key],
            "values_above": per_key_above[key],
            "scalar_fraction_outside": (
                per_key_below[key] + per_key_above[key]
            ) / per_key_values[key],
        }
    report = {
        "episodes": len(episodes),
        "episode_names": episodes,
        "frames": frames,
        "all_values_finite": True,
        "windows_outside_any_training_bound": windows_outside_any,
        "window_fraction_outside_any_training_bound": windows_outside_any / frames,
        "bounds_by_key": keys,
    }
    if name == "piper_left_arm":
        total = gripper_open + gripper_closed
        report["gripper"] = {
            "binary": True,
            "open_values": gripper_open,
            "closed_values": gripper_closed,
            "open_fraction": gripper_open / total,
            "closed_fraction": gripper_closed / total,
            "transitions_in_overlapping_action_windows": gripper_transitions,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_dir = Path(__file__).resolve().parents[2] / "hydra_configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train_zarr_pi05_egoverse_multihead_v2")

    train_datasets = {
        name: hydra.utils.instantiate(dataset_cfg)
        for name, dataset_cfg in cfg.data.train_datasets.items()
    }
    valid_datasets = {
        name: hydra.utils.instantiate(dataset_cfg)
        for name, dataset_cfg in cfg.data.valid_datasets.items()
    }
    descriptor = _load_stats(args.norm_stats, train_datasets)

    report = {"train": {}, "valid": {}, "split_checks": {}}
    for name, dataset in train_datasets.items():
        report["train"][name] = _scan_dataset(name, dataset, descriptor)
    for name, dataset in valid_datasets.items():
        report["valid"][name] = _scan_dataset(name, dataset, descriptor)

    for name in sorted(train_datasets):
        train_episodes = set(report["train"][name]["episode_names"])
        valid_episodes = set(report["valid"][name]["episode_names"])
        overlap = sorted(train_episodes & valid_episodes)
        if overlap:
            raise ValueError(f"{name} train/valid episode leakage: {overlap}")
        report["split_checks"][name] = {
            "train_valid_disjoint": True,
            "overlap": [],
        }

    piper_used = set(report["train"]["piper_left_arm"]["episode_names"])
    piper_used.update(report["valid"]["piper_left_arm"]["episode_names"])
    if any("teddy_0912_7" in name for name in piper_used):
        raise ValueError("independent PIPER test session teddy_0912_7 leaked into train/valid")
    report["split_checks"]["independent_piper_test_excluded"] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"FULL_AUDIT_OK output={args.output}")


if __name__ == "__main__":
    main()
