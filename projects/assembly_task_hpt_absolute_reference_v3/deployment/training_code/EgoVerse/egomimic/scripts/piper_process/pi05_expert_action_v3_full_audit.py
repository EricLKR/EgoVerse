#!/usr/bin/env python3
"""Read every V3 train/validation window and enforce its native-rate contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from egomimic.scripts.piper_process.pi05_expert_action_v3_preflight import EXPECTED


def _scan(name, dataset, scale_rad):
    frames = 0
    episode_names = []
    max_residual = 0.0
    max_residual_location = None
    max_residual_detail = None
    residual_degrees = []
    residual_scale_violations = 0
    open_values = 0
    closed_values = 0
    transitions = 0
    for episode_name, leaf in sorted(dataset.datasets.items()):
        episode_names.append(episode_name)
        episode_frames = len(leaf)
        frames += episode_frames
        store = leaf.episode_reader._store

        # Check every raw numeric value once, rather than decoding the same JPEG
        # up to 30 times through overlapping action windows.  This is both more
        # complete for numeric signals and fast enough to run before every job.
        for mapped_key, info in leaf.key_map.items():
            if info.get("key_type") == "annotation_keys":
                continue
            zarr_key = info["zarr_key"]
            if zarr_key not in store:
                raise KeyError(f"{name}:{episode_name} missing raw key {zarr_key}")
            array = store[zarr_key]
            array_frames = int(array.shape[0])
            if array_frames < episode_frames:
                raise ValueError(
                    f"{name}:{episode_name} {zarr_key} length "
                    f"{array_frames} is shorter than usable frames {episode_frames}"
                )
            if zarr_key in leaf._image_keys:
                for image_index in sorted({0, episode_frames // 2, episode_frames - 1}):
                    encoded = array[image_index : image_index + 1][0]
                    if not encoded or len(encoded) < 16:
                        raise ValueError(
                            f"{name}:{episode_name} {zarr_key}[{image_index}] is empty"
                        )
                continue
            if zarr_key in leaf._json_keys:
                continue
            values = np.asarray(array[:])
            if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                raise ValueError(f"{name}:{episode_name} {zarr_key} contains NaN/Inf")

        # Exercise the complete mapping/transform stack at episode boundaries
        # and midpoint, including JPEG decoding and native 30-step padding.
        for index in sorted({0, episode_frames // 2, episode_frames - 1}):
            sample = leaf[index]
            for key, shape in EXPECTED[name].items():
                if key not in sample:
                    raise KeyError(f"{name}:{episode_name}[{index}] missing {key}")
                value = sample[key]
                if tuple(value.shape) != shape:
                    raise ValueError(
                        f"{name}:{episode_name}[{index}] {key} shape "
                        f"{tuple(value.shape)} != {shape}"
                    )
                if not torch.isfinite(value).all():
                    raise ValueError(
                        f"{name}:{episode_name}[{index}] {key} contains NaN/Inf"
                    )
        if name == "piper_left_arm":
            joints = np.asarray(store["cmd_joint"][:], dtype=np.float64)[..., :6]
            current = np.asarray(store["obs_joint"][:], dtype=np.float64)[..., :6]
            if (
                joints.ndim != 2
                or current.ndim != 2
                or joints.shape[0] < episode_frames
                or current.shape[0] < episode_frames
                or joints.shape[1] != 6
                or current.shape[1] != 6
            ):
                raise ValueError(f"{episode_name}: unexpected raw joint shapes")
            offsets = np.arange(30, dtype=np.int64)[None, :]
            for start in range(0, episode_frames, 1024):
                stops = min(start + 1024, episode_frames)
                starts = np.arange(start, stops, dtype=np.int64)[:, None]
                indices = np.minimum(starts + offsets, joints.shape[0] - 1)
                residual = np.abs(joints[indices] - current[start:stops, None]) / scale_rad
                per_window = residual.max(axis=(1, 2))
                residual_degrees.append(np.degrees(per_window * scale_rad))
                local_index = int(per_window.argmax())
                sample_max = float(per_window[local_index])
                if sample_max > max_residual:
                    max_residual = sample_max
                    max_residual_location = f"{episode_name}[{start + local_index}]"
                    _, offset_index, joint_index = np.unravel_index(
                        int(residual.argmax()), residual.shape
                    )
                    absolute_start = start + local_index
                    target_index = int(indices[local_index, offset_index])
                    max_residual_detail = {
                        "joint_index": int(joint_index),
                        "action_offset": int(offset_index),
                        "target_frame": target_index,
                        "current_rad": float(current[absolute_start, joint_index]),
                        "target_rad": float(joints[target_index, joint_index]),
                        "absolute_residual_deg": float(
                            np.degrees(
                                abs(
                                    joints[target_index, joint_index]
                                    - current[absolute_start, joint_index]
                                )
                            )
                        ),
                    }
                residual_scale_violations += int((per_window > 1.0 + 1e-5).sum())

            gripper = np.asarray(store["cmd_gripper"][:], dtype=np.float64).reshape(-1)
            binary_distance = np.minimum(np.abs(gripper), np.abs(gripper - 1.0))
            if float(binary_distance.max(initial=0.0)) > 2e-4:
                raise ValueError(f"non-binary gripper in {episode_name}")
            is_open = gripper >= 0.5
            open_values += int(is_open.sum())
            closed_values += int((~is_open).sum())
            transitions += int((is_open[1:] != is_open[:-1]).sum())
    if frames != len(dataset):
        raise AssertionError(f"{name}: scanned {frames}, expected {len(dataset)}")
    report = {
        "episodes": len(episode_names),
        "episode_names": episode_names,
        "frames": frames,
        "all_values_finite": True,
    }
    if name == "piper_left_arm":
        total = open_values + closed_values
        if open_values == 0 or closed_values == 0 or transitions == 0:
            raise ValueError("PIPER audit found insufficient gripper class/transition coverage")
        report.update(
            {
                "max_abs_normalized_joint_residual": max_residual,
                "max_joint_residual_location": max_residual_location,
                "max_joint_residual_detail": max_residual_detail,
                "joint_residual_scale_violations": residual_scale_violations,
                "joint_residual_abs_deg_quantiles": {
                    str(quantile): float(
                        np.quantile(np.concatenate(residual_degrees), quantile)
                    )
                    for quantile in (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
                },
                "gripper_open_fraction": open_values / total,
                "gripper_closed_fraction": closed_values / total,
                "gripper_transitions_in_windows": transitions,
            }
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_dir = Path(__file__).resolve().parents[2] / "hydra_configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train_zarr_pi05_egoverse_expert_action_v3")
    scale_rad = math.radians(
        float(cfg.model.robomimic_model.auxiliary.joint_residual_scale_deg)
    )
    train = {
        name: hydra.utils.instantiate(dataset_cfg)
        for name, dataset_cfg in cfg.data.train_datasets.items()
    }
    valid = {
        name: hydra.utils.instantiate(dataset_cfg)
        for name, dataset_cfg in cfg.data.valid_datasets.items()
    }
    report = {"train": {}, "valid": {}, "split_checks": {}}
    for name, dataset in train.items():
        report["train"][name] = _scan(name, dataset, scale_rad)
    for name, dataset in valid.items():
        report["valid"][name] = _scan(name, dataset, scale_rad)
    for name in sorted(train):
        overlap = set(report["train"][name]["episode_names"]) & set(
            report["valid"][name]["episode_names"]
        )
        if overlap:
            raise ValueError(f"{name} train/valid leakage: {sorted(overlap)}")
        report["split_checks"][name] = {"train_valid_disjoint": True}
    piper_used = set(report["train"]["piper_left_arm"]["episode_names"])
    piper_used.update(report["valid"]["piper_left_arm"]["episode_names"])
    if any("teddy_0912_7" in name for name in piper_used):
        raise ValueError("independent PIPER test session leaked into train/valid")
    report["split_checks"]["independent_piper_test_excluded"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    violations = sum(
        report[split]["piper_left_arm"]["joint_residual_scale_violations"]
        for split in ("train", "valid")
    )
    if violations:
        raise ValueError(
            f"joint residual scale exceeded in {violations} PIPER windows; "
            f"complete diagnostics were written to {args.output}"
        )
    print(f"EXPERT_ACTION_V3_FULL_AUDIT_OK output={args.output}")


if __name__ == "__main__":
    main()
