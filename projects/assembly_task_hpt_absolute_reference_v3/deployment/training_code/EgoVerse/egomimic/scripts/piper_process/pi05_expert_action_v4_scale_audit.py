#!/usr/bin/env python3
"""Derive V4 per-joint bounded-residual scales from the training split only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import hydra
import numpy as np
from hydra import compose, initialize_config_dir


JOINT_DIMS = 6
QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)


def _scan_dataset(dataset) -> tuple[np.ndarray, dict]:
    chunks: list[np.ndarray] = []
    episodes: list[str] = []
    frames = 0
    for episode_name, leaf in sorted(dataset.datasets.items()):
        episodes.append(episode_name)
        episode_frames = len(leaf)
        frames += episode_frames
        store = leaf.episode_reader._store
        joints = np.asarray(store["cmd_joint"][:], dtype=np.float64)[..., :JOINT_DIMS]
        current = np.asarray(store["obs_joint"][:], dtype=np.float64)[..., :JOINT_DIMS]
        if joints.ndim != 2 or current.ndim != 2:
            raise ValueError(f"{episode_name}: joint arrays must be rank two")
        if joints.shape[1] != JOINT_DIMS or current.shape[1] != JOINT_DIMS:
            raise ValueError(f"{episode_name}: expected six joint columns")
        if min(joints.shape[0], current.shape[0]) < episode_frames:
            raise ValueError(f"{episode_name}: joint arrays are shorter than usable frames")
        if not np.isfinite(joints).all() or not np.isfinite(current).all():
            raise ValueError(f"{episode_name}: non-finite joint value")
        offsets = np.arange(30, dtype=np.int64)[None, :]
        for start in range(0, episode_frames, 1024):
            stop = min(start + 1024, episode_frames)
            starts = np.arange(start, stop, dtype=np.int64)[:, None]
            indices = np.minimum(starts + offsets, joints.shape[0] - 1)
            delta_deg = np.degrees(
                np.abs(joints[indices] - current[start:stop, None, :])
            )
            chunks.append(delta_deg.reshape(-1, JOINT_DIMS))
    if frames != len(dataset):
        raise AssertionError(f"scanned {frames} frames but dataset reports {len(dataset)}")
    return np.concatenate(chunks, axis=0), {
        "episodes": episodes,
        "episode_count": len(episodes),
        "frames": frames,
    }


def _stats(values: np.ndarray) -> dict:
    return {
        "joint_abs_residual_deg_quantiles": {
            str(q): np.quantile(values, q, axis=0).tolist() for q in QUANTILES
        },
        "max_abs_residual_deg": values.max(axis=0).tolist(),
    }


def _recommended_scales(
    train_values: np.ndarray,
    *,
    target_quantile: float,
    target_utilization: float,
    maximum_utilization: float,
    round_up_deg: float,
) -> np.ndarray:
    quantile = np.quantile(train_values, target_quantile, axis=0)
    maximum = train_values.max(axis=0)
    raw = np.maximum(
        quantile / target_utilization,
        maximum / maximum_utilization,
    )
    return np.ceil(raw / round_up_deg) * round_up_deg


def _utilization(values: np.ndarray, scale: np.ndarray) -> dict:
    ratio = values / scale[None, :]
    return {
        "max_per_joint": ratio.max(axis=0).tolist(),
        "max_global": float(ratio.max()),
        "violations_ge_1": int((ratio >= 1.0).sum()),
        "samples_ge_0_95": int((ratio >= 0.95).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-config",
        default="train_zarr_pi05_piper_expert_action_v3_lora",
    )
    parser.add_argument("--target-quantile", type=float, default=0.999)
    parser.add_argument("--target-utilization", type=float, default=0.80)
    parser.add_argument("--maximum-utilization", type=float, default=0.95)
    parser.add_argument("--round-up-deg", type=float, default=5.0)
    parser.add_argument("--expected-scales-deg", type=float, nargs=6)
    args = parser.parse_args()
    if not 0.0 < args.target_quantile <= 1.0:
        raise ValueError("target quantile must be in (0,1]")
    if not 0.0 < args.target_utilization < args.maximum_utilization < 1.0:
        raise ValueError("require 0 < target utilization < maximum utilization < 1")
    if not math.isfinite(args.round_up_deg) or args.round_up_deg <= 0:
        raise ValueError("round-up step must be positive")

    config_dir = Path(__file__).resolve().parents[2] / "hydra_configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name=args.source_config)
    if list(cfg.data.train_datasets) != ["piper_left_arm"]:
        raise ValueError("V4 scale derivation must use the PIPER-only training split")

    train_dataset = hydra.utils.instantiate(
        cfg.data.train_datasets.piper_left_arm
    )
    valid_dataset = hydra.utils.instantiate(
        cfg.data.valid_datasets.piper_left_arm
    )
    train_values, train_meta = _scan_dataset(train_dataset)
    valid_values, valid_meta = _scan_dataset(valid_dataset)
    train_names = set(train_meta["episodes"])
    valid_names = set(valid_meta["episodes"])
    overlap = train_names & valid_names
    if overlap:
        raise ValueError(f"train/valid episode leakage: {sorted(overlap)}")
    if any("teddy_0912_7" in name for name in train_names | valid_names):
        raise ValueError("independent test session leaked into train/valid")

    scales = _recommended_scales(
        train_values,
        target_quantile=args.target_quantile,
        target_utilization=args.target_utilization,
        maximum_utilization=args.maximum_utilization,
        round_up_deg=args.round_up_deg,
    )
    if args.expected_scales_deg is not None:
        expected = np.asarray(args.expected_scales_deg, dtype=np.float64)
        if not np.array_equal(scales, expected):
            raise ValueError(
                f"configured scales {expected.tolist()} do not match "
                f"training-only recommendation {scales.tolist()}"
            )

    report = {
        "source_config": args.source_config,
        "derivation": {
            "source": "training split only",
            "target_quantile": args.target_quantile,
            "target_utilization": args.target_utilization,
            "maximum_utilization": args.maximum_utilization,
            "round_up_deg": args.round_up_deg,
        },
        "recommended_scale_deg_j1_to_j6": scales.tolist(),
        "train": {
            **train_meta,
            **_stats(train_values),
            "utilization": _utilization(train_values, scales),
        },
        "valid_verification_only": {
            **valid_meta,
            **_stats(valid_values),
            "utilization": _utilization(valid_values, scales),
        },
        "split_checks": {
            "train_valid_disjoint": True,
            "independent_test_excluded": True,
            "validation_not_used_to_derive_scales": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["train"]["utilization"]["violations_ge_1"]:
        raise ValueError("derived scales do not cover the complete training split")
    if report["valid_verification_only"]["utilization"]["violations_ge_1"]:
        raise ValueError(
            "training-derived scales do not cover validation; revise the "
            "training-only safety-margin rule before training"
        )
    print(f"EXPERT_ACTION_V4_SCALE_AUDIT_OK output={args.output}")


if __name__ == "__main__":
    main()
