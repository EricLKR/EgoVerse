#!/usr/bin/env python3
"""Fail-fast config/data/round-trip checks for PIPER pi0.5 V4 Bounded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.models.pi05_expert_action_v4_bounded import (
    bounded_joint_from_latent,
    bounded_joint_to_latent,
)
from egomimic.scripts.piper_process.pi05_expert_action_v3_preflight import EXPECTED


def _check_sample(sample: dict, scale_rad: torch.Tensor) -> dict:
    for key, shape in EXPECTED["piper_left_arm"].items():
        value = sample.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"missing/non-tensor {key}")
        if tuple(value.shape) != shape:
            raise ValueError(f"{key}: expected {shape}, got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{key} contains NaN/Inf")
    gripper = sample["actions_joints"][..., 6]
    distance = torch.minimum(gripper.abs(), (gripper - 1.0).abs())
    if distance.max().item() > 2e-4:
        raise ValueError("gripper target is not binary")
    current = sample["observations.state.joints"][:6][None]
    target = sample["actions_joints"][..., :6][None]
    latent = bounded_joint_to_latent(target, current, scale_rad)
    decoded = bounded_joint_from_latent(latent, current, scale_rad)
    torch.testing.assert_close(decoded, target, atol=2e-6, rtol=2e-6)
    utilization = ((target - current[:, None]) / scale_rad[None, None]).abs()
    return {
        "shapes": {key: list(EXPECTED["piper_left_arm"][key]) for key in EXPECTED["piper_left_arm"]},
        "gripper_values": sorted(torch.unique(gripper).tolist()),
        "max_abs_utilization": utilization.max().item(),
        "round_trip_max_abs_rad": (decoded - target).abs().max().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-audit", type=Path, required=True)
    parser.add_argument(
        "--config-name",
        default="train_zarr_pi05_piper_expert_action_v4_bounded_lora",
    )
    args = parser.parse_args()
    config_dir = Path(__file__).resolve().parents[2] / "hydra_configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name)
    model = cfg.model.robomimic_model
    if list(model.domains) != ["piper_left_arm"]:
        raise ValueError("V4 formal policy must be PIPER-only")
    if int(model.observation_history) != 1:
        raise ValueError("V4 requires one observation frame")
    if int(model.config.model.action_dim) != 32:
        raise ValueError("V4 requires action_dim=32")
    if int(model.config.model.action_horizon) != 30:
        raise ValueError("V4 requires native action_horizon=30")
    scales_deg = torch.tensor(
        list(model.auxiliary.joint_residual_scale_deg), dtype=torch.float32
    )
    if scales_deg.shape != (6,) or not torch.isfinite(scales_deg).all():
        raise ValueError("V4 requires six finite residual scales")
    scale_audit = json.loads(args.scale_audit.read_text(encoding="utf-8"))
    audited = torch.tensor(
        scale_audit["recommended_scale_deg_j1_to_j6"], dtype=torch.float32
    )
    torch.testing.assert_close(scales_deg, audited, atol=0, rtol=0)
    if scale_audit["train"]["utilization"]["violations_ge_1"] != 0:
        raise ValueError("scale audit has training violations")
    if scale_audit["valid_verification_only"]["utilization"]["violations_ge_1"] != 0:
        raise ValueError("scale audit has validation violations")

    report = {
        "config": args.config_name,
        "model_target": str(model._target_),
        "joint_residual_scale_deg": scales_deg.tolist(),
        "scale_audit": str(args.scale_audit),
        "datasets": {},
        "model": OmegaConf.to_container(model, resolve=True),
    }
    scale_rad = torch.deg2rad(scales_deg)
    names_by_split = {}
    for split in ("train_datasets", "valid_datasets"):
        dataset_cfg = cfg.data[split].piper_left_arm
        if bool(dataset_cfg.reject_outliers):
            raise ValueError(f"{split} must retain audited tails")
        if not bool(dataset_cfg.error_on_nan_inf):
            raise ValueError(f"{split} must reject NaN/Inf")
        dataset = hydra.utils.instantiate(dataset_cfg)
        names = sorted(dataset.datasets)
        names_by_split[split] = set(names)
        if not names or len(dataset) <= 0:
            raise ValueError(f"{split} is empty")
        index = min(len(dataset) - 1, max(1, len(dataset) // 2))
        report["datasets"][split] = {
            "frames": len(dataset),
            "episodes": names,
            "checked_index": index,
            **_check_sample(dataset[index], scale_rad),
        }
    overlap = names_by_split["train_datasets"] & names_by_split["valid_datasets"]
    if overlap:
        raise ValueError(f"train/valid leakage: {sorted(overlap)}")
    if any(
        "teddy_0912_7" in name
        for name in names_by_split["train_datasets"] | names_by_split["valid_datasets"]
    ):
        raise ValueError("independent test leaked into train/valid")
    report["split_checks"] = {
        "train_valid_disjoint": True,
        "independent_test_excluded": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"EXPERT_ACTION_V4_PREFLIGHT_OK output={args.output}")


if __name__ == "__main__":
    main()
