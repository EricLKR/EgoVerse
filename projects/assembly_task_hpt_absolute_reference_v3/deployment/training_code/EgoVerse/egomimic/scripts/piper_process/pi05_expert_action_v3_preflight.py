#!/usr/bin/env python3
"""Fail-fast real-data/config contract check for pi0.5 Expert-Action V3."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.models.pi05_expert_action_v3 import JOINT_DIMS
from egomimic.rldb.embodiment.pi05_expert_action_v3 import PiperExpertActionV3
from egomimic.rldb.embodiment.human_single_arm import get_keymap as get_human_keymap
from egomimic.rldb.embodiment.piper import Piper


EXPECTED = {
    "piper_left_arm": {
        "observations.images.front_img_1": (3, 480, 640),
        "observations.images.wrist_img": (3, 480, 640),
        "observations.state.joints": (7,),
        "actions_cartesian_pose": (30, 9),
        "actions_joints": (30, 7),
    },
    "human_left_arm": {
        "observations.images.front_img_1": (3, 480, 640),
        "observations.state.ee_pose": (9,),
        "actions_cartesian_pose": (30, 9),
    },
}


def _check_sample(name: str, sample: dict, scale_rad: float) -> dict:
    report = {}
    for key, expected_shape in EXPECTED[name].items():
        if key not in sample:
            raise KeyError(f"{name} sample is missing {key!r}")
        value = sample[key]
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"{name}.{key} must be a finite tensor")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name}.{key}: expected {expected_shape}, got {tuple(value.shape)}"
            )
        report[key] = list(value.shape)
    if name == "piper_left_arm":
        gripper = sample["actions_joints"][..., 6]
        binary_distance = torch.minimum(gripper.abs(), (gripper - 1.0).abs())
        if binary_distance.max().item() > 2e-4:
            raise ValueError("PIPER gripper labels are not binary")
        current = sample["observations.state.joints"][:JOINT_DIMS]
        residual = (
            sample["actions_joints"][..., :JOINT_DIMS] - current[None]
        ) / scale_rad
        maximum = residual.abs().max().item()
        if maximum > 1.0 + 1e-5:
            raise ValueError(
                f"PIPER residual exceeds configured V3 scale: {maximum:.6f}"
            )
        report["max_abs_normalized_joint_residual"] = maximum
        report["gripper_values"] = sorted(torch.unique(gripper).tolist())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--config-name",
        default="train_zarr_pi05_egoverse_expert_action_v3",
    )
    parser.add_argument("--expected-domain", action="append")
    args = parser.parse_args()
    config_dir = Path(__file__).resolve().parents[2] / "hydra_configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name)

    model = cfg.model.robomimic_model
    expected_domains = args.expected_domain or [
        "piper_left_arm",
        "human_left_arm",
    ]
    if list(model.domains) != expected_domains:
        raise ValueError(f"unexpected domains: {list(model.domains)}")
    for split in ("train_datasets", "valid_datasets"):
        if list(cfg.data[split]) != expected_domains:
            raise ValueError(
                f"{split} domains {list(cfg.data[split])} do not exactly match "
                f"model domains {expected_domains}"
            )
    if int(model.observation_history) != 1:
        raise ValueError("V3 must use one current observation")
    if int(model.config.model.action_dim) != 32:
        raise ValueError("V3 action_dim must be 32")
    if int(model.config.model.action_horizon) != 30:
        raise ValueError("V3 action_horizon must be 30")
    if int(model.auxiliary.action_horizon) != 30:
        raise ValueError("V3 auxiliary horizon must be 30")
    if str(model.finetune_mode) != "action_expert":
        raise ValueError("V3 pilot must tune the Gemma action expert")
    if not Piper.EXTRINSICS:
        raise ValueError("calibrated PIPER front extrinsics are missing")
    for split in ("train_datasets", "valid_datasets"):
        for name, dataset_cfg in cfg.data[split].items():
            if bool(dataset_cfg.reject_outliers):
                raise ValueError(
                    f"{split}.{name} must keep audited quantile-tail samples; "
                    "reject_outliers must be false"
                )
            if not bool(dataset_cfg.error_on_nan_inf):
                raise ValueError(f"{split}.{name} must reject NaN/Inf")

    piper_map = PiperExpertActionV3.get_keymap(
        keymap_mode="joint_cartesian_camframe"
    )
    mappings = [piper_map]
    if "human_left_arm" in expected_domains:
        mappings.append(
            get_human_keymap(
                keymap_mode="cartesian_camframe", has_head_pose=True, arm="left"
            )
        )
    for mapping in mappings:
        for info in mapping.values():
            if info.get("key_type") in {"camera_keys", "proprio_keys"}:
                if "history" in info or "horizon" in info:
                    raise AssertionError("single-frame observation acquired future/history")

    scale_rad = math.radians(float(model.auxiliary.joint_residual_scale_deg))
    report = {
        "config": args.config_name,
        "model": OmegaConf.to_container(model, resolve=True),
        "datasets": {},
    }
    for split in ("train_datasets", "valid_datasets"):
        report["datasets"][split] = {}
        for name, dataset_cfg in cfg.data[split].items():
            dataset = hydra.utils.instantiate(dataset_cfg)
            if len(dataset) <= 0:
                raise ValueError(f"{split}.{name} is empty")
            index = min(len(dataset) - 1, max(1, len(dataset) // 2))
            sample_report = _check_sample(name, dataset[index], scale_rad)
            sample_report["frames"] = len(dataset)
            sample_report["checked_index"] = index
            report["datasets"][split][name] = sample_report

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
