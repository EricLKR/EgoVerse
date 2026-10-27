#!/usr/bin/env python3
"""Fail-fast real-data contract check for EgoVerse pi0.5 MultiHead V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.rldb.embodiment.piper import Piper
from egomimic.rldb.embodiment.piper_human_joint import PiperHumanJoint
from egomimic.rldb.embodiment.human_single_arm import get_keymap as get_human_keymap


def _finite(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite tensor")


def _check_sample(name: str, sample: dict) -> dict:
    if name == "piper_left_arm":
        required = {
            "observations.images.front_img_1": (3,),
            "observations.images.wrist_img": (3,),
            "observations.state.joints": (7,),
            "actions_cartesian_pose": (100, 9),
            "actions_joints": (100, 7),
        }
    elif name == "human_left_arm":
        required = {
            "observations.images.front_img_1": (3,),
            "observations.state.ee_pose": (9,),
            "actions_cartesian_pose": (100, 9),
        }
    else:
        raise ValueError(f"unexpected embodiment {name!r}")
    missing = sorted(set(required) - set(sample))
    if missing:
        raise KeyError(f"{name} sample is missing {missing}")
    report = {}
    for key, leading_shape in required.items():
        value = sample[key]
        _finite(f"{name}.{key}", value)
        if tuple(value.shape[: len(leading_shape)]) != leading_shape:
            raise ValueError(
                f"{name}.{key}: expected leading shape {leading_shape}, got {tuple(value.shape)}"
            )
        report[key] = list(value.shape)
    if name == "piper_left_arm":
        gripper = sample["actions_joints"][..., 6]
        binary_distance = torch.minimum(gripper.abs(), (gripper - 1.0).abs())
        if binary_distance.max().item() > 2e-4:
            raise ValueError("PIPER gripper labels are not binary")
        report["gripper_values"] = sorted(torch.unique(gripper).tolist())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_dir = Path(__file__).resolve().parents[2] / "hydra_configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train_zarr_pi05_egoverse_multihead_v2")

    model = cfg.model.robomimic_model
    if list(model.domains) != ["piper_left_arm", "human_left_arm"]:
        raise ValueError(f"unexpected domains: {list(model.domains)}")
    if any(str(value) != "actions_cartesian_pose" for value in model.ac_keys.values()):
        raise ValueError("both shared pi0.5 heads must use actions_cartesian_pose")
    if int(model.observation_history) != 1:
        raise ValueError("MultiHead V2 formal contract requires one current observation")
    if int(model.config.model.action_horizon) != 100:
        raise ValueError("pi0.5 action horizon must be 100")
    if int(model.joint_head.control_horizon) != 30:
        raise ValueError("PIPER joint head control horizon must be 30")
    if not Piper.EXTRINSICS:
        raise ValueError("calibrated PIPER front extrinsics are missing")

    # Single-frame observations must not silently acquire an action horizon.
    piper_map = PiperHumanJoint.get_keymap(
        keymap_mode="joint_cartesian_camframe"
    )
    human_map = get_human_keymap(
        keymap_mode="cartesian_camframe", has_head_pose=True, arm="left"
    )
    for mapping in (piper_map, human_map):
        for info in mapping.values():
            if info.get("key_type") in {"camera_keys", "proprio_keys"}:
                if "history" in info or "horizon" in info:
                    raise AssertionError(
                        "single-frame observation unexpectedly declares history/horizon"
                    )

    report = {
        "config": "train_zarr_pi05_egoverse_multihead_v2",
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
            sample_report = _check_sample(name, dataset[index])
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
