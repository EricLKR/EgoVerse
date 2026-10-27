#!/usr/bin/env python3
"""Fail-fast data/config checks for PIPER and PIPER+human pi0.5 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.utils.action_utils import (
    HumanLeftXYZ6DActionConverter,
    PiperJointActionConverter,
)


def _shape(value) -> list[int]:
    return list(value.shape) if hasattr(value, "shape") else []


def _check_sample(name: str, sample: dict, *, mixed: bool) -> dict:
    required = ["observations.images.front_img_1"]
    if name == "piper_left_arm":
        required += [
            "observations.images.wrist_img",
            "observations.state.joints",
            "actions_joints",
        ]
        expected_action_dim = 7
        converter = PiperJointActionConverter(gripper_weight=3.0)
        action_key = "actions_joints"
    elif name == "human_left_arm" and mixed:
        required += ["observations.state.ee_pose", "actions_cartesian_pose"]
        expected_action_dim = 9
        converter = HumanLeftXYZ6DActionConverter()
        action_key = "actions_cartesian_pose"
    else:
        raise AssertionError(f"Unexpected dataset {name!r}")

    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"{name} sample is missing {missing}; keys={sorted(sample)}")

    action = sample[action_key]
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"{name}.{action_key} is not a tensor: {type(action)}")
    if action.ndim != 2 or tuple(action.shape) != (100, expected_action_dim):
        raise ValueError(
            f"{name}.{action_key} expected [100,{expected_action_dim}], got {_shape(action)}"
        )
    if not torch.isfinite(action).all():
        raise ValueError(f"{name}.{action_key} contains NaN/Inf")

    packed = converter.to32(action.unsqueeze(0))
    restored = converter.from32(packed)
    weights = converter.loss_weights32(action.unsqueeze(0))
    if packed.shape != (1, 100, 32) or weights.shape != packed.shape:
        raise ValueError(
            f"{name} converter returned packed={tuple(packed.shape)}, "
            f"weights={tuple(weights.shape)}"
        )
    if not torch.equal(restored, action.unsqueeze(0)):
        raise ValueError(f"{name} action converter round-trip is not exact")

    report = {
        "sample_keys": sorted(sample),
        "action_key": action_key,
        "action_shape": _shape(action),
        "front_shape": _shape(sample["observations.images.front_img_1"]),
        "active_pi05_channels": int((weights[0, 0] > 0).sum().item()),
        "action_min": action.amin(dim=0).tolist(),
        "action_max": action.amax(dim=0).tolist(),
    }
    if "observations.images.wrist_img" in sample:
        report["wrist_shape"] = _shape(sample["observations.images.wrist_img"])
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("piper", "mixed"), required=True)
    parser.add_argument("--require-weights", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_name = (
        "train_zarr_pi05_piper"
        if args.variant == "piper"
        else "train_zarr_pi05_piper_human"
    )
    config_dir = Path(__file__).resolve().parents[2] / "hydra_configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)

    report = {
        "variant": args.variant,
        "config": config_name,
        "piper_hpc": os.environ.get("PIPER_HPC"),
        "datasets": {},
        "dataloaders": {
            "train": OmegaConf.to_container(
                cfg.data.train_dataloader_params, resolve=True
            ),
            "valid": OmegaConf.to_container(
                cfg.data.valid_dataloader_params, resolve=True
            ),
        },
    }
    for split, loader_cfgs in report["dataloaders"].items():
        for name, loader_cfg in loader_cfgs.items():
            if int(loader_cfg["batch_size"]) != 2:
                raise ValueError(
                    f"{split} dataloader {name!r} must use the smoke-validated "
                    f"partial-finetune batch_size=2; got {loader_cfg['batch_size']}"
                )
    for split in ("train_datasets", "valid_datasets"):
        report["datasets"][split] = {}
        for name, dataset_cfg in cfg.data[split].items():
            dataset = hydra.utils.instantiate(dataset_cfg)
            if len(dataset) <= 0:
                raise ValueError(f"{split}.{name} is empty")
            sample_report = _check_sample(
                name, dataset[0], mixed=args.variant == "mixed"
            )
            sample_report["frames"] = len(dataset)
            report["datasets"][split][name] = sample_report

    model_cfg = cfg.model.robomimic_model
    report["model"] = {
        "domains": list(model_cfg.domains),
        "action_dim": int(model_cfg.config.model.action_dim),
        "action_horizon": int(model_cfg.config.model.action_horizon),
        "camera_slot_maps": OmegaConf.to_container(
            model_cfg.camera_slot_maps, resolve=True
        ),
        "gradient_checkpointing": bool(model_cfg.gradient_checkpointing),
        "finetune_mode": str(model_cfg.finetune_mode),
        "train_multimodal_projector": bool(model_cfg.train_multimodal_projector),
    }
    if report["model"]["action_dim"] != 32:
        raise ValueError("pi0.5 action_dim must be 32")
    if report["model"]["action_horizon"] != 100:
        raise ValueError("pi0.5 action_horizon must be 100")
    if report["model"]["finetune_mode"] != "action_expert":
        raise ValueError(
            "Formal PIPER pi0.5 configs must use action_expert partial finetuning"
        )

    weight_dir = model_cfg.config.pytorch_weight_path
    if isinstance(weight_dir, str) and weight_dir.lower() == "null":
        weight_dir = None
    if args.require_weights and not weight_dir:
        raise ValueError("PI05_PYTORCH_WEIGHTS is required for this preflight")
    if weight_dir:
        weight_file = Path(weight_dir) / "model.safetensors"
        if not weight_file.is_file():
            raise FileNotFoundError(weight_file)
        report["weights"] = {
            "path": str(weight_file),
            "bytes": weight_file.stat().st_size,
            "sha256": _sha256(weight_file),
        }

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
