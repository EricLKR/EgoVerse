#!/usr/bin/env python3
"""Fail-fast config/data/sampler audit for pi0.5 Mixed CamFrame V5."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.models.pi05_expert_action_v3 import JOINT_SLICE
from egomimic.models.pi05_expert_action_v4_bounded import (
    bounded_joint_from_latent,
    bounded_joint_to_latent,
    build_expert_action_target_v4,
)
from egomimic.pl_utils.pl_data_utils_v5 import EpisodeUniformWindowSampler
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


def _episode_lengths(dataset) -> dict[str, int]:
    return {name: len(child) for name, child in sorted(dataset.datasets.items())}


def _check_sample(name: str, sample: dict, scale_rad: torch.Tensor) -> dict:
    result = {}
    for key, shape in EXPECTED[name].items():
        value = sample.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} missing tensor {key}")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name}.{key}: expected {shape}, got {tuple(value.shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}.{key} contains NaN/Inf")
        result[key] = list(value.shape)

    if name == "piper_left_arm":
        current = sample["observations.state.joints"][:6][None]
        target = sample["actions_joints"][..., :6][None]
        latent = bounded_joint_to_latent(target, current, scale_rad)
        decoded = bounded_joint_from_latent(latent, current, scale_rad)
        torch.testing.assert_close(decoded, target, atol=2e-6, rtol=2e-6)
        gripper = sample["actions_joints"][..., 6]
        binary_error = torch.minimum(gripper.abs(), (gripper - 1).abs()).max()
        if binary_error.item() > 2e-4:
            raise ValueError("PIPER gripper is not binary")
        result["max_joint_scale_utilization"] = float(
            ((target - current[:, None]) / scale_rad[None, None]).abs().max()
        )
        result["gripper_values"] = sorted(torch.unique(gripper).tolist())
    else:
        pose = torch.zeros(1, 30, 32)
        pose[..., :9] = sample["actions_cartesian_pose"][None]
        _, weights = build_expert_action_target_v4(
            pose,
            torch.ones_like(pose),
            pose_weight=1.0,
            joint_residual_scale_rad=scale_rad,
        )
        if torch.count_nonzero(weights[..., JOINT_SLICE]).item() != 0:
            raise AssertionError("Human joint channels are not exactly masked")
        if torch.count_nonzero(weights[..., 15:]).item() != 0:
            raise AssertionError("Human reserved channels are not exactly masked")
        result["joint_and_reserved_mask_zero"] = True
    return result


def _episode_probe_indices(dataset, horizon: int = 30) -> list[int]:
    """Return first/middle/last complete-window indices for every episode."""
    output: list[int] = []
    for _, global_indices in sorted(dataset._global_indices_by_dataset.items()):
        valid = global_indices[: len(global_indices) - horizon + 1]
        if not valid:
            raise ValueError("episode has no complete action window")
        output.extend((valid[0], valid[len(valid) // 2], valid[-1]))
    return sorted(set(output))


def _numeric_dataset_config(dataset_cfg):
    """Clone a dataset config with cameras removed before full-frame scanning."""
    numeric_cfg = copy.deepcopy(dataset_cfg)
    OmegaConf.update(
        numeric_cfg, "resolver.key_map.norm_mode", True, force_add=True
    )
    return numeric_cfg


def _rotation6d_validity(rotation: torch.Tensor) -> tuple[float, float]:
    c1 = rotation[..., :3]
    c2 = rotation[..., 3:6]
    norm_error = torch.maximum((c1.norm(dim=-1) - 1).abs(), (c2.norm(dim=-1) - 1).abs())
    dot_error = (c1 * c2).sum(dim=-1).abs()
    return float(norm_error.max()), float(dot_error.max())


def _full_numeric_scan(name: str, dataset, scale_rad: torch.Tensor) -> dict:
    """Scan every transformed frame without decoding images."""
    max_scale_utilization = 0.0
    max_rotation_norm_error = 0.0
    max_rotation_dot_error = 0.0
    gripper_values: set[float] = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        if name == "piper_left_arm":
            state = sample.get("observations.state.joints")
            action = sample.get("actions_joints")
            pose = sample.get("actions_cartesian_pose")
            if tuple(getattr(state, "shape", ())) != (7,):
                raise ValueError(f"PIPER frame {index} state shape is {getattr(state, 'shape', None)}")
            if tuple(getattr(action, "shape", ())) != (30, 7):
                raise ValueError(f"PIPER frame {index} action shape is {getattr(action, 'shape', None)}")
            if tuple(getattr(pose, "shape", ())) != (30, 9):
                raise ValueError(f"PIPER frame {index} pose shape is {getattr(pose, 'shape', None)}")
            tensors = (state, action, pose)
            if not all(torch.isfinite(value).all() for value in tensors):
                raise ValueError(f"PIPER frame {index} contains NaN/Inf")
            values = torch.unique(action[..., 6]).tolist()
            gripper_values.update(float(value) for value in values)
            binary_error = torch.minimum(action[..., 6].abs(), (action[..., 6] - 1).abs()).max()
            if float(binary_error) > 2e-4:
                raise ValueError(f"PIPER frame {index} gripper is not binary")
            utilization = (
                (action[..., :6] - state[:6][None]) / scale_rad[None]
            ).abs().max()
            max_scale_utilization = max(max_scale_utilization, float(utilization))
            rotation = pose[..., 3:9]
        else:
            state = sample.get("observations.state.ee_pose")
            pose = sample.get("actions_cartesian_pose")
            if tuple(getattr(state, "shape", ())) != (9,):
                raise ValueError(f"Human frame {index} state shape is {getattr(state, 'shape', None)}")
            if tuple(getattr(pose, "shape", ())) != (30, 9):
                raise ValueError(f"Human frame {index} pose shape is {getattr(pose, 'shape', None)}")
            if not torch.isfinite(state).all() or not torch.isfinite(pose).all():
                raise ValueError(f"Human frame {index} contains NaN/Inf")
            rotation = torch.cat((state[3:9][None], pose[..., 3:9]), dim=0)
        norm_error, dot_error = _rotation6d_validity(rotation)
        max_rotation_norm_error = max(max_rotation_norm_error, norm_error)
        max_rotation_dot_error = max(max_rotation_dot_error, dot_error)

    # The conversion should produce orthonormal 6D rotation columns.  Keep a
    # small allowance for float32 roundoff, but reject coordinate corruption.
    if max_rotation_norm_error > 2e-3 or max_rotation_dot_error > 2e-3:
        raise ValueError(
            f"{name} invalid 6D rotation: norm={max_rotation_norm_error}, "
            f"dot={max_rotation_dot_error}"
        )
    result = {
        "frames_scanned": len(dataset),
        "max_rotation_unit_norm_error": max_rotation_norm_error,
        "max_rotation_abs_column_dot": max_rotation_dot_error,
    }
    if name == "piper_left_arm":
        result["max_joint_scale_utilization"] = max_scale_utilization
        result["gripper_values"] = sorted(gripper_values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config-name", default="train_zarr_pi05_mixed_camframe_v5_lora"
    )
    args = parser.parse_args()

    config_dir = Path(__file__).resolve().parents[2] / "hydra_configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name)

    model = cfg.model.robomimic_model
    expected_domains = ["piper_left_arm", "human_left_arm"]
    if list(model.domains) != expected_domains:
        raise ValueError(f"unexpected model domains: {list(model.domains)}")
    if str(model._target_) != (
        "egomimic.algo.pi_egoverse_mixed_camframe_v5."
        "PIEgoVerseMixedCamFrameV5LoRA"
    ):
        raise ValueError(f"unexpected V5 target {model._target_}")
    if int(model.observation_history) != 1:
        raise ValueError("V5 requires one observation frame")
    if int(model.config.model.action_horizon) != 30:
        raise ValueError("V5 requires native 30-step horizon")
    if int(model.config.model.action_dim) != 32:
        raise ValueError("V5 requires 32 action-expert channels")
    if not 0.0 < float(model.human_pose_final_weight) <= 0.5:
        raise ValueError("Human final loss weight must be in (0,0.5]")
    if int(model.human_pose_ramp_steps) <= 0:
        raise ValueError("Human loss ramp must be enabled")
    if int(model.human_update_interval) != 2:
        raise ValueError("V5 audited domain update ratio requires interval=2")
    if str(cfg.data._target_) != (
        "egomimic.pl_utils.pl_data_utils_v5."
        "EpisodeBalancedMultiDataModuleWrapper"
    ):
        raise ValueError("V5 must use its episode-balanced data module")
    if int(cfg.data.valid_future_horizon) != 30:
        raise ValueError("validation must exclude incomplete 30-step windows")
    if not Piper.EXTRINSICS:
        raise ValueError("calibrated PIPER front extrinsics are missing")

    scale_deg = torch.tensor(
        list(model.auxiliary.joint_residual_scale_deg), dtype=torch.float32
    )
    if scale_deg.shape != (6,) or not torch.isfinite(scale_deg).all():
        raise ValueError("V5 requires six finite joint residual scales")
    scale_rad = torch.deg2rad(scale_deg)

    report = {
        "config": args.config_name,
        "model_target": str(model._target_),
        "action_contract": {"horizon": 30, "dimension": 32},
        "domain_update_ratio": {"piper": 2, "human": 1},
        "joint_residual_scale_deg": scale_deg.tolist(),
        "model": OmegaConf.to_container(model, resolve=True),
        "datasets": {},
    }
    names_by_split: dict[str, dict[str, set[str]]] = {}
    for split in ("train_datasets", "valid_datasets"):
        if list(cfg.data[split]) != expected_domains:
            raise ValueError(f"{split} does not contain exactly both domains")
        report["datasets"][split] = {}
        names_by_split[split] = {}
        for name, dataset_cfg in cfg.data[split].items():
            if bool(dataset_cfg.reject_outliers):
                raise ValueError(f"{split}.{name} must retain audited tails")
            if not bool(dataset_cfg.error_on_nan_inf):
                raise ValueError(f"{split}.{name} must fail on NaN/Inf")
            dataset = hydra.utils.instantiate(dataset_cfg)
            lengths = _episode_lengths(dataset)
            if not lengths:
                raise ValueError(f"{split}.{name} is empty")
            if min(lengths.values()) < 30:
                raise ValueError(f"{split}.{name} contains an episode shorter than 30")
            names_by_split[split][name] = set(lengths)
            probe_indices = _episode_probe_indices(dataset)
            probes = [
                _check_sample(name, dataset[index], scale_rad)
                for index in probe_indices
            ]
            numeric_dataset = hydra.utils.instantiate(
                _numeric_dataset_config(dataset_cfg)
            )
            numeric_scan = _full_numeric_scan(name, numeric_dataset, scale_rad)
            entry = {
                "frames": len(dataset),
                "episodes": len(lengths),
                "min_episode_frames": min(lengths.values()),
                "max_episode_frames": max(lengths.values()),
                "probe_indices": probe_indices,
                "probes": probes,
                "full_numeric_scan": numeric_scan,
            }
            if split == "train_datasets":
                sampler_cfg = OmegaConf.to_container(
                    cfg.data.train_sampler_params[name], resolve=True
                )
                sampler = EpisodeUniformWindowSampler(dataset, **sampler_cfg)
                sampled = list(iter(sampler))
                entry["episode_sampler"] = {
                    "config": sampler_cfg,
                    "samples": len(sampled),
                    "unique_samples": len(set(sampled)),
                }
                if len(sampled) != len(sampler):
                    raise AssertionError("sampler length contract failed")
            report["datasets"][split][name] = entry

    for domain in expected_domains:
        overlap = names_by_split["train_datasets"][domain] & names_by_split[
            "valid_datasets"
        ][domain]
        if overlap:
            raise ValueError(f"{domain} train/valid leakage: {sorted(overlap)}")
    all_piper = names_by_split["train_datasets"]["piper_left_arm"] | names_by_split[
        "valid_datasets"
    ]["piper_left_arm"]
    if any("teddy_0912_7" in name for name in all_piper):
        raise ValueError("independent PIPER test session leaked into train/valid")

    report["gates"] = {
        "train_valid_disjoint": True,
        "independent_test_excluded": True,
        "human_joint_gripper_masked": True,
        "episode_uniform_sampling": True,
        "all_numeric_frames_scanned": True,
        "every_episode_camera_probed": True,
        "native_30_step_contract": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"PI05_MIXED_CAMFRAME_V5_PREFLIGHT_OK output={args.output}")


if __name__ == "__main__":
    main()
