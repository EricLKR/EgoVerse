"""Preflight the Piper joint-space data and Hydra training pipeline."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import egomimic.trainHydra  # noqa: F401 -- registers project resolvers
from egomimic.robot.piper.piper_kinematics import PiperKinematicsSolver


EPISODE_RE = re.compile(r"^piper-(?P<session>teddy_0912_[1-7])-ep(?P<episode>\d{4})\.zarr$")
REQUIRED = {
    "obs_joint": 6,
    "cmd_joint": 6,
    "cmd_ee_pose": 7,
    "obs_gripper": 1,
    "cmd_gripper": 1,
    "images.front_1": None,
    "images.wrist_1": None,
}
EXPECTED_SPLIT_FRAMES = {"train": 18178, "valid": 3315, "test": 3294}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def split_for_session(session: str) -> str:
    index = int(session.rsplit("_", 1)[-1])
    return "train" if index <= 5 else "valid" if index == 6 else "test"


def scan_stores(data_dir: Path, errors: list[str]) -> dict[str, Any]:
    paths = sorted(data_dir.glob("*.zarr"))
    if len(paths) != 70:
        errors.append(f"expected 70 episode stores, found {len(paths)}")
    fk = PiperKinematicsSolver()
    split_frames: Counter[str] = Counter()
    split_episodes: Counter[str] = Counter()
    session_episodes: dict[str, list[int]] = defaultdict(list)
    ranges = {
        "obs_joint_min": np.full(6, np.inf),
        "obs_joint_max": np.full(6, -np.inf),
        "cmd_joint_min": np.full(6, np.inf),
        "cmd_joint_max": np.full(6, -np.inf),
    }
    gripper_counts: Counter[int] = Counter()
    max_fk_position_error = 0.0
    max_fk_quaternion_error = 0.0

    for path in paths:
        match = EPISODE_RE.match(path.name)
        if not match:
            errors.append(f"unexpected episode name {path.name}")
            continue
        session = match.group("session")
        episode = int(match.group("episode"))
        session_episodes[session].append(episode)
        split = split_for_session(session)
        group = zarr.open_group(str(path), mode="r")
        n = int(group.attrs.get("total_frames", -1))
        if n <= 0:
            errors.append(f"{path.name}: invalid total_frames={n}")
            continue
        split_frames[split] += n
        split_episodes[split] += 1
        for key, width in REQUIRED.items():
            if key not in group:
                errors.append(f"{path.name}: missing {key}")
                continue
            shape = tuple(group[key].shape)
            if shape[0] < n:
                errors.append(f"{path.name}: {key} has {shape[0]} rows, needs {n}")
            if width is not None and (len(shape) != 2 or shape[1] != width):
                errors.append(f"{path.name}: {key} shape={shape}, expected (*,{width})")
        if any(key not in group for key in REQUIRED):
            continue
        obs_joint = np.asarray(group["obs_joint"][:n], dtype=np.float64)
        cmd_joint = np.asarray(group["cmd_joint"][:n], dtype=np.float64)
        cmd_pose = np.asarray(group["cmd_ee_pose"][:n], dtype=np.float64)
        cmd_gripper = np.asarray(group["cmd_gripper"][:n], dtype=np.float64)
        for key, value in {
            "obs_joint": obs_joint,
            "cmd_joint": cmd_joint,
            "cmd_ee_pose": cmd_pose,
            "cmd_gripper": cmd_gripper,
        }.items():
            if not np.isfinite(value).all():
                errors.append(f"{path.name}: non-finite {key}")
        if not np.isin(cmd_gripper, (0.0, 1.0)).all():
            errors.append(f"{path.name}: cmd_gripper is not binary")
        gripper_counts.update(cmd_gripper.astype(np.int64).reshape(-1).tolist())
        ranges["obs_joint_min"] = np.minimum(ranges["obs_joint_min"], obs_joint.min(0))
        ranges["obs_joint_max"] = np.maximum(ranges["obs_joint_max"], obs_joint.max(0))
        ranges["cmd_joint_min"] = np.minimum(ranges["cmd_joint_min"], cmd_joint.min(0))
        ranges["cmd_joint_max"] = np.maximum(ranges["cmd_joint_max"], cmd_joint.max(0))
        recomputed = fk.fk(cmd_joint)
        max_fk_position_error = max(
            max_fk_position_error, float(np.max(np.abs(recomputed[:, :3] - cmd_pose[:, :3])))
        )
        qdot = np.abs(np.sum(recomputed[:, 3:] * cmd_pose[:, 3:], axis=1))
        max_fk_quaternion_error = max(
            max_fk_quaternion_error, float(np.max(1.0 - np.clip(qdot, 0.0, 1.0)))
        )

    expected_sessions = {f"teddy_0912_{index}" for index in range(1, 8)}
    if set(session_episodes) != expected_sessions:
        errors.append(f"sessions={sorted(session_episodes)}, expected={sorted(expected_sessions)}")
    for session, episodes in sorted(session_episodes.items()):
        if sorted(episodes) != list(range(10)):
            errors.append(f"{session}: episodes={sorted(episodes)}, expected 0..9")
    for split, expected in EXPECTED_SPLIT_FRAMES.items():
        if split_frames[split] != expected:
            errors.append(f"{split} frames={split_frames[split]}, expected={expected}")
    if max_fk_position_error > 1e-10 or max_fk_quaternion_error > 1e-10:
        errors.append(
            "cmd_ee_pose does not match FK(cmd_joint): "
            f"position={max_fk_position_error}, quaternion={max_fk_quaternion_error}"
        )
    return {
        "episodes": dict(split_episodes),
        "frames": dict(split_frames),
        "sessions": {key: sorted(value) for key, value in session_episodes.items()},
        "ranges_rad": ranges,
        "cmd_gripper_counts": {"closed_0": gripper_counts[0], "open_1": gripper_counts[1]},
        "max_fk_position_error": max_fk_position_error,
        "max_fk_quaternion_error": max_fk_quaternion_error,
    }


def check_pipeline(code_dir: Path, data_dir: Path, errors: list[str]) -> dict[str, Any]:
    config_dir = str(code_dir / "egomimic/hydra_configs")
    datasets = {}
    for split, data_name in (
        ("train_valid", "piper_solo_teddy_0912_joint"),
        ("held_out_test", "piper_solo_teddy_0912_joint_test"),
    ):
        with initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = compose(
                config_name="train_zarr_joint",
                overrides=[
                    f"data={data_name}",
                    f"data.train_datasets.piper_left_arm.resolver.folder_path={data_dir}",
                    f"data.valid_datasets.piper_left_arm.resolver.folder_path={data_dir}",
                ],
            )
        datasets[split] = {
            "train": instantiate(cfg.data.train_datasets.piper_left_arm),
            "valid": instantiate(cfg.data.valid_datasets.piper_left_arm),
        }

    expected_lengths = {
        ("train_valid", "train"): EXPECTED_SPLIT_FRAMES["train"],
        ("train_valid", "valid"): EXPECTED_SPLIT_FRAMES["valid"],
        ("held_out_test", "train"): EXPECTED_SPLIT_FRAMES["train"],
        ("held_out_test", "valid"): EXPECTED_SPLIT_FRAMES["test"],
    }
    report = {}
    for group_name, group in datasets.items():
        report[group_name] = {}
        for split, dataset in group.items():
            length = len(dataset)
            expected = expected_lengths[(group_name, split)]
            if length != expected:
                errors.append(f"{group_name}.{split} samples={length}, expected={expected}")
            indices = sorted({0, max(0, length // 2), max(0, length - 1)})
            samples = []
            for index in indices:
                sample = dataset[index]
                required = (
                    "actions_joints",
                    "observations.state.joints",
                    "observations.images.front_img_1",
                    "observations.images.wrist_img",
                )
                missing = [key for key in required if key not in sample]
                if missing:
                    errors.append(f"{group_name}.{split}[{index}] missing {missing}")
                    continue
                action = sample["actions_joints"]
                proprio = sample["observations.state.joints"]
                if not isinstance(action, torch.Tensor) or tuple(action.shape) != (100, 7):
                    errors.append(f"{group_name}.{split}[{index}] action shape={getattr(action, 'shape', None)}")
                if not isinstance(proprio, torch.Tensor) or tuple(proprio.shape) != (7,):
                    errors.append(f"{group_name}.{split}[{index}] proprio shape={getattr(proprio, 'shape', None)}")
                if not torch.isfinite(action).all() or not torch.isfinite(proprio).all():
                    errors.append(f"{group_name}.{split}[{index}] non-finite transformed tensors")
                if not torch.all((action[:, 6] == 0) | (action[:, 6] == 1)):
                    errors.append(f"{group_name}.{split}[{index}] transformed gripper is non-binary")
                samples.append(
                    {
                        "index": index,
                        "action_shape": list(action.shape),
                        "proprio_shape": list(proprio.shape),
                        "front_shape": list(sample["observations.images.front_img_1"].shape),
                        "wrist_shape": list(sample["observations.images.wrist_img"].shape),
                    }
                )
            report[group_name][split] = {"samples": length, "checked_examples": samples}
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    report = {
        "stored_data": scan_stores(args.data_dir, errors),
        "training_pipeline": check_pipeline(args.code_dir, args.data_dir, errors),
    }
    report["status"] = "PASS" if not errors else "FAIL"
    report["errors"] = errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=jsonable) + "\n")
    print(json.dumps(report, indent=2, default=jsonable))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
