#!/usr/bin/env python3
"""Offline validation/test evaluation for a joint-space PIPER checkpoint.

Produces one PNG and JSON per episode plus ``summary_metrics.json``.  The plot
contains all six commanded joints, the independently binary gripper, and
Cartesian position/rotation errors obtained by FK of the predicted joints
against the stored ``cmd_ee_pose``.  This script never connects to the robot.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from egomimic.robot.piper.piper_kinematics import PiperKinematicsSolver
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix


EMBODIMENT = "piper_left_arm"
ACTION_KEY = "piper_left_arm_actions_joints"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--query-frequency", type=int, default=10)
    parser.add_argument("--action-horizon-frames", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def _load_policy(path: Path, device: torch.device, inference_steps: int):
    from egomimic.models.denoising_policy import DenoisingPolicy
    from egomimic.pl_utils.pl_model import ModelWrapper

    policy = ModelWrapper.load_from_checkpoint(
        str(path), weights_only=False, map_location="cpu"
    ).to(device)
    policy.eval()
    policy.model.device = device
    for head in policy.model.nets.policy.heads.values():
        if isinstance(head, DenoisingPolicy):
            head.num_inference_steps = inference_steps
    return policy


def _load_dataset(path: Path, policy):
    import hydra
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    dataset = hydra.utils.instantiate(cfg.valid_datasets[EMBODIMENT])
    dataset.set_norm_stats_from(policy.model.norm_stats)
    return dataset


def _records(dataset):
    result = []
    for name in sorted(dataset._global_indices_by_dataset):
        leaf = dataset.datasets[name]
        result.append(
            {
                "name": name,
                "leaf": leaf,
                "frames": len(leaf),
                "indices": list(dataset._global_indices_by_dataset[name]),
            }
        )
    return result


def _ground_truth(record):
    store = record["leaf"].episode_reader._store
    n = record["frames"]
    required = ("cmd_joint", "cmd_gripper", "cmd_ee_pose")
    missing = [key for key in required if key not in store]
    if missing:
        raise RuntimeError(
            f"{record['name']} lacks {missing}; reconvert with --calibration-json"
        )
    joints = np.asarray(store["cmd_joint"][:n], dtype=np.float64)
    gripper = np.asarray(store["cmd_gripper"][:n], dtype=np.float64).reshape(n, 1)
    pose = np.asarray(store["cmd_ee_pose"][:n], dtype=np.float64)
    return np.concatenate((joints, gripper), axis=-1), pose


def _predict(policy, dataset, record, query_frames, batch_size):
    from egomimic.pl_utils.pl_data_utils import annotation_collate

    output = {}
    for start in range(0, len(query_frames), batch_size):
        frames = query_frames[start : start + batch_size]
        samples = [dataset[record["indices"][frame]] for frame in frames]
        batch = annotation_collate(samples)
        processed = policy.model.process_batch_for_training({EMBODIMENT: batch})
        with torch.inference_mode():
            chunks = policy.model.forward_eval(processed)[ACTION_KEY]
        for frame, chunk in zip(frames, chunks.detach().cpu().numpy()):
            output[frame] = chunk
    return output


def _reconstruct(chunks, frame_count, query_frequency, horizon_frames):
    result = np.full((frame_count, 7), np.nan, dtype=np.float64)
    for query in range(0, frame_count, query_frequency):
        chunk = chunks[query]
        stop = min(query + query_frequency, frame_count)
        for frame in range(query, stop):
            offset = frame - query
            index = min(len(chunk) - 1, int(round(offset * len(chunk) / horizon_frames)))
            result[frame] = chunk[index]
    return result


def _pose_errors(pred_pose, gt_pose):
    position = np.linalg.norm(pred_pose[:, :3] - gt_pose[:, :3], axis=-1)
    pred_rot = _xyzwxyz_to_matrix(pred_pose)[:, :3, :3]
    gt_rot = _xyzwxyz_to_matrix(gt_pose)[:, :3, :3]
    relative = np.swapaxes(pred_rot, -1, -2) @ gt_rot
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return position, np.rad2deg(np.arccos(cosine))


def _metrics(pred, gt, gt_pose, solver):
    joint_delta = pred[:, :6] - gt[:, :6]
    pred_binary = pred[:, 6] >= 0.5
    gt_binary = gt[:, 6] >= 0.5
    tp = np.sum(pred_binary & gt_binary)
    fp = np.sum(pred_binary & ~gt_binary)
    fn = np.sum(~pred_binary & gt_binary)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    pred_pose = solver.fk(pred[:, :6])
    position, rotation = _pose_errors(pred_pose, gt_pose)
    return {
        "frames": int(len(gt)),
        "joint_mae_deg": float(np.mean(np.abs(np.rad2deg(joint_delta)))),
        "joint_rmse_deg": float(np.sqrt(np.mean(np.rad2deg(joint_delta) ** 2))),
        "joint_mae_deg_per_joint": np.mean(
            np.abs(np.rad2deg(joint_delta)), axis=0
        ).tolist(),
        "gripper_accuracy": float(np.mean(pred_binary == gt_binary)),
        "gripper_open_precision": float(precision),
        "gripper_open_recall": float(recall),
        "gripper_open_f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "cmd_ee_pose_position_mae_m": float(position.mean()),
        "cmd_ee_pose_position_rmse_m": float(np.sqrt(np.mean(position**2))),
        "cmd_ee_pose_rotation_mae_deg": float(rotation.mean()),
        "cmd_ee_pose_rotation_rmse_deg": float(np.sqrt(np.mean(rotation**2))),
    }, pred_pose, position, rotation


def _plot(path, name, pred, gt, position, rotation, dpi):
    frames = np.arange(len(gt))
    fig, axes = plt.subplots(5, 2, figsize=(13, 13), sharex=True)
    axes = axes.reshape(-1)
    for index in range(6):
        axes[index].plot(frames, np.rad2deg(gt[:, index]), color="black", label="GT")
        axes[index].plot(frames, np.rad2deg(pred[:, index]), color="#d62728", label="model")
        axes[index].set_ylabel(f"J{index + 1} (deg)")
        axes[index].grid(alpha=0.25)
    axes[6].step(frames, gt[:, 6] >= 0.5, where="post", color="black", label="GT")
    axes[6].step(frames, pred[:, 6] >= 0.5, where="post", color="#1b9e77", label="model")
    axes[6].set_yticks([0, 1], labels=["close", "open"])
    axes[6].set_ylabel("gripper")
    axes[6].grid(alpha=0.25)
    axes[7].plot(frames, position * 100.0, color="#4c78a8")
    axes[7].set_ylabel("FK position err (cm)")
    axes[7].grid(alpha=0.25)
    axes[8].plot(frames, rotation, color="#8e5ea2")
    axes[8].set_ylabel("FK rotation err (deg)")
    axes[8].set_xlabel("episode frame (30 Hz)")
    axes[8].grid(alpha=0.25)
    axes[9].axis("off")
    axes[0].legend(fontsize=8)
    axes[6].legend(fontsize=8)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("Refusing test evaluation without --allow-test")
    if args.query_frequency <= 0 or args.query_frequency > args.action_horizon_frames:
        raise ValueError("query frequency must be in [1, action_horizon_frames]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = _load_policy(args.checkpoint, device, args.num_inference_steps)
    dataset = _load_dataset(args.data_config, policy)
    records = _records(dataset)
    if args.max_episodes is not None:
        records = records[: args.max_episodes]
    if not records:
        raise RuntimeError("No episodes were selected by the validation data config")
    solver = PiperKinematicsSolver(device="cpu")
    all_metrics = []
    for number, record in enumerate(records, 1):
        query_frames = list(range(0, record["frames"], args.query_frequency))
        chunks = _predict(policy, dataset, record, query_frames, args.batch_size)
        pred = _reconstruct(
            chunks, record["frames"], args.query_frequency, args.action_horizon_frames
        )
        gt, gt_pose = _ground_truth(record)
        metrics, _, position, rotation = _metrics(pred, gt, gt_pose, solver)
        metrics["episode"] = record["name"]
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", record["name"])
        (args.output_dir / f"{stem}_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
        _plot(
            args.output_dir / f"{stem}_joint_eval.png",
            f"{args.split}: {record['name']} | qf={args.query_frequency}",
            pred,
            gt,
            position,
            rotation,
            args.dpi,
        )
        all_metrics.append(metrics)
        print(f"[{number}/{len(records)}] {record['name']}: {metrics}", flush=True)

    numeric_keys = [
        key for key, value in all_metrics[0].items()
        if isinstance(value, (int, float)) and key != "frames"
    ]
    weights = np.asarray([item["frames"] for item in all_metrics], dtype=np.float64)
    summary = {
        "split": args.split,
        "episodes": len(all_metrics),
        "frames": int(weights.sum()),
        **{
            key: float(np.average([item[key] for item in all_metrics], weights=weights))
            for key in numeric_keys
        },
    }
    (args.output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
