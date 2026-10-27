#!/usr/bin/env python3
"""Offline trajectory plots for a Piper checkpoint.

This script evaluates recorded observations only; it never connects to or moves
the robot.  For every selected episode it produces two views:

* ``first_action``: the first action of a freshly predicted chunk at each query.
* ``receding_horizon``: the command stream reconstructed with the same chunk
  indexing and query frequency used by ``egomimic.robot.piper_rollout``.

The held-out test split is deliberately protected by ``--allow-test``.  Use the
validation split for iteration and checkpoint selection, then run the test split
once after those choices are frozen.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    # Allow ``python /absolute/path/to/this_script.py`` without relying on an
    # editable install or the caller's working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from egomimic.robot.piper.gripper_utils import binarize_gripper_commands
from egomimic.utils.pose_utils import _matrix_to_xyz6d, _xyz6d_to_matrix, _xyzwxyz_to_matrix


EMBODIMENT = "piper_left_arm"
ACTION_KEY = "piper_left_arm_actions_cartesian"
DEFAULT_ACTION_HORIZON = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required acknowledgement before reading the held-out test split.",
    )
    parser.add_argument(
        "--expected-task-name",
        action="append",
        default=[],
        help="Fail if an episode has a different task_name (repeatable).",
    )
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        help="Episode name/hash or integer position in the sorted split (repeatable).",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--query-stride", type=int, default=10)
    parser.add_argument("--query-frequency", type=int, default=20)
    parser.add_argument(
        "--action-horizon-frames", type=int, default=DEFAULT_ACTION_HORIZON
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.split == "test" and not args.allow_test:
        raise SystemExit(
            "Refusing to read the held-out test split without --allow-test. "
            "Use validation for development/checkpoint selection."
        )
    for name in ("query_stride", "query_frequency", "action_horizon_frames", "batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.query_frequency > args.action_horizon_frames:
        raise ValueError("--query-frequency cannot exceed --action-horizon-frames")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(name)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "episode"


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _rotation_error_deg(pred_xyz6d: np.ndarray, gt_xyz6d: np.ndarray) -> np.ndarray:
    """Geodesic SO(3) error in degrees for matching ``(..., 9)`` poses."""
    pred = np.asarray(pred_xyz6d, dtype=np.float64).reshape(-1, 9)
    gt = np.asarray(gt_xyz6d, dtype=np.float64).reshape(-1, 9)
    pred_rot = _xyz6d_to_matrix(pred)[:, :3, :3]
    gt_rot = _xyz6d_to_matrix(gt)[:, :3, :3]
    relative = np.swapaxes(pred_rot, -1, -2) @ gt_rot
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] != 10:
        raise ValueError(f"Expected matching (T, 10) arrays, got {pred.shape} and {gt.shape}")
    finite = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    if not finite.any():
        raise ValueError("No finite prediction/ground-truth pairs")
    p = pred[finite]
    g = gt[finite]
    xyz_delta = p[:, :3] - g[:, :3]
    pos = np.linalg.norm(xyz_delta, axis=1)
    rot = _rotation_error_deg(p[:, :9], g[:, :9])
    raw_gripper = np.clip(p[:, 9], 0.0, 1.0)
    cmd_gripper = binarize_gripper_commands(
        raw_gripper, close_threshold=0.3, open_threshold=0.7
    ).reshape(-1)
    gt_gripper = g[:, 9].round().clip(0.0, 1.0)
    return {
        "frames": int(len(p)),
        "position_l2_mean_m": float(pos.mean()),
        "position_l2_median_m": float(np.median(pos)),
        "position_l2_max_m": float(pos.max()),
        "xyz_rmse_m": float(np.sqrt(np.mean(xyz_delta**2))),
        "rotation_error_mean_deg": float(rot.mean()),
        "rotation_error_median_deg": float(np.median(rot)),
        "rotation_error_max_deg": float(rot.max()),
        "gripper_command_accuracy": float(np.mean(cmd_gripper == gt_gripper)),
        "gripper_raw_brier": float(np.mean((raw_gripper - gt_gripper) ** 2)),
    }


def _reconstruct_receding_horizon(
    chunks_by_frame: dict[int, np.ndarray],
    episode_frames: int,
    query_frequency: int,
    action_horizon_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Match ``PiperRollout``'s round(offset*T/horizon) chunk indexing."""
    result = np.full((episode_frames, 10), np.nan, dtype=np.float32)
    source_query = np.full(episode_frames, -1, dtype=np.int64)
    for query_frame in range(0, episode_frames, query_frequency):
        chunk = chunks_by_frame[query_frame]
        n_steps = int(chunk.shape[0])
        stop = min(query_frame + query_frequency, episode_frames)
        for frame in range(query_frame, stop):
            offset = frame - query_frame
            step = min(n_steps - 1, int(round(offset * n_steps / action_horizon_frames)))
            result[frame] = chunk[step]
            source_query[frame] = query_frame
    if np.any(source_query < 0):
        raise RuntimeError("Receding-horizon reconstruction left uncovered frames")
    return result, source_query


def _plot_episode(
    *,
    output_path: Path,
    title: str,
    gt: np.ndarray,
    pred: np.ndarray,
    pred_frames: np.ndarray,
    query_frames: np.ndarray,
    dpi: int,
) -> None:
    frames = np.arange(len(gt))
    pred = np.asarray(pred)
    pred_frames = np.asarray(pred_frames)
    query_frames = np.asarray(query_frames)
    pos_error = np.linalg.norm(pred[:, :3] - gt[pred_frames, :3], axis=1)
    rot_error = _rotation_error_deg(pred[:, :9], gt[pred_frames, :9])
    raw_gripper = np.clip(pred[:, 9], 0.0, 1.0)
    cmd_gripper = binarize_gripper_commands(
        raw_gripper, close_threshold=0.3, open_threshold=0.7
    ).reshape(-1)

    fig, axes = plt.subplots(6, 1, figsize=(14, 13), sharex=True, constrained_layout=True)
    colors = ("#d95f02", "#1b9e77", "#7570b3")
    labels = ("X (m)", "Y (m)", "Z (m)")
    for dim, (axis, color, label) in enumerate(zip(axes[:3], colors, labels)):
        axis.plot(frames, gt[:, dim], color="black", linewidth=1.25, label="GT")
        axis.plot(pred_frames, pred[:, dim], color=color, linewidth=1.1, label="model")
        query_mask = np.isin(pred_frames, query_frames)
        axis.scatter(
            pred_frames[query_mask], pred[query_mask, dim], s=13, color="#c43c39",
            zorder=4, label="query" if dim == 0 else None,
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8)

    grip_axis = axes[3]
    grip_axis.step(frames, gt[:, 9], where="post", color="black", linewidth=1.25, label="GT command")
    grip_axis.plot(pred_frames, raw_gripper, color="#e6ab02", linewidth=1.0, label="raw model")
    grip_axis.step(pred_frames, cmd_gripper, where="post", color="#1b9e77", linewidth=1.1, label="0.3/0.7 hysteresis")
    grip_axis.set_yticks([0.0, 1.0], labels=["close", "open"])
    grip_axis.set_ylim(-0.12, 1.12)
    grip_axis.set_ylabel("gripper")
    grip_axis.grid(alpha=0.25)
    grip_axis.legend(loc="best", fontsize=8)

    axes[4].plot(pred_frames, pos_error, color="#4c78a8", linewidth=1.0)
    axes[4].scatter(query_frames, pos_error[np.isin(pred_frames, query_frames)], s=12, color="#c43c39", zorder=4)
    axes[4].set_ylabel("position\nerror (m)")
    axes[4].grid(alpha=0.25)

    axes[5].plot(pred_frames, rot_error, color="#8e5ea2", linewidth=1.0)
    axes[5].scatter(query_frames, rot_error[np.isin(pred_frames, query_frames)], s=12, color="#c43c39", zorder=4)
    axes[5].set_ylabel("rotation\nerror (deg)")
    axes[5].set_xlabel("episode frame index (30 Hz)")
    axes[5].grid(alpha=0.25)

    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _load_policy(checkpoint: Path, device: torch.device, inference_steps: int):
    from egomimic.models.denoising_policy import DenoisingPolicy
    from egomimic.pl_utils.pl_model import ModelWrapper

    policy = ModelWrapper.load_from_checkpoint(
        str(checkpoint), weights_only=False, map_location="cpu"
    )
    policy = policy.to(device)
    policy.eval()
    policy.model.device = device
    if getattr(policy.model, "diffusion", False):
        for head in policy.model.nets.policy.heads.values():
            if isinstance(head, DenoisingPolicy):
                head.num_inference_steps = inference_steps
    return policy


def _load_dataset(config_path: Path, policy):
    import hydra
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    dataset_cfg = cfg.valid_datasets[EMBODIMENT]
    dataset = hydra.utils.instantiate(dataset_cfg)
    dataset.set_norm_stats_from(policy.model.norm_stats)
    return dataset


def _episode_records(dataset) -> list[dict]:
    records = []
    for position, episode_name in enumerate(sorted(dataset._global_indices_by_dataset)):
        leaf = dataset.datasets[episode_name]
        metadata = dict(leaf.metadata)
        records.append(
            {
                "position": position,
                "name": episode_name,
                "task_name": str(metadata.get("task_name", "unknown")),
                "frames": int(len(leaf)),
                "global_indices": list(dataset._global_indices_by_dataset[episode_name]),
                "leaf": leaf,
            }
        )
    return records


def _select_records(records: list[dict], selectors: list[str], max_episodes: int | None) -> list[dict]:
    if not selectors:
        selected = records
    else:
        selected = []
        by_name = {r["name"]: r for r in records}
        for selector in selectors:
            if selector in by_name:
                record = by_name[selector]
            else:
                try:
                    record = records[int(selector)]
                except (ValueError, IndexError) as exc:
                    raise ValueError(
                        f"Unknown --episode {selector!r}; use a name/hash or a 0-based position"
                    ) from exc
            if record not in selected:
                selected.append(record)
    return selected[:max_episodes] if max_episodes is not None else selected


def _ground_truth(record: dict) -> np.ndarray:
    store = record["leaf"].episode_reader._store
    # Zarr arrays can have chunk-aligned physical capacity beyond the logical
    # episode length.  Those trailing rows are zero-filled (including a zero
    # quaternion), so always honor the metadata-backed dataset length.
    frame_count = record["frames"]
    pose_xyzwxyz = np.asarray(store["obs_ee_pose"][:frame_count], dtype=np.float64)
    gripper = np.asarray(store["cmd_gripper"][:frame_count], dtype=np.float64).reshape(-1, 1)
    pose_xyz6d = _matrix_to_xyz6d(_xyzwxyz_to_matrix(pose_xyzwxyz))
    gt = np.concatenate([pose_xyz6d, gripper], axis=-1).astype(np.float32)
    if gt.shape != (record["frames"], 10):
        raise ValueError(
            f"Unexpected GT shape for {record['name']}: {gt.shape}; expected ({record['frames']}, 10)"
        )
    return gt


def _predict_chunks(
    policy,
    dataset,
    global_indices: list[int],
    local_frames: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    from egomimic.pl_utils.pl_data_utils import annotation_collate

    predictions: dict[int, np.ndarray] = {}
    for frame_batch in _chunks(local_frames, batch_size):
        samples = [dataset[global_indices[frame]] for frame in frame_batch]
        batch = annotation_collate(samples)
        processed = policy.model.process_batch_for_training({EMBODIMENT: batch})
        with torch.inference_mode():
            output = policy.model.forward_eval(processed)[ACTION_KEY]
        chunks_np = output.detach().cpu().numpy().astype(np.float32)
        for frame, chunk in zip(frame_batch, chunks_np):
            predictions[frame] = chunk
    return predictions


def _aggregate_metrics(per_episode: list[dict], view: str) -> dict:
    if not per_episode:
        return {}
    metric_keys = [
        key for key in per_episode[0][view]
        if key != "frames"
    ]
    weights = np.asarray([item[view]["frames"] for item in per_episode], dtype=np.float64)
    return {
        "episodes": len(per_episode),
        "frames": int(weights.sum()),
        **{
            key: float(np.average([item[view][key] for item in per_episode], weights=weights))
            for key in metric_keys
        },
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = _load_policy(args.checkpoint, device, args.num_inference_steps)
    dataset = _load_dataset(args.data_config, policy)
    records = _episode_records(dataset)

    expected = set(args.expected_task_name)
    observed = {record["task_name"] for record in records}
    if expected and not observed.issubset(expected):
        raise RuntimeError(
            f"Data-config split mismatch: expected task_name in {sorted(expected)}, "
            f"observed {sorted(observed)}"
        )
    selected = _select_records(records, args.episode, args.max_episodes)
    if not selected:
        raise RuntimeError("No episodes selected")

    print(
        f"device={device} split={args.split} episodes={len(selected)}/{len(records)} "
        f"query_stride={args.query_stride} query_frequency={args.query_frequency}",
        flush=True,
    )
    episode_results = []
    for ep_number, record in enumerate(selected, start=1):
        n_frames = record["frames"]
        first_queries = list(range(0, n_frames, args.query_stride))
        rollout_queries = list(range(0, n_frames, args.query_frequency))
        inference_frames = sorted(set(first_queries) | set(rollout_queries))
        print(
            f"[{ep_number}/{len(selected)}] {record['task_name']}/{record['name']}: "
            f"{n_frames} frames, {len(inference_frames)} model queries",
            flush=True,
        )
        chunks_by_frame = _predict_chunks(
            policy, dataset, record["global_indices"], inference_frames, args.batch_size
        )
        gt = _ground_truth(record)

        first_pred = np.stack([chunks_by_frame[frame][0] for frame in first_queries])
        first_frames = np.asarray(first_queries, dtype=np.int64)
        receding_pred, source_query = _reconstruct_receding_horizon(
            chunks_by_frame,
            n_frames,
            args.query_frequency,
            args.action_horizon_frames,
        )

        stem = _safe_name(f"{record['task_name']}_{record['name']}")
        common_title = (
            f"{args.split}: {record['task_name']} / {record['name']} | "
            f"checkpoint={args.checkpoint.name}"
        )
        first_png = args.output_dir / f"{stem}_first_action.png"
        receding_png = args.output_dir / f"{stem}_receding_horizon.png"
        _plot_episode(
            output_path=first_png,
            title=f"First action per query (stride={args.query_stride})\n{common_title}",
            gt=gt,
            pred=first_pred,
            pred_frames=first_frames,
            query_frames=first_frames,
            dpi=args.dpi,
        )
        all_frames = np.arange(n_frames, dtype=np.int64)
        _plot_episode(
            output_path=receding_png,
            title=(
                f"Receding horizon (query frequency={args.query_frequency}, "
                f"action horizon={args.action_horizon_frames})\n{common_title}"
            ),
            gt=gt,
            pred=receding_pred,
            pred_frames=all_frames,
            query_frames=np.asarray(rollout_queries, dtype=np.int64),
            dpi=args.dpi,
        )
        npz_path = args.output_dir / f"{stem}_predictions.npz"
        np.savez_compressed(
            npz_path,
            gt=gt,
            first_action_prediction=first_pred,
            first_action_frames=first_frames,
            receding_horizon_prediction=receding_pred,
            receding_horizon_source_query=source_query,
            rollout_query_frames=np.asarray(rollout_queries, dtype=np.int64),
        )
        result = {
            "episode": record["name"],
            "task_name": record["task_name"],
            "total_frames": n_frames,
            "first_action": _metrics(first_pred, gt[first_frames]),
            "receding_horizon": _metrics(receding_pred, gt),
            "files": {
                "first_action_plot": str(first_png),
                "receding_horizon_plot": str(receding_png),
                "predictions": str(npz_path),
            },
        }
        episode_results.append(result)
        with open(args.output_dir / f"{stem}_metrics.json", "w") as handle:
            json.dump(_to_jsonable(result), handle, indent=2)

    summary = {
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "data_config": str(args.data_config.resolve()),
        "device": str(device),
        "seed": args.seed,
        "query_stride": args.query_stride,
        "query_frequency": args.query_frequency,
        "action_horizon_frames": args.action_horizon_frames,
        "num_inference_steps": args.num_inference_steps,
        "observed_task_names": sorted(observed),
        "aggregate": {
            "first_action": _aggregate_metrics(episode_results, "first_action"),
            "receding_horizon": _aggregate_metrics(episode_results, "receding_horizon"),
        },
        "episodes": episode_results,
    }
    summary_path = args.output_dir / "summary_metrics.json"
    with open(summary_path, "w") as handle:
        json.dump(_to_jsonable(summary), handle, indent=2)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
