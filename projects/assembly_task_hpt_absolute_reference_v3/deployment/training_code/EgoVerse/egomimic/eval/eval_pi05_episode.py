"""Episode-level PIPER evaluation for pi0.5 checkpoints.

Unlike :mod:`eval_pi05_offline`, which renders one independent 100-action
chunk per validation sample, this evaluator reconstructs the trajectory that
would be executed by replanning every ``query_frequency`` raw 30 Hz frames.
Its plots intentionally match the Clean V2 four-frame HPT evaluation layout:
six joints, binary gripper, FK position/rotation error, and temporal derivative
errors.  It is offline-only and never connects to a robot.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from egomimic.eval.eval import Eval
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.robot.piper.piper_kinematics import PiperKinematicsSolver
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix


EMBODIMENT_NAME = "piper_left_arm"
ACTION_KEY = "actions_joints"
PREDICTION_KEY = f"{EMBODIMENT_NAME}_{ACTION_KEY}"


def _episode_name(value) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                "PI05EpisodeEval requires validation batch_size=1; "
                f"episode_hash batch had length {len(value)}"
            )
        value = value[0]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("episode_hash array must contain one item")
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _pose_errors(pred_pose: np.ndarray, gt_pose: np.ndarray):
    position = np.linalg.norm(pred_pose[:, :3] - gt_pose[:, :3], axis=-1)
    pred_rot = _xyzwxyz_to_matrix(pred_pose)[:, :3, :3]
    gt_rot = _xyzwxyz_to_matrix(gt_pose)[:, :3, :3]
    relative = np.swapaxes(pred_rot, -1, -2) @ gt_rot
    cosine = np.clip(
        (np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return position, np.rad2deg(np.arccos(cosine))


def _reconstruct(
    chunks: dict[int, np.ndarray],
    frame_count: int,
    query_frequency: int,
    action_horizon_frames: int,
) -> np.ndarray:
    result = np.full((frame_count, 7), np.nan, dtype=np.float64)
    expected_queries = range(0, frame_count, query_frequency)
    missing = [query for query in expected_queries if query not in chunks]
    if missing:
        raise RuntimeError(f"missing pi0.5 chunks at episode frames {missing[:10]}")
    for query in expected_queries:
        chunk = chunks[query]
        stop = min(query + query_frequency, frame_count)
        for frame in range(query, stop):
            offset = frame - query
            index = min(
                len(chunk) - 1,
                int(round(offset * len(chunk) / action_horizon_frames)),
            )
            result[frame] = chunk[index]
    if not np.isfinite(result).all():
        raise RuntimeError("episode reconstruction left non-finite actions")
    return result


def _metrics(pred, gt, solver):
    joint_delta = pred[:, :6] - gt[:, :6]
    pred_binary = pred[:, 6] >= 0.5
    gt_binary = gt[:, 6] >= 0.5
    tp = np.sum(pred_binary & gt_binary)
    fp = np.sum(pred_binary & ~gt_binary)
    fn = np.sum(~pred_binary & gt_binary)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    pred_pose = solver.fk(pred[:, :6])
    gt_pose = solver.fk(gt[:, :6])
    position, rotation = _pose_errors(pred_pose, gt_pose)
    pred_deg = np.rad2deg(pred[:, :6])
    gt_deg = np.rad2deg(gt[:, :6])
    velocity_error = np.diff(pred_deg, axis=0) - np.diff(gt_deg, axis=0)
    acceleration_error = np.diff(pred_deg, n=2, axis=0) - np.diff(gt_deg, n=2, axis=0)
    jerk_error = np.diff(pred_deg, n=3, axis=0) - np.diff(gt_deg, n=3, axis=0)
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
        "gripper_open_f1": float(
            2 * precision * recall / max(precision + recall, 1e-12)
        ),
        "gripper_predicted_transitions": int(
            np.sum(pred_binary[1:] != pred_binary[:-1])
        ),
        "gripper_gt_transitions": int(np.sum(gt_binary[1:] != gt_binary[:-1])),
        "joint_velocity_error_mae_deg_per_frame": float(
            np.mean(np.abs(velocity_error))
        ),
        "joint_acceleration_error_mae_deg_per_frame2": float(
            np.mean(np.abs(acceleration_error))
        ),
        "joint_jerk_error_mae_deg_per_frame3": float(np.mean(np.abs(jerk_error))),
        "predicted_joint_velocity_abs_p99_deg_per_frame": float(
            np.quantile(np.abs(np.diff(pred_deg, axis=0)), 0.99)
        ),
        "predicted_joint_acceleration_abs_p99_deg_per_frame2": float(
            np.quantile(np.abs(np.diff(pred_deg, n=2, axis=0)), 0.99)
        ),
        "cmd_ee_pose_position_mae_m": float(position.mean()),
        "cmd_ee_pose_position_rmse_m": float(np.sqrt(np.mean(position**2))),
        "cmd_ee_pose_rotation_mae_deg": float(rotation.mean()),
        "cmd_ee_pose_rotation_rmse_deg": float(np.sqrt(np.mean(rotation**2))),
    }, position, rotation


def _plot(path, name, pred, gt, position, rotation, dpi):
    frames = np.arange(len(gt))
    fig, axes = plt.subplots(5, 2, figsize=(13, 13), sharex=True)
    axes = axes.reshape(-1)
    for index in range(6):
        axes[index].plot(frames, np.rad2deg(gt[:, index]), color="black", label="GT")
        axes[index].plot(
            frames,
            np.rad2deg(pred[:, index]),
            color="#d62728",
            label="model",
        )
        axes[index].set_ylabel(f"J{index + 1} (deg)")
        axes[index].grid(alpha=0.25)
    axes[6].step(frames, gt[:, 6] >= 0.5, where="post", color="black", label="GT")
    axes[6].step(
        frames,
        pred[:, 6] >= 0.5,
        where="post",
        color="#1b9e77",
        label="model",
    )
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
    pred_deg = np.rad2deg(pred[:, :6])
    gt_deg = np.rad2deg(gt[:, :6])
    velocity_error = np.mean(
        np.abs(np.diff(pred_deg, axis=0) - np.diff(gt_deg, axis=0)), axis=1
    )
    acceleration_error = np.mean(
        np.abs(np.diff(pred_deg, n=2, axis=0) - np.diff(gt_deg, n=2, axis=0)),
        axis=1,
    )
    axes[9].plot(frames[1:], velocity_error, label="velocity error")
    axes[9].plot(frames[2:], acceleration_error, label="acceleration error")
    axes[9].set_ylabel("mean abs error (deg)")
    axes[9].set_xlabel("episode frame (30 Hz)")
    axes[9].grid(alpha=0.25)
    axes[9].legend(fontsize=8)
    axes[0].legend(fontsize=8)
    axes[6].legend(fontsize=8)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


class PI05EpisodeEval(Eval):
    """Reconstruct and plot complete PIPER episodes from pi0.5 action chunks."""

    def __init__(
        self,
        output_dir: str,
        query_frequency: int = 10,
        action_horizon_frames: int = 30,
        dpi: int = 160,
        limit_val_batches: float = 1.0,
    ):
        if query_frequency <= 0 or query_frequency > action_horizon_frames:
            raise ValueError("query_frequency must be in [1, action_horizon_frames]")
        self.output_dir = Path(output_dir)
        self.query_frequency = int(query_frequency)
        self.action_horizon_frames = int(action_horizon_frames)
        self.dpi = int(dpi)
        self.trainer = None
        self.model = None
        self._episodes: OrderedDict[str, dict] = OrderedDict()
        self.override_dict = {
            "strategy": "auto",
            "limit_train_batches": 0,
            "limit_val_batches": limit_val_batches,
            "check_val_every_n_epoch": 1,
            "profiler": None,
            "max_epochs": 1,
            "min_epochs": 1,
        }

    def on_validation_start(self):
        if not self.trainer.is_global_zero:
            raise RuntimeError("PI05EpisodeEval supports exactly one evaluation process")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._episodes.clear()

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        embodiment_id = get_embodiment_id(EMBODIMENT_NAME)
        if embodiment_id not in batch:
            raise KeyError(f"PIPER embodiment {embodiment_id} is missing from eval batch")
        piper_batch = batch[embodiment_id]
        action = piper_batch.get(ACTION_KEY)
        if not isinstance(action, torch.Tensor) or action.shape[0] != 1:
            raise ValueError(
                "PI05EpisodeEval requires one PIPER sample per validation step; "
                f"got {None if action is None else tuple(action.shape)}"
            )
        if "episode_hash" not in piper_batch:
            raise KeyError("episode_hash is required for episode reconstruction")
        name = _episode_name(piper_batch["episode_hash"])
        record = self._episodes.setdefault(name, {"gt": [], "chunks": {}})
        frame = len(record["gt"])

        gt_batch = self.model.norm_stats.unnormalize(
            {ACTION_KEY: action}, embodiment_id
        )
        gt_chunk = gt_batch[ACTION_KEY][0].detach().float().cpu().numpy()
        record["gt"].append(np.asarray(gt_chunk[0], dtype=np.float64))

        if frame % self.query_frequency == 0:
            with torch.inference_mode():
                predictions = self.model.forward_eval({embodiment_id: piper_batch})
            chunk = predictions[PREDICTION_KEY][0].detach().float().cpu().numpy()
            record["chunks"][frame] = np.asarray(chunk, dtype=np.float64)

    def on_validation_end(self):
        if not self.trainer.is_global_zero:
            return
        solver = PiperKinematicsSolver(device="cpu")
        all_metrics = []
        for number, (name, record) in enumerate(self._episodes.items(), 1):
            gt = np.stack(record["gt"])
            pred = _reconstruct(
                record["chunks"],
                len(gt),
                self.query_frequency,
                self.action_horizon_frames,
            )
            metrics, position, rotation = _metrics(pred, gt, solver)
            metrics["episode"] = name
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            (self.output_dir / f"{stem}_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            _plot(
                self.output_dir / f"{stem}_joint_eval.png",
                f"test: {name} | pi0.5 | qf={self.query_frequency}",
                pred,
                gt,
                position,
                rotation,
                self.dpi,
            )
            np.savez_compressed(
                self.output_dir / f"{stem}_predictions.npz",
                pred=pred,
                gt=gt,
            )
            all_metrics.append(metrics)
            print(f"[{number}/{len(self._episodes)}] {name}: {metrics}", flush=True)

        if not all_metrics:
            raise RuntimeError("no PIPER episodes were evaluated")
        numeric_keys = [
            key
            for key, value in all_metrics[0].items()
            if isinstance(value, (int, float)) and key != "frames"
        ]
        weights = np.asarray([item["frames"] for item in all_metrics], dtype=np.float64)
        summary = {
            "protocol": "episode reconstruction; first qf raw frames per pi0.5 chunk",
            "query_frequency": self.query_frequency,
            "action_horizon_frames": self.action_horizon_frames,
            "episodes": len(all_metrics),
            "frames": int(weights.sum()),
            **{
                key: float(
                    np.average([item[key] for item in all_metrics], weights=weights)
                )
                for key in numeric_keys
            },
        }
        (self.output_dir / "summary_metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        (self.output_dir / "COMPLETED").touch()
        print(json.dumps(summary, indent=2), flush=True)
