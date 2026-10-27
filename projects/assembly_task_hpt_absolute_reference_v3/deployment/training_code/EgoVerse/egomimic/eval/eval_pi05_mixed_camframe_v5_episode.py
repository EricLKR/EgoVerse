"""Independent full-episode evaluation for Mixed CamFrame V5.

This evaluator is deliberately implemented as an ``Eval.run`` hook so that
``trainHydra`` first reconstructs the official frozen pi0.5 base and the V5
normalization contract.  It then overlays the compact V5 checkpoint and runs
deployment-style absolute-time replanning on both held-out domains.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from egomimic.eval.eval import Eval
from egomimic.eval.eval_pi05_episode import (
    ACTION_KEY as PIPER_ACTION_KEY,
    PREDICTION_KEY as PIPER_PREDICTION_KEY,
    _metrics as piper_metrics,
    _plot as plot_piper_episode,
    _reconstruct as reconstruct_piper_episode,
)
from egomimic.pl_utils.pl_data_utils import annotation_collate
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.robot.piper.piper_kinematics import PiperKinematicsSolver
from egomimic.scripts.aria_process.aria_plot_pi05_episode_eval import (
    episode_metrics as human_episode_metrics,
    episode_records as human_episode_records,
    plot_episode as plot_human_episode,
    predict_queries as predict_human_queries,
    reconstruct_episode as reconstruct_human_episode,
)


PIPER_NAME = "piper_left_arm"
HUMAN_NAME = "human_left_arm"
EXPECTED_CHECKPOINT_FORMAT = "egomimic_trainable_state_and_buffers_v5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _episode_records(dataset):
    return [
        {
            "name": name,
            "frames": len(dataset.datasets[name]),
            "indices": list(dataset._global_indices_by_dataset[name]),
        }
        for name in sorted(dataset._global_indices_by_dataset)
    ]


def _weighted_summary(reports: list[dict], metric_names: list[str]) -> dict:
    if not reports:
        raise RuntimeError("independent test produced no episode reports")
    weights = np.asarray([item["frames"] for item in reports], dtype=np.float64)
    return {
        "episodes": len(reports),
        "frames": int(weights.sum()),
        **{
            key: float(np.average([item[key] for item in reports], weights=weights))
            for key in metric_names
        },
    }


def _representative_report(reports: list[dict], metric: str) -> dict:
    median = float(np.median([item[metric] for item in reports]))
    return min(reports, key=lambda item: abs(float(item[metric]) - median))


class PI05MixedCamFrameV5EpisodeEval(Eval):
    """Evaluate the validation-selected V5 checkpoint once on independent test."""

    def __init__(
        self,
        output_dir: str,
        query_frequency: int = 10,
        action_horizon_frames: int = 30,
        human_batch_size: int = 1,
        dpi: int = 180,
        seed: int = 42,
    ):
        if query_frequency <= 0 or query_frequency > action_horizon_frames:
            raise ValueError("query_frequency must be in [1, action_horizon_frames]")
        self.output_dir = Path(output_dir)
        self.query_frequency = int(query_frequency)
        self.action_horizon_frames = int(action_horizon_frames)
        self.human_batch_size = int(human_batch_size)
        self.dpi = int(dpi)
        self.seed = int(seed)
        self.trainer = None
        self.model = None
        self.override_dict = {
            "strategy": "auto",
            "limit_train_batches": 0,
            "limit_val_batches": 0,
            "profiler": None,
            "devices": 1,
            "num_nodes": 1,
        }

    def on_validation_start(self):
        pass

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def on_validation_end(self):
        pass

    def _load_compact_checkpoint(self, model_wrapper, checkpoint_path: Path) -> dict:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_format") != EXPECTED_CHECKPOINT_FORMAT:
            raise RuntimeError(
                "V5 independent test requires the compact V5 checkpoint format; "
                f"got {payload.get('checkpoint_format')!r}"
            )
        incompatible = model_wrapper.load_state_dict(payload["state_dict"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "unexpected V5 checkpoint keys: "
                + ", ".join(incompatible.unexpected_keys[:20])
            )
        loaded = set(payload["state_dict"])
        required_tokens = (
            "egoverse_v5_conditioning.state_adapters.0",
            "egoverse_v5_conditioning.state_adapters.1",
            "egoverse_v5_conditioning.embodiment_embeddings",
            "egoverse_camera_slot_embeddings",
            "joint_residual_scale_rad_vector",
            "gripper_head",
            "egoverse_lora_a",
            "egoverse_lora_b",
        )
        missing_tokens = [token for token in required_tokens if not any(token in k for k in loaded)]
        if missing_tokens:
            raise RuntimeError("compact checkpoint contract missing: " + ", ".join(missing_tokens))
        return payload

    def _evaluate_piper(self, policy, dataset, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        embodiment_id = get_embodiment_id(PIPER_NAME)
        solver = PiperKinematicsSolver(device="cpu")
        reports = []

        for number, record in enumerate(_episode_records(dataset), 1):
            gt = []
            chunks = {}
            print(
                f"[PIPER {number}] {record['name']} ({record['frames']} frames)",
                flush=True,
            )
            for frame, index in enumerate(record["indices"]):
                batch = annotation_collate([dataset[index]])
                processed = policy.model.process_batch_for_training({PIPER_NAME: batch})
                piper_batch = processed[embodiment_id]
                raw_target = policy.model.norm_stats.unnormalize(
                    {PIPER_ACTION_KEY: piper_batch[PIPER_ACTION_KEY].clone()},
                    embodiment_id,
                )[PIPER_ACTION_KEY]
                gt.append(raw_target[0, 0].detach().float().cpu().numpy())
                if frame % self.query_frequency == 0:
                    with torch.inference_mode():
                        prediction = policy.model.forward_eval(processed)[
                            PIPER_PREDICTION_KEY
                        ]
                    chunks[frame] = prediction[0].detach().float().cpu().numpy()

            gt_array = np.asarray(gt, dtype=np.float64)
            pred_array = reconstruct_piper_episode(
                chunks,
                len(gt_array),
                self.query_frequency,
                self.action_horizon_frames,
            )
            metrics, position, rotation = piper_metrics(pred_array, gt_array, solver)
            metrics.update(episode=record["name"], queries=len(chunks))
            stem = _safe_stem(record["name"])
            (output_dir / f"{stem}_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            plot_piper_episode(
                output_dir / f"{stem}_joint_eval.png",
                f"independent test: {record['name']} | V5 | qf={self.query_frequency}",
                pred_array,
                gt_array,
                position,
                rotation,
                self.dpi,
            )
            np.savez_compressed(
                output_dir / f"{stem}_predictions.npz",
                pred=pred_array,
                gt=gt_array,
            )
            reports.append(metrics)

        metric_names = [
            key
            for key, value in reports[0].items()
            if isinstance(value, (int, float)) and key not in {"frames", "queries"}
        ]
        summary = _weighted_summary(reports, metric_names)
        summary.update(
            protocol="full_episode_absolute_time_replan_joint_space",
            query_frequency=self.query_frequency,
            action_horizon_frames=self.action_horizon_frames,
        )
        representative = _representative_report(reports, "joint_mae_deg")
        summary["representative_episode"] = representative["episode"]
        shutil.copy2(
            output_dir / f"{_safe_stem(representative['episode'])}_joint_eval.png",
            output_dir / "piper_representative_joint_eval.png",
        )
        (output_dir / "summary_metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary

    def _evaluate_human(self, policy, dataset, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        records = human_episode_records(dataset)
        for number, record in enumerate(records, 1):
            print(
                f"[HUMAN {number}/{len(records)}] {record['name']} "
                f"({record['frames']} frames)",
                flush=True,
            )
            query_frames = list(range(0, record["frames"], self.query_frequency))
            predictions, targets = predict_human_queries(
                policy,
                dataset,
                record,
                query_frames,
                self.human_batch_size,
            )
            reconstruction = reconstruct_human_episode(
                record,
                predictions,
                targets,
                query_frames,
                self.query_frequency,
                self.action_horizon_frames,
            )
            report = human_episode_metrics(reconstruction)
            report.update(episode=record["name"], queries=len(query_frames))
            stem = _safe_stem(record["name"])
            (output_dir / f"{stem}_metrics.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            plot_human_episode(
                output_dir / f"{stem}_human_episode_eval.png",
                (
                    f"independent test: {record['name']} | V5 full episode | "
                    f"qf={self.query_frequency} | world frame"
                ),
                reconstruction,
                self.dpi,
            )
            np.savez_compressed(
                output_dir / f"{stem}_episode_predictions.npz",
                predicted_world=reconstruction["predicted_world"],
                target_world=reconstruction["target_world"],
                raw_gt_world=reconstruction["raw_gt_world"],
                source_query=reconstruction["source_query"],
                source_model_step=reconstruction["source_model_step"],
            )
            reports.append(report)

        metric_names = [
            "position_mae_m",
            "position_rmse_m",
            "rotation_geodesic_mae_deg",
            "rotation_geodesic_rmse_deg",
            "position_velocity_error_mae_m_per_frame",
            "position_acceleration_error_mae_m_per_frame2",
            "target_reconstruction_position_mae_m",
        ]
        summary = _weighted_summary(reports, metric_names)
        summary.update(
            protocol="full_episode_absolute_time_replan_world_frame",
            query_frequency=self.query_frequency,
            action_horizon_frames=self.action_horizon_frames,
            queries=int(sum(item["queries"] for item in reports)),
        )
        representative = _representative_report(reports, "position_mae_m")
        summary["representative_episode"] = representative["episode"]
        shutil.copy2(
            output_dir
            / f"{_safe_stem(representative['episode'])}_human_episode_eval.png",
            output_dir / "human_representative_episode_eval.png",
        )
        (output_dir / "summary_metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary

    def run(self, trainer, model_wrapper, datamodule, cfg):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = Path(str(cfg.ckpt_path)).resolve()
        payload = self._load_compact_checkpoint(model_wrapper, checkpoint_path)
        model_wrapper.to(device).eval()
        model_wrapper.model.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoint_path.txt").write_text(
            str(checkpoint_path) + "\n", encoding="utf-8"
        )
        piper = self._evaluate_piper(
            model_wrapper,
            datamodule.valid_datasets[PIPER_NAME],
            self.output_dir / "piper",
        )
        human = self._evaluate_human(
            model_wrapper,
            datamodule.valid_datasets[HUMAN_NAME],
            self.output_dir / "human",
        )
        summary = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_format": payload["checkpoint_format"],
            "global_step": int(payload["global_step"]),
            "selection_rule": "lowest validation loss; independent test not used for selection",
            "seed": self.seed,
            "piper": piper,
            "human": human,
        }
        (self.output_dir / "summary_metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (self.output_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return {"V5Test/completed": 1.0}
