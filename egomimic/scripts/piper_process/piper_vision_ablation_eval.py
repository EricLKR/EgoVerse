#!/usr/bin/env python3
"""Offline Piper evaluation entry point for the three vision variants.

Validation is the default and the only split allowed for model/checkpoint
selection.  Test access requires two explicit acknowledgements so it cannot be
opened accidentally during ablation iteration.

Unlike the older Piper-only evaluator, ground truth is read after the data
config's transform list.  This is required for the mixed HPT runs, whose Piper
targets are expressed in the calibrated front-camera frame rather than raw
``base_link`` coordinates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


VARIANTS = {
    "resnet18": {
        "valid": "piper_human_cotrain_aria0919.yaml",
        "test": "piper_solo_teddy_0912_test_camframe.yaml",
    },
    "dinov3_front": {
        "valid": "piper_human_cotrain_aria0919_dinov3_front.yaml",
        "test": "piper_solo_teddy_0912_test_dinov3_front.yaml",
    },
    "dinov3_both": {
        "valid": "piper_human_cotrain_aria0919_dinov3_both.yaml",
        "test": "piper_solo_teddy_0912_test_dinov3_both.yaml",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--selection-complete", action="store_true")
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--query-stride", type=int, default=10)
    parser.add_argument("--query-frequency", type=int, default=10)
    parser.add_argument("--action-horizon-frames", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def _validate_split(args: argparse.Namespace) -> None:
    if args.split == "test" and not (args.allow_test and args.selection_complete):
        raise SystemExit(
            "Held-out test is locked. Finish variant/checkpoint selection on validation, "
            "then pass both --selection-complete and --allow-test."
        )


def _install_transformed_ground_truth() -> Path:
    from egomimic.scripts.piper_process import piper_plot_checkpoint_eval as evaluator

    def transformed_ground_truth(record: dict) -> np.ndarray:
        # All three mixed-training configs use Piper.cartesian_camframe.  Do
        # the same vectorized SE(3) transform here instead of indexing the
        # leaf once per frame (which would needlessly decode RGB/DINO arrays).
        from egomimic.rldb.embodiment.piper import Piper
        from egomimic.utils.pose_utils import _matrix_to_xyz6d, _xyzwxyz_to_matrix

        frame_count = record["frames"]
        store = record["leaf"].episode_reader._store
        pose_base = np.asarray(store["obs_ee_pose"][:frame_count], dtype=np.float64)
        gripper = np.asarray(store["cmd_gripper"][:frame_count], dtype=np.float64).reshape(-1, 1)
        world_from_camera = next(iter(Piper.EXTRINSICS.values()))
        camera_from_world = np.linalg.inv(world_from_camera)
        camera_from_ee = camera_from_world[None, :, :] @ _xyzwxyz_to_matrix(pose_base)
        pose_camera = _matrix_to_xyz6d(camera_from_ee)
        gt = np.concatenate([pose_camera, gripper], axis=-1).astype(np.float32)
        if gt.shape != (frame_count, 10):
            raise ValueError(
                f"Unexpected transformed GT shape for {record['name']}: {gt.shape}"
            )
        return gt

    evaluator._ground_truth = transformed_ground_truth
    return Path(evaluator.__file__).resolve()


def main() -> None:
    args = _parse_args()
    _validate_split(args)
    script = _install_transformed_ground_truth()
    config_dir = Path(__file__).resolve().parents[2] / "hydra_configs" / "data"
    data_config = config_dir / VARIANTS[args.vision_variant][args.split]
    expected_task = "teddy_0912_6" if args.split == "valid" else "teddy_0912_7"

    # Importing and calling the old main after replacing _ground_truth would
    # require mutating this wrapper's argv.  A child process is intentionally
    # avoided because the patch must be active, so call it in-process.
    from egomimic.scripts.piper_process import piper_plot_checkpoint_eval as evaluator

    forwarded = [
        str(script),
        "--checkpoint", str(args.checkpoint),
        "--data-config", str(data_config),
        "--output-dir", str(args.output_dir),
        "--split", args.split,
        "--expected-task-name", expected_task,
        "--query-stride", str(args.query_stride),
        "--query-frequency", str(args.query_frequency),
        "--action-horizon-frames", str(args.action_horizon_frames),
        "--batch-size", str(args.batch_size),
        "--num-inference-steps", str(args.num_inference_steps),
        "--device", args.device,
        "--seed", str(args.seed),
        "--dpi", str(args.dpi),
    ]
    if args.max_episodes is not None:
        forwarded += ["--max-episodes", str(args.max_episodes)]
    for episode in args.episode:
        forwarded += ["--episode", episode]
    if args.split == "test":
        forwarded.append("--allow-test")

    old_argv = sys.argv
    try:
        sys.argv = forwarded
        evaluator.main()
    finally:
        sys.argv = old_argv

    summary_path = args.output_dir / "summary_metrics.json"
    summary = json.loads(summary_path.read_text())
    summary["vision_variant"] = args.vision_variant
    summary["selection_policy"] = (
        "validation-only selection" if args.split == "valid" else "frozen final test"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
