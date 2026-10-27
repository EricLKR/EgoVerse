#!/usr/bin/env python
"""Offline sweep of deterministic flow/IK seeds on one saved live observation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.policies.egoverse_hpt.configuration_egoverse_hpt import EgoVerseHPTConfig
from lerobot.policies.egoverse_hpt.modeling_egoverse_hpt import EgoVerseHPTPolicy
from lerobot.policies.utils import prepare_observation_for_inference


def load_calibration(path: Path) -> list[tuple[float, float]]:
    calibration = json.loads(path.read_text())
    return [
        (
            float(calibration[f"joint_{index}"]["range_min"]) / 1000.0,
            float(calibration[f"joint_{index}"]["range_max"]) / 1000.0,
        )
        for index in range(6)
    ]


def normalized_arm_state(
    raw_state: np.ndarray, limits: list[tuple[float, float]]
) -> np.ndarray:
    raw_deg = np.asarray(raw_state, dtype=np.float64).reshape(-1)[:6] * 1e-3
    low = np.asarray([item[0] for item in limits], dtype=np.float64)
    high = np.asarray([item[1] for item in limits], dtype=np.float64)
    return (raw_deg - low) / (high - low) * 200.0 - 100.0


def best_startup_candidate(
    report: dict,
    current_arm: np.ndarray,
    *,
    min_suffix_actions: int,
    max_alignment_joint_rms: float,
    max_alignment_xyz_m: float,
    startup_max_joint_abs: float,
    startup_max_joint_rms: float,
) -> dict | None:
    steps = report.get("steps") or []
    proprio = np.asarray(report.get("proprio_physical", []), dtype=np.float64)
    current_xyz = proprio.reshape(-1, proprio.shape[-1])[0, :3]
    candidates = []
    for offset, step in enumerate(steps):
        suffix = len(steps) - offset
        command = np.asarray(step.get("normalized_joint_command", []), dtype=np.float64)
        xyz = np.asarray(step.get("predicted_xyz_base_m", []), dtype=np.float64)
        if command.shape != (6,) or xyz.shape != (3,) or suffix < min_suffix_actions:
            continue
        delta = command - current_arm
        rms = float(np.sqrt(np.mean(delta**2)))
        max_abs = float(np.max(np.abs(delta)))
        xyz_distance = float(np.linalg.norm(xyz - current_xyz))
        aligned = rms <= max_alignment_joint_rms and xyz_distance <= max_alignment_xyz_m
        startup_safe = rms <= startup_max_joint_rms and max_abs <= startup_max_joint_abs
        ik_safe = bool(step.get("ik_converged", False))
        candidates.append(
            {
                "offset": offset,
                "suffix_actions": suffix,
                "joint_rms": rms,
                "joint_max_abs": max_abs,
                "xyz_distance_m": xyz_distance,
                "ik_safe": ik_safe,
                "aligned": bool(aligned and ik_safe),
                "startup_safe": startup_safe,
                "score": rms / max_alignment_joint_rms + xyz_distance / max_alignment_xyz_m,
            }
        )
    aligned = [item for item in candidates if item["aligned"]]
    return None if not aligned else min(aligned, key=lambda item: item["score"])


def max_joint_jump_rms(steps: list[dict]) -> float | None:
    commands = np.asarray(
        [step["normalized_joint_command"] for step in steps if step.get("ik_converged")],
        dtype=np.float64,
    )
    if len(commands) < 2:
        return None
    return float(np.max(np.sqrt(np.mean(np.diff(commands, axis=0) ** 2, axis=1))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--query-frequency", type=int, default=25)
    parser.add_argument("--min-suffix-actions", type=int, default=11)
    parser.add_argument("--max-alignment-joint-rms", type=float, default=15.0)
    parser.add_argument("--max-alignment-xyz-m", type=float, default=0.1)
    parser.add_argument("--startup-max-joint-abs", type=float, default=10.0)
    parser.add_argument("--startup-max-joint-rms", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    run_dir = args.manifest.parent
    with np.load(run_dir / manifest["predictions_npz"], allow_pickle=False) as archive:
        state = np.asarray(archive["input_observation_state"], dtype=np.float32)
    front = np.asarray(Image.open(run_dir / manifest["front_png"]).convert("RGB"))
    wrist = np.asarray(Image.open(run_dir / manifest["wrist_png"]).convert("RGB"))
    observation = {
        "observation.state": state,
        "observation.images.435if": front,
        "observation.images.405": wrist,
    }
    device = torch.device(args.device)
    batch = prepare_observation_for_inference(
        observation, device, task=None, robot_type="piper"
    )
    limits = load_calibration(args.calibration_json)
    current_arm = normalized_arm_state(state, limits)
    config = EgoVerseHPTConfig(
        ckpt_path=args.checkpoint,
        joint_limits_deg=limits,
        device=args.device,
        action_frame="absolute",
        action_horizon_frames=30,
        query_frequency=args.query_frequency,
        num_inference_steps=10,
        normalize_proprio_input=True,
        workspace_xyz_min=[0.057787, -0.348144, 0.121358],
        workspace_xyz_max=[0.319566, 0.014069, 0.420660],
    )
    policy = EgoVerseHPTPolicy(config)

    results = []
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        policy.config.inference_seed = seed
        policy.reset()
        started = time.perf_counter()
        error = None
        try:
            policy.predict_action_chunk(batch)
        except RuntimeError as exc:
            error = str(exc)
        report = policy.last_prediction_report or {}
        steps = report.get("steps") or []
        startup = best_startup_candidate(
            report,
            current_arm,
            min_suffix_actions=args.min_suffix_actions,
            max_alignment_joint_rms=args.max_alignment_joint_rms,
            max_alignment_xyz_m=args.max_alignment_xyz_m,
            startup_max_joint_abs=args.startup_max_joint_abs,
            startup_max_joint_rms=args.startup_max_joint_rms,
        )
        result = {
            "seed": seed,
            "accepted": error is None,
            "deployable": bool(error is None and startup and startup["startup_safe"]),
            "processed_steps": len(steps),
            "ik_failures": sum(not step.get("ik_converged", False) for step in steps),
            "startup": startup,
            "max_joint_jump_rms": max_joint_jump_rms(steps),
            "elapsed_ms": (time.perf_counter() - started) * 1e3,
            "error": error,
        }
        results.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)

    deployable = [item for item in results if item["deployable"]]
    deployable.sort(
        key=lambda item: (
            item["startup"]["joint_rms"],
            item["startup"]["joint_max_abs"],
            item["ik_failures"],
            float("inf") if item["max_joint_jump_rms"] is None else item["max_joint_jump_rms"],
            item["seed"],
        )
    )
    print(
        json.dumps(
            {
                "best": deployable[:10],
                "accepted_count": sum(item["accepted"] for item in results),
                "deployable_count": len(deployable),
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
