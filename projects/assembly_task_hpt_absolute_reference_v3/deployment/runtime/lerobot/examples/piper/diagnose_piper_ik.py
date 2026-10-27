#!/usr/bin/env python
"""Offline diagnostics for Piper Cartesian-to-joint IK.

This script never connects to CAN or a camera.  It separates three questions:

1. Can IK recover an FK pose when seeded with the exact originating joints?
2. Can IK continuously replay a real recorded joint trajectory after FK?
3. For one saved policy prediction, does failure come from the predicted
   rotation, insufficient random retries, or a discontinuous joint branch?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from egomimic.robot.piper.piper_kinematics import (
    PiperKinematicsSolver,
    raw_joint_to_rad,
)
from egomimic.utils.pose_utils import _xyz6d_to_matrix


def load_limits(calibration_json: Path) -> tuple[np.ndarray, np.ndarray]:
    calibration = json.loads(calibration_json.read_text())
    low = np.asarray(
        [calibration[f"joint_{index}"]["range_min"] * 1e-3 for index in range(6)],
        dtype=np.float64,
    )
    high = np.asarray(
        [calibration[f"joint_{index}"]["range_max"] * 1e-3 for index in range(6)],
        dtype=np.float64,
    )
    return low, high


def pose_from_matrix(matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.concatenate([matrix[:3, 3], quat_xyzw[[3, 0, 1, 2]]])


def matrix_from_pose(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = pose[:3]
    matrix[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    return matrix


def normalized_delta(
    q: np.ndarray,
    reference: np.ndarray,
    low_rad: np.ndarray,
    high_rad: np.ndarray,
) -> np.ndarray:
    return (q - reference) / (high_rad - low_rad) * 200.0


def solve_with_diagnostics(
    solver: PiperKinematicsSolver,
    target_pose: np.ndarray,
    seed_q: np.ndarray,
    *,
    num_retries: int,
    random_seed: int,
    pos_tolerance: float = 2e-3,
    rot_tolerance: float = 3e-2,
    seed_noise_std: float = 0.15,
    max_iterations: int = 100,
    max_normalized_joint_step: float | None = None,
    position_weight: float = 1.0,
    orientation_weight: float = 1.0,
) -> dict:
    """Replicate deployment IK while retaining rotation/candidate details."""
    device = solver.device
    position = np.asarray(target_pose[:3], dtype=np.float32)
    quat_wxyz = np.asarray(target_pose[3:7], dtype=np.float32)
    rotation = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix().astype(np.float32)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    target = pk.Transform3d(matrix=torch.from_numpy(matrix).unsqueeze(0).to(device))

    global_low = solver.chain.low.clone()
    global_high = solver.chain.high.clone()
    seed_t = torch.as_tensor(seed_q, dtype=torch.float32, device=device)
    if max_normalized_joint_step is None:
        low = global_low
        high = global_high
    else:
        radius = (global_high - global_low) * (max_normalized_joint_step / 200.0)
        low = torch.maximum(global_low, seed_t - radius)
        high = torch.minimum(global_high, seed_t + radius)
    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    noise = torch.randn(
        num_retries,
        solver.NUM_JOINTS,
        generator=generator,
        device=device,
    ) * seed_noise_std
    retry_configs = torch.clamp(seed_t.unsqueeze(0) + noise, low, high)
    retry_configs[0] = torch.clamp(seed_t, low, high)

    try:
        # PseudoInverseIK's projected joint-limit refinement consults the
        # chain limits.  Temporarily narrow them to the local trust region;
        # restore them immediately after the solve.
        solver.chain.low = low
        solver.chain.high = high
        ik_solver = pk.PseudoInverseIK(
            solver.chain,
            retry_configs=retry_configs,
            pos_tolerance=pos_tolerance,
            rot_tolerance=rot_tolerance,
            max_iterations=max_iterations,
            joint_limits=torch.stack([low, high], dim=1),
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            clamp_to_limits=True,
            early_stopping_no_improvement="all",
            early_stopping_no_improvement_patience=5,
        )
        with torch.no_grad():
            result = ik_solver.solve(target)
    finally:
        solver.chain.low = global_low
        solver.chain.high = global_high

    solutions = result.solutions[0]
    converged = result.converged[0]
    err_pos = result.err_pos[0]
    err_rot = result.err_rot[0]
    total_error = err_pos + err_rot
    candidates = torch.nonzero(converged).squeeze(-1)
    if candidates.numel():
        span = (high - low).clamp_min(1e-6)
        distance = ((solutions[candidates] - seed_t) / span).norm(dim=-1)
        selected = candidates[distance.argmin()]
    else:
        selected = total_error.argmin()

    selected_index = int(selected.item())
    q = solutions[selected].cpu().numpy().astype(np.float64)
    retry_rows = []
    for index in range(num_retries):
        candidate_q = solutions[index].cpu().numpy().astype(np.float64)
        retry_rows.append(
            {
                "retry": index,
                "converged": bool(converged[index].item()),
                "position_error_m": float(err_pos[index].item()),
                "rotation_error_rad": float(err_rot[index].item()),
                "seed_distance_normalized_rms": float(
                    np.sqrt(
                        np.mean(
                            normalized_delta(
                                candidate_q,
                                seed_q,
                                low.cpu().numpy(),
                                high.cpu().numpy(),
                            )
                            ** 2
                        )
                    )
                ),
            }
        )
    return {
        "converged": bool(converged[selected].item()),
        "selected_retry": selected_index,
        "position_error_m": float(err_pos[selected].item()),
        "rotation_error_rad": float(err_rot[selected].item()),
        "solution_q_rad": q.tolist(),
        "converged_candidates": int(converged.sum().item()),
        "retries": retry_rows,
    }


def summarize_steps(steps: list[dict]) -> dict:
    successes = [row for row in steps if row["converged"]]
    longest_failure_run = 0
    current_failure_run = 0
    for row in steps:
        if row["converged"]:
            current_failure_run = 0
        else:
            current_failure_run += 1
            longest_failure_run = max(longest_failure_run, current_failure_run)
    return {
        "steps": len(steps),
        "successes": len(successes),
        "success_rate": len(successes) / len(steps) if steps else 0.0,
        "longest_consecutive_failure_run": longest_failure_run,
        "max_selected_position_error_m": max(
            (row["position_error_m"] for row in steps), default=None
        ),
        "max_selected_rotation_error_rad": max(
            (row["rotation_error_rad"] for row in steps), default=None
        ),
        "max_seed_joint_rms_normalized": max(
            (row["seed_joint_rms_normalized"] for row in steps), default=None
        ),
        "max_seed_joint_abs_normalized": max(
            (row["seed_joint_max_abs_normalized"] for row in steps), default=None
        ),
    }


def evaluate_targets(
    solver: PiperKinematicsSolver,
    targets: list[np.ndarray],
    initial_seed: np.ndarray,
    low_rad: np.ndarray,
    high_rad: np.ndarray,
    *,
    num_retries: int,
    update_seed_on_success: bool,
    label: str,
    max_normalized_joint_step: float | None = None,
    orientation_weight: float = 1.0,
    rot_tolerance: float = 3e-2,
) -> dict:
    # Recorded feedback may sit a fraction of a degree beyond the calibration
    # extrema; deployment IK clamps the seed to the calibrated box too.
    seed_q = np.clip(initial_seed.copy(), low_rad, high_rad)
    rows = []
    for index, target in enumerate(targets):
        solved = solve_with_diagnostics(
            solver,
            target,
            seed_q,
            num_retries=num_retries,
            random_seed=1000 + index,
            max_normalized_joint_step=max_normalized_joint_step,
            orientation_weight=orientation_weight,
            rot_tolerance=rot_tolerance,
        )
        q = np.asarray(solved.pop("solution_q_rad"), dtype=np.float64)
        delta = normalized_delta(q, seed_q, low_rad, high_rad)
        row = {
            "index": index,
            **solved,
            "seed_joint_delta_normalized": delta.tolist(),
            "seed_joint_rms_normalized": float(np.sqrt(np.mean(delta**2))),
            "seed_joint_max_abs_normalized": float(np.max(np.abs(delta))),
        }
        rows.append(row)
        if row["converged"] and update_seed_on_success:
            seed_q = q
    return {"label": label, "summary": summarize_steps(rows), "steps": rows}


def evaluate_production_local_ik(
    solver: PiperKinematicsSolver,
    targets: list[np.ndarray],
    initial_seed: np.ndarray,
    low_rad: np.ndarray,
    high_rad: np.ndarray,
) -> dict:
    """Exercise the exact public IK method now used by deployment."""
    seed_q = np.clip(initial_seed.copy(), low_rad, high_rad)
    rows = []
    for index, target in enumerate(targets):
        q, converged, position_error = solver.ik(
            target,
            seed_joint_angles_rad=seed_q,
            num_retries=8,
            pos_tolerance=2e-3,
            rot_tolerance=3e-2,
            seed_noise_std=0.15,
            max_normalized_joint_step=15.0,
            position_weight=1.0,
            orientation_weight=1.0,
        )
        diagnostics = dict(solver.last_ik_diagnostics or {})
        delta = normalized_delta(q, seed_q, low_rad, high_rad)
        rows.append(
            {
                "index": index,
                "converged": bool(converged),
                "selected_retry": diagnostics.get("selected_retry"),
                "position_error_m": float(position_error),
                "rotation_error_rad": float(diagnostics.get("rotation_error_rad", 0.0)),
                "converged_candidates": int(diagnostics.get("converged_candidates", 0)),
                "seed_joint_delta_normalized": delta.tolist(),
                "seed_joint_rms_normalized": float(np.sqrt(np.mean(delta**2))),
                "seed_joint_max_abs_normalized": float(np.max(np.abs(delta))),
                "retries": [],
            }
        )
        if converged:
            seed_q = np.asarray(q, dtype=np.float64)
    return {
        "label": "production local-continuity IK",
        "summary": summarize_steps(rows),
        "steps": rows,
    }


def evaluate_position_only_targets(
    solver: PiperKinematicsSolver,
    target_positions: list[np.ndarray],
    initial_seed: np.ndarray,
    low_rad: np.ndarray,
    high_rad: np.ndarray,
) -> dict:
    """Track XYZ only from the previous solution, leaving orientation free."""
    seed_q = np.clip(initial_seed.copy(), low_rad, high_rad)
    rows = []
    span_rad = high_rad - low_rad
    proximity_weight = 1e-4
    for index, target_xyz in enumerate(target_positions):
        def residual(q: np.ndarray) -> np.ndarray:
            position_residual = solver.fk(q)[:3] - target_xyz
            # Position-only IK is underdetermined.  A weak normalized-joint
            # residual removes arbitrary null-space drift while keeping the
            # XYZ objective dominant.
            proximity_residual = np.sqrt(proximity_weight) * (q - seed_q) / span_rad
            return np.concatenate([position_residual, proximity_residual])

        result = least_squares(
            residual,
            seed_q,
            bounds=(low_rad, high_rad),
            method="trf",
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
            # FK runs in float32; scipy's default ~1e-8 finite-difference
            # step is swallowed by float32 precision and yields a false zero
            # Jacobian.  One milliradian is still small mechanically while
            # producing a stable numerical derivative.
            diff_step=1e-3,
            max_nfev=300,
        )
        q = np.asarray(result.x, dtype=np.float64)
        position_error = float(np.linalg.norm(solver.fk(q)[:3] - target_xyz))
        delta = normalized_delta(q, seed_q, low_rad, high_rad)
        converged = bool(position_error <= 2e-3)
        row = {
            "index": index,
            "converged": converged,
            "selected_retry": 0,
            "position_error_m": position_error,
            "rotation_error_rad": 0.0,
            "converged_candidates": int(converged),
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_nfev": int(result.nfev),
            "seed_joint_delta_normalized": delta.tolist(),
            "seed_joint_rms_normalized": float(np.sqrt(np.mean(delta**2))),
            "seed_joint_max_abs_normalized": float(np.max(np.abs(delta))),
            "retries": [],
        }
        rows.append(row)
        if converged:
            seed_q = q
    return {
        "label": "model XYZ only; orientation unconstrained",
        "summary": summarize_steps(rows),
        "steps": rows,
    }


def roundtrip_test(
    solver: PiperKinematicsSolver,
    q_samples: np.ndarray,
    low_rad: np.ndarray,
    high_rad: np.ndarray,
) -> dict:
    rows = []
    for index, q in enumerate(q_samples):
        target = solver.fk(q)
        solved = solve_with_diagnostics(
            solver,
            target,
            q,
            num_retries=8,
            random_seed=2000 + index,
        )
        q_solved = np.asarray(solved.pop("solution_q_rad"), dtype=np.float64)
        delta = normalized_delta(q_solved, q, low_rad, high_rad)
        rows.append(
            {
                "index": index,
                **solved,
                "seed_joint_delta_normalized": delta.tolist(),
                "seed_joint_rms_normalized": float(np.sqrt(np.mean(delta**2))),
                "seed_joint_max_abs_normalized": float(np.max(np.abs(delta))),
            }
        )
    return {"summary": summarize_steps(rows), "steps": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory-npz", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    low_deg, high_deg = load_limits(args.calibration_json)
    low_rad = np.radians(low_deg)
    high_rad = np.radians(high_deg)
    solver = PiperKinematicsSolver(
        device="cpu", joint_limits_rad=(low_rad, high_rad)
    )

    manifest = json.loads(args.manifest.read_text())
    prediction_npz = args.manifest.parent / manifest["predictions_npz"]
    with np.load(prediction_npz, allow_pickle=False) as archive:
        current_raw = np.asarray(archive["input_observation_state"], dtype=np.float64).reshape(-1)
        prediction = np.asarray(archive["full_model_chunk_unnormalized"], dtype=np.float64)
    current_q = raw_joint_to_rad(current_raw[:6])
    current_pose = solver.fk(current_q)

    with np.load(args.trajectory_npz, allow_pickle=False) as archive:
        state_key = "observation.state" if "observation.state" in archive.files else "state"
        recorded_raw = np.asarray(archive[state_key], dtype=np.float64)
    recorded_q = raw_joint_to_rad(recorded_raw[:, :6])
    recorded_targets = [solver.fk(q) for q in recorded_q]

    query_indices = [int(index) for index in manifest["query_indices"]]
    full_pose_targets = []
    fixed_rotation_targets = []
    current_matrix = matrix_from_pose(current_pose)
    for model_index in query_indices:
        predicted_matrix = _xyz6d_to_matrix(prediction[model_index : model_index + 1, :9])[0]
        full_pose_targets.append(pose_from_matrix(predicted_matrix))
        fixed_rotation = predicted_matrix.copy()
        fixed_rotation[:3, :3] = current_matrix[:3, :3]
        fixed_rotation_targets.append(pose_from_matrix(fixed_rotation))

    report = {
        "inputs": {
            "manifest": str(args.manifest),
            "trajectory_npz": str(args.trajectory_npz),
            "calibration_json": str(args.calibration_json),
            "inference_seed": manifest.get("inference_seed"),
            "query_indices": query_indices,
            "current_joint_rad": current_q.tolist(),
            "current_pose_xyzwxyz": current_pose.tolist(),
            "recorded_start_joint_rad": recorded_q[0].tolist(),
            "current_to_recorded_start_joint_delta_normalized": normalized_delta(
                current_q, recorded_q[0], low_rad, high_rad
            ).tolist(),
        },
        "current_and_recorded_fk_ik_roundtrip": roundtrip_test(
            solver,
            np.concatenate([current_q[None], recorded_q], axis=0),
            low_rad,
            high_rad,
        ),
        "recorded_trajectory_sequential_ik": evaluate_targets(
            solver,
            recorded_targets,
            recorded_q[0],
            low_rad,
            high_rad,
            num_retries=8,
            update_seed_on_success=True,
            label="recorded FK targets, previous solved joint seed",
        ),
        "prediction_full_pose_8_retries": evaluate_targets(
            solver,
            full_pose_targets,
            current_q,
            low_rad,
            high_rad,
            num_retries=8,
            update_seed_on_success=True,
            label="model XYZ + model rotation",
        ),
        "prediction_full_pose_64_retries": evaluate_targets(
            solver,
            full_pose_targets,
            current_q,
            low_rad,
            high_rad,
            num_retries=64,
            update_seed_on_success=True,
            label="model XYZ + model rotation",
        ),
        "prediction_local_full_pose_8_retries": evaluate_targets(
            solver,
            full_pose_targets,
            current_q,
            low_rad,
            high_rad,
            num_retries=8,
            update_seed_on_success=True,
            label="local trust-region model XYZ + model rotation",
            max_normalized_joint_step=15.0,
        ),
        "prediction_local_soft_rotation_8_retries": evaluate_targets(
            solver,
            full_pose_targets,
            current_q,
            low_rad,
            high_rad,
            num_retries=8,
            update_seed_on_success=True,
            label="local trust-region, position-priority model pose",
            max_normalized_joint_step=15.0,
            orientation_weight=0.25,
            rot_tolerance=8e-2,
        ),
        "prediction_production_local_ik": evaluate_production_local_ik(
            solver,
            full_pose_targets,
            current_q,
            low_rad,
            high_rad,
        ),
        "prediction_full_pose_from_recorded_start_8_retries": evaluate_targets(
            solver,
            full_pose_targets,
            recorded_q[0],
            low_rad,
            high_rad,
            num_retries=8,
            update_seed_on_success=True,
            label="model XYZ + model rotation, seeded from recorded trajectory start",
        ),
        "prediction_fixed_current_rotation_8_retries": evaluate_targets(
            solver,
            fixed_rotation_targets,
            current_q,
            low_rad,
            high_rad,
            num_retries=8,
            update_seed_on_success=True,
            label="model XYZ + current measured rotation",
        ),
        "prediction_fixed_current_rotation_64_retries": evaluate_targets(
            solver,
            fixed_rotation_targets,
            current_q,
            low_rad,
            high_rad,
            num_retries=64,
            update_seed_on_success=True,
            label="model XYZ + current measured rotation",
        ),
        "prediction_position_only": evaluate_position_only_targets(
            solver,
            [target[:3] for target in full_pose_targets],
            current_q,
            low_rad,
            high_rad,
        ),
        "prediction_position_only_from_recorded_start": evaluate_position_only_targets(
            solver,
            [target[:3] for target in full_pose_targets],
            recorded_q[0],
            low_rad,
            high_rad,
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    compact = {
        key: value["summary"]
        for key, value in report.items()
        if isinstance(value, dict) and "summary" in value
    }
    print(json.dumps(compact, indent=2, allow_nan=False))
    print(f"report={args.output_json}")


if __name__ == "__main__":
    main()
