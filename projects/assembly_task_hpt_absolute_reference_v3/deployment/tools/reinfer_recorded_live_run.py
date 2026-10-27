#!/usr/bin/env python3
"""Re-run HPT inference on a recorded live rollout without opening robot I/O.

The asynchronous deployment log already contains every complete 30-step model
chunk.  This utility additionally reconstructs each inference request from the
saved camera JPEGs and raw proprioception, runs the same checkpoint again, and
compares the repeated chunk with the live chunk.  It also reports which action
offsets RTC actually consumed from every plan.

No Piper, camera, or CAN interface is constructed or connected by this file.
Its only external side effects are result files under --output-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = BUNDLE_ROOT / "robot"
if str(ROBOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOT_DIR))

import deploy_assembly_new1_hpt_joint as legacy  # noqa: E402
from joint_space_adapter import JointSpaceAdapter  # noqa: E402
from egomimic.robot.piper.piper_kinematics import PiperKinematicsSolver  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="0 repeats every recorded request; otherwise stop after N requests",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise RuntimeError(f"empty run log: {path}")
    return rows


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if rgb.shape != (480, 640, 3):
        raise RuntimeError(f"invalid recorded RGB image {path}: {rgb.shape}")
    return rgb


def checkpoint_args(
    start: dict[str, Any], first_report: dict[str, Any], device: str
) -> SimpleNamespace:
    trajectory_filter = first_report["trajectory_filter"]
    return SimpleNamespace(
        checkpoint=Path(start["checkpoint"]),
        device=device,
        model_action_horizon=int(start["checkpoint_contract"]["model_action_horizon"]),
        query_frequency=int(start["query_frequency"]),
        inference_steps=int(start["inference_steps"]),
        seed=int(start["seed"]),
        allow_legacy_joint_head=False,
        gripper_threshold=0.5,
        joint_limit_margin_deg=0.0,
        max_joint_command_step=float(trajectory_filter["max_joint_command_step"]),
        max_joint_command_accel=float(trajectory_filter["max_joint_command_accel"]),
        temporal_ensemble=bool(first_report["temporal_ensemble"]),
        temporal_replan_interval=int(first_report["temporal_replan_interval"]),
        temporal_ensemble_decay=float(first_report["temporal_ensemble_decay"]),
        chunk_smoothing=bool(first_report["chunk_smoothing"]),
        chunk_smoothing_window=int(first_report["chunk_smoothing_window"]),
        chunk_smoothing_sigma=float(first_report["chunk_smoothing_sigma"]),
    )


def request_records(
    frame_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    results: dict[int, dict[str, Any]] = {}
    for row in frame_rows:
        request = row.get("async_request")
        if request:
            requests.append(
                {
                    "request_id": int(request["request_id"]),
                    "request_tick": int(request["request_tick"]),
                    "frame": int(row["frame"]),
                    "frame_row": row,
                }
            )
        result = row.get("async_result")
        report = row.get("new_chunk_report")
        if result and report:
            results[int(result["plan_id"])] = {
                "frame": int(row["frame"]),
                "result": result,
                "report": report,
            }
    missing = [item["request_id"] for item in requests if item["request_id"] not in results]
    if missing:
        raise RuntimeError(f"recorded requests lack accepted prediction reports: {missing}")
    return requests, results


def active_offsets(frame_rows: list[dict[str, Any]]) -> dict[int, list[int]]:
    offsets: dict[int, list[int]] = {}
    for row in frame_rows:
        status = row.get("queue_status") or {}
        plan_id = status.get("active_plan_id")
        offset = status.get("active_plan_offset")
        if plan_id is None or offset is None:
            continue
        offsets.setdefault(int(plan_id), []).append(int(offset))
    return offsets


def physical_chunk(report: dict[str, Any]) -> np.ndarray:
    chunk = np.asarray(report["selected_physical_actions"], dtype=np.float64)
    if chunk.shape != (30, 7) or not np.isfinite(chunk).all():
        raise RuntimeError(f"invalid decoded physical chunk: {chunk.shape}")
    return chunk


def fk_z_mm(solver: PiperKinematicsSolver, joints_rad: np.ndarray) -> np.ndarray:
    return np.asarray(solver.fk(joints_rad), dtype=np.float64)[..., 2] * 1000.0


def main() -> None:
    args = parse_args()
    if args.max_requests < 0:
        raise ValueError("--max-requests must be non-negative")
    rows = load_jsonl(args.run_log)
    start = next((row for row in rows if row.get("event") == "start"), None)
    if start is None:
        raise RuntimeError("run log has no start record")
    frame_rows = [
        row
        for row in rows
        if isinstance(row.get("frame"), int) and row.get("mode") == "execute"
    ]
    if not frame_rows:
        raise RuntimeError("run log has no execute frames")
    requests, live_results = request_records(frame_rows)
    if args.max_requests:
        requests = requests[: args.max_requests]
    first_report = live_results[requests[0]["request_id"]]["report"]

    checkpoint = Path(start["checkpoint"])
    expected_sha = str(start["checkpoint_sha256"])
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"checkpoint SHA mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    image_summary = args.image_dir / "summary.json"
    if image_summary.is_file():
        summary = json.loads(image_summary.read_text(encoding="utf-8"))
        if int(summary.get("dropped", 0)) != 0 or int(summary.get("errors", 0)) != 0:
            raise RuntimeError(f"recorded image stream is incomplete: {summary}")

    model_calibration = Path(start["model_calibration_json"])
    joint_limits, gripper_stroke_mm = legacy.load_calibration(model_calibration)
    adapter = JointSpaceAdapter.from_files(
        mapping_path=Path(start["role_mapping_json"]),
        model_calibration_path=model_calibration,
        hardware_calibration_path=Path(start["hardware_calibration_json"]),
        joint_zero_offset_path=Path(start["joint_zero_offset_json"]),
        allow_experimental_joint_mapping=bool(start["allow_experimental_joint_mapping"]),
        mapping_mode=str(start["joint_space_adapter"]["mapping"]),
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(int(start["seed"]))
    np.random.seed(int(start["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(start["seed"]))

    policy = legacy.make_policy(
        checkpoint_args(start, first_report, str(device)),
        joint_limits,
        gripper_stroke_mm,
    )
    solver = PiperKinematicsSolver(device="cpu")
    plan_offsets = active_offsets(frame_rows)
    comparisons: list[dict[str, Any]] = []

    for ordinal, request in enumerate(requests, start=1):
        request_id = request["request_id"]
        frame = request["frame"]
        frame_row = request["frame_row"]
        raw_state = np.asarray(frame_row["raw_state"], dtype=np.float32)
        if raw_state.shape != (7,) or not np.isfinite(raw_state).all():
            raise RuntimeError(f"invalid raw state at frame {frame}: {raw_state}")
        observation: dict[str, Any] = {
            **{f"joint_{index}.pos": float(raw_state[index]) for index in range(7)},
            "435if": read_rgb(args.image_dir / "front" / f"frame_{frame:06d}.jpg"),
            "405": read_rgb(args.image_dir / "wrist" / f"frame_{frame:06d}.jpg"),
            "side": read_rgb(args.image_dir / "side" / f"frame_{frame:06d}.jpg"),
        }
        model_input = legacy.hardware_observation_for_model(
            observation,
            float(start["j2_command_offset_deg"]),
            float(start.get("j3_command_offset_deg", 0.0)),
            float(start.get("j5_command_offset_deg", 0.0)),
        )
        batch = legacy.live_observation_to_batch(model_input, device, adapter)
        with torch.inference_mode():
            policy.predict_action_chunk(batch)
        repeated_report = policy.last_prediction_report
        if repeated_report is None:
            raise RuntimeError(f"no prediction report for request {request_id}")

        live_entry = live_results[request_id]
        live_report = live_entry["report"]
        live_full = np.asarray(live_report["full_model_chunk_unnormalized"], dtype=np.float64)
        repeat_full = np.asarray(
            repeated_report["full_model_chunk_unnormalized"], dtype=np.float64
        )
        if live_full.shape != repeat_full.shape:
            raise RuntimeError(
                f"request {request_id} model chunk shape changed: "
                f"live={live_full.shape}, repeated={repeat_full.shape}"
            )
        full_joint_delta_deg = np.rad2deg(repeat_full[:, :6] - live_full[:, :6])
        live_selected = physical_chunk(live_report)
        repeated_selected = physical_chunk(repeated_report)
        selected_delta_deg = np.rad2deg(
            repeated_selected[:, :6] - live_selected[:, :6]
        )
        live_z = fk_z_mm(solver, live_selected[:, :6])
        repeated_z = fk_z_mm(solver, repeated_selected[:, :6])
        live_gripper = np.asarray(
            live_report["full_control_gripper_commands_pct"], dtype=np.float64
        )
        repeated_gripper = np.asarray(
            repeated_report["full_control_gripper_commands_pct"], dtype=np.float64
        )
        # queue_status is emitted after controller.advance(), so its active
        # offset names the *next* action.  Subtract one to recover the action
        # offset that was actually consumed on that frame.
        next_offsets = plan_offsets.get(request_id, [])
        consumed = [offset - 1 for offset in next_offsets if offset > 0]
        comparisons.append(
            {
                "ordinal": ordinal,
                "request_id": request_id,
                "request_frame": frame,
                "request_tick": request["request_tick"],
                "result_frame": live_entry["frame"],
                "install_tick": int(live_entry["result"]["install_tick"]),
                "inference_delay_action_ticks": int(
                    live_entry["result"]["inference_delay_action_ticks"]
                ),
                "executed_offset_min": min(consumed) if consumed else None,
                "executed_offset_max": max(consumed) if consumed else None,
                "executed_offset_count": len(set(consumed)),
                "unexecuted_tail_actions": (
                    30 - 1 - max(consumed) if consumed else None
                ),
                "live_peak_z_mm": float(live_z.max()),
                "live_peak_z_offset": int(live_z.argmax()),
                "repeated_peak_z_mm": float(repeated_z.max()),
                "repeated_peak_z_offset": int(repeated_z.argmax()),
                "peak_z_difference_mm": float(repeated_z.max() - live_z.max()),
                "full_joint_rms_difference_deg": float(
                    np.sqrt(np.mean(np.square(full_joint_delta_deg)))
                ),
                "full_joint_max_abs_difference_deg": float(
                    np.max(np.abs(full_joint_delta_deg))
                ),
                "selected_joint_rms_difference_deg": float(
                    np.sqrt(np.mean(np.square(selected_delta_deg)))
                ),
                "selected_joint_max_abs_difference_deg": float(
                    np.max(np.abs(selected_delta_deg))
                ),
                "gripper_command_mismatch_steps": int(
                    np.sum(live_gripper != repeated_gripper)
                ),
                "live_open_steps": int(np.sum(live_gripper >= 50.0)),
                "repeated_open_steps": int(np.sum(repeated_gripper >= 50.0)),
            }
        )
        print(
            f"[{ordinal:02d}/{len(requests):02d}] request={request_id:02d} "
            f"frame={frame:04d} selected_rms={comparisons[-1]['selected_joint_rms_difference_deg']:.3f}deg "
            f"peak_z={comparisons[-1]['live_peak_z_mm']:.1f}/{comparisons[-1]['repeated_peak_z_mm']:.1f}mm "
            f"gripper_mismatch={comparisons[-1]['gripper_command_mismatch_steps']}"
        )

    actual_q_rad = np.deg2rad(
        np.asarray([row["raw_state"][:6] for row in frame_rows], dtype=np.float64)
        / 1000.0
    )
    actual_z_mm = fk_z_mm(solver, actual_q_rad)
    transition_frames: dict[str, list[int]] = {}
    for row in frame_rows:
        transition = (row.get("gripper_state_machine") or {}).get("transition")
        if transition:
            transition_frames.setdefault(str(transition), []).append(int(row["frame"]))

    rms_values = np.asarray(
        [item["selected_joint_rms_difference_deg"] for item in comparisons]
    )
    max_values = np.asarray(
        [item["selected_joint_max_abs_difference_deg"] for item in comparisons]
    )
    peak_differences = np.asarray(
        [item["peak_z_difference_mm"] for item in comparisons]
    )
    summary = {
        "schema": "assembly_recorded_live_reinference_v1",
        "strictly_hardware_free": True,
        "run_log": str(args.run_log),
        "image_dir": str(args.image_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "request_count": len(comparisons),
        "selected_joint_rms_difference_deg_mean": float(rms_values.mean()),
        "selected_joint_rms_difference_deg_p95": float(np.percentile(rms_values, 95)),
        "selected_joint_rms_difference_deg_max": float(rms_values.max()),
        "selected_joint_max_abs_difference_deg_max": float(max_values.max()),
        "peak_z_difference_mm_mean": float(peak_differences.mean()),
        "peak_z_difference_mm_max_abs": float(np.max(np.abs(peak_differences))),
        "gripper_command_mismatch_steps_total": int(
            sum(item["gripper_command_mismatch_steps"] for item in comparisons)
        ),
        "transition_frames": transition_frames,
        "comparisons": comparisons,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "reinference_comparison.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    request_frames = np.asarray(
        [item["request_frame"] for item in comparisons], dtype=np.int64
    )
    live_peaks = np.asarray([item["live_peak_z_mm"] for item in comparisons])
    repeated_peaks = np.asarray(
        [item["repeated_peak_z_mm"] for item in comparisons]
    )
    executed_max = np.asarray(
        [
            np.nan if item["executed_offset_max"] is None else item["executed_offset_max"]
            for item in comparisons
        ],
        dtype=np.float64,
    )
    gripper_open = np.asarray(
        [item["live_open_steps"] for item in comparisons], dtype=np.float64
    )

    figure, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    axes[0].plot(np.arange(len(actual_z_mm)), actual_z_mm, color="black", linewidth=1.2, label="measured FK Z")
    axes[0].scatter(request_frames, live_peaks, s=22, color="#1f77b4", label="live logged full-30 peak")
    axes[0].scatter(request_frames, repeated_peaks, s=15, marker="x", color="#d62728", label="JPEG re-inference full-30 peak")
    axes[0].set_ylabel("gripper-base Z (mm)")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(request_frames, executed_max, color="#9467bd", marker="o", markersize=3)
    axes[1].axhline(29, color="black", linestyle="--", linewidth=1, label="complete 30-step tail")
    axes[1].set_ylabel("largest plan offset\nactually consumed")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    axes[2].step(request_frames, gripper_open, where="mid", color="#2ca02c")
    axes[2].set_ylabel("open steps in\nfull 30-step forecast")
    axes[2].set_xlabel("recorded control frame")
    axes[2].set_ylim(-1, 31)
    axes[2].grid(alpha=0.25)

    for transition, frames in transition_frames.items():
        color = "#ff7f0e" if transition == "open->close" else "#17becf"
        for frame in frames:
            for axis in axes:
                axis.axvline(frame, color=color, alpha=0.5, linewidth=1)
            axes[0].text(
                frame,
                axes[0].get_ylim()[1],
                transition,
                color=color,
                fontsize=8,
                rotation=90,
                va="top",
                ha="right",
            )
    figure.suptitle(
        "Recorded live run: complete 30-step predictions versus consumed RTC plans\n"
        "Re-inference uses saved JPEG observations; no robot I/O"
    )
    figure.tight_layout()
    plot_path = args.output_dir / "reinference_comparison.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    print(
        json.dumps(
            {
                "json": str(json_path),
                "plot": str(plot_path),
                **{key: value for key, value in summary.items() if key not in {"comparisons"}},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
