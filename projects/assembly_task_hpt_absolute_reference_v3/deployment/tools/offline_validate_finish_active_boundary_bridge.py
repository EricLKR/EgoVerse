#!/usr/bin/env python3
"""Hardware-free boundary-controller replay from a recorded execute JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_ROOT / "robot"))

import deploy_assembly_new1_hpt_joint_async_rtc as base  # noqa: E402
import deploy_assembly_new1_hpt_joint_async_rtc_boundary_bridge as bridge  # noqa: E402


class ReplayPolicy:
    def __init__(
        self,
        joint_limits_deg: list[list[float]],
        max_step: float,
        max_accel: float,
    ) -> None:
        limits = np.asarray(joint_limits_deg, dtype=np.float64)
        self._joint_low_rad = np.deg2rad(limits[:, 0])
        self._joint_high_rad = np.deg2rad(limits[:, 1])
        self._last_planned_velocity_driver = np.zeros(6, dtype=np.float64)
        self.config = SimpleNamespace(
            max_joint_command_step=float(max_step),
            max_joint_command_accel=float(max_accel),
        )

    def _joint_rad_to_driver(self, joints: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64)
        span = self._joint_high_rad - self._joint_low_rad
        return np.clip((joints - self._joint_low_rad) / span * 200.0 - 100.0, -100, 100)

    def _filter_driver_action(self, raw, current, velocity):
        action = np.asarray(raw, dtype=np.float64).reshape(7).copy()
        current = np.asarray(current, dtype=np.float64).reshape(6)
        velocity = np.asarray(velocity, dtype=np.float64).reshape(6)
        max_step = float(self.config.max_joint_command_step)
        max_accel = float(self.config.max_joint_command_accel)
        desired_velocity = np.clip(action[:6] - current, -max_step, max_step)
        next_velocity = np.clip(
            desired_velocity,
            velocity - max_accel,
            velocity + max_accel,
        )
        command = np.clip(current + next_velocity, -100.0, 100.0)
        action[:6] = command
        return action.astype(np.float32), command - current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--bridge-actions", type=int, default=8)
    parser.add_argument("--new-max-step", type=float, default=None)
    parser.add_argument("--new-max-accel", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    start = next(row for row in rows if row.get("event") == "start")
    frames = [
        row
        for row in rows
        if isinstance(row.get("frame"), int) and row.get("mode") == "execute"
    ]
    return start, frames


def accepted_plans(frames: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    plans = {}
    for row in frames:
        install = row.get("async_result") or {}
        report = row.get("new_chunk_report") or {}
        physical = report.get("selected_physical_actions")
        if not install.get("accepted") or physical is None:
            continue
        plans[int(row["frame"])] = {
            "plan_id": int(install["plan_id"]),
            "start_tick": int(install["request_tick"]),
            "actions": np.asarray(physical, dtype=np.float32),
            "prediction_report": report,
            "result_age_ms": float(install["result_age_ms"]),
        }
    if not plans:
        raise RuntimeError("log has no accepted plans with decoded actions")
    return plans


def driver_to_deg(policy: ReplayPolicy, driver: np.ndarray) -> np.ndarray:
    driver = np.asarray(driver, dtype=np.float64)
    low = np.rad2deg(policy._joint_low_rad)
    high = np.rad2deg(policy._joint_high_rad)
    return low + (driver + 100.0) / 200.0 * (high - low)


def command_from_row(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [row["model_action"][f"joint_{index}.pos"] for index in range(6)],
        dtype=np.float64,
    )


def replay(
    controller,
    frames_by_tick: dict[int, dict[str, Any]],
    plans: dict[int, dict[str, Any]],
    max_age_ms: float,
) -> list[dict[str, Any]]:
    first_tick = min(plans)
    previous_row = frames_by_tick[first_tick - 1]
    previous_previous = frames_by_tick[first_tick - 2]
    previous_command = command_from_row(previous_row)
    previous_previous_command = command_from_row(previous_previous)
    previous_state = np.asarray(previous_row["model_driver_state"][:6], dtype=np.float64)
    controller.action_tick = int(first_tick)
    controller._last_velocity = previous_command - previous_state
    if isinstance(controller, bridge.BoundaryBridgedDecodedJointRtcController):
        controller._last_sent_driver = previous_command.copy()
        controller._previous_sent_driver = previous_previous_command.copy()
        controller._last_command_velocity_driver = (
            previous_command - previous_previous_command
        )

    trace = []
    for tick in range(first_tick, max(frames_by_tick) + 1):
        row = frames_by_tick[tick]
        if tick in plans:
            plan = plans[tick]
            installed = controller.install(
                plan_id=plan["plan_id"],
                start_tick=plan["start_tick"],
                actions=plan["actions"],
                prediction_report=plan["prediction_report"],
                result_age_ms=plan["result_age_ms"],
                max_result_age_ms=max_age_ms,
            )
            if not installed["accepted"]:
                raise RuntimeError(f"offline install rejected at {tick}: {installed}")
        current = np.asarray(row["model_driver_state"][:6], dtype=np.float64)
        action, status = controller.peek(np.r_[current, 0.0])
        trace.append(
            {
                "tick": tick,
                "plan_id": int(status["active_plan_id"]),
                "command_driver": action[:6].astype(np.float64),
                "boundary": bool(
                    trace and int(status["active_plan_id"]) != trace[-1]["plan_id"]
                ),
                "boundary_event": status.get("last_boundary_activation"),
            }
        )
        controller.advance()
    return trace


def metrics(trace: list[dict[str, Any]], policy: ReplayPolicy) -> dict[str, Any]:
    driver = np.asarray([row["command_driver"] for row in trace])
    degrees = driver_to_deg(policy, driver)
    step = np.diff(degrees, axis=0)
    accel = np.diff(step, axis=0)
    step_rms = np.sqrt(np.mean(np.square(step), axis=1))
    accel_rms = np.sqrt(np.mean(np.square(accel), axis=1))
    boundary_mask = np.asarray([row["boundary"] for row in trace[1:]], dtype=bool)
    boundary_accel_mask = np.asarray(
        [row["boundary"] for row in trace[2:]], dtype=bool
    )
    events = [row["boundary_event"] for row in trace if row["boundary"]]
    return {
        "samples": len(trace),
        "boundary_count": int(boundary_mask.sum()),
        "boundary_step_rms_deg_mean": float(step_rms[boundary_mask].mean()),
        "boundary_step_rms_deg_max": float(step_rms[boundary_mask].max()),
        "nonboundary_step_rms_deg_mean": float(step_rms[~boundary_mask].mean()),
        "step_rms_deg_max": float(step_rms.max()),
        "accel_rms_deg_mean": float(accel_rms.mean()),
        "accel_rms_deg_max": float(accel_rms.max()),
        "boundary_accel_rms_deg_mean": float(
            accel_rms[boundary_accel_mask].mean()
        ),
        "boundary_accel_rms_deg_max": float(
            accel_rms[boundary_accel_mask].max()
        ),
        "max_abs_driver_acceleration": float(
            np.max(np.abs(np.diff(driver, n=2, axis=0)))
        ),
        "boundary_blend_actions": sorted(
            {
                int(event.get("rtc_overlap_blend_actions", 0))
                for event in events
                if event is not None
            }
        ),
    }


def main() -> None:
    args = parse_args()
    start, frames = read_rows(args.run_log)
    frames_by_tick = {int(row["frame"]): row for row in frames}
    plans = accepted_plans(frames)
    trajectory_filter = next(
        row["new_chunk_report"]["trajectory_filter"]
        for row in frames
        if (row.get("new_chunk_report") or {}).get("trajectory_filter")
    )
    limits = start["joint_space_adapter"]["model_joint_limits_deg"]
    old_policy = ReplayPolicy(
        limits,
        trajectory_filter["max_joint_command_step"],
        trajectory_filter["max_joint_command_accel"],
    )
    new_max_step = float(
        trajectory_filter["max_joint_command_step"]
        if args.new_max_step is None
        else args.new_max_step
    )
    new_max_accel = float(args.new_max_accel)
    old_tuned_policy = ReplayPolicy(limits, new_max_step, new_max_accel)
    new_policy = ReplayPolicy(
        limits,
        new_max_step,
        new_max_accel,
    )
    old = base.DecodedJointRtcController(
        old_policy,
        int(start["rtc_min_suffix_actions"]),
        int(start["rtc_blend_actions"]),
        finish_active_plan_before_install=True,
    )
    new = bridge.BoundaryBridgedDecodedJointRtcController(
        new_policy,
        int(start["rtc_min_suffix_actions"]),
        int(args.bridge_actions),
        finish_active_plan_before_install=True,
    )
    old_tuned = base.DecodedJointRtcController(
        old_tuned_policy,
        int(start["rtc_min_suffix_actions"]),
        int(start["rtc_blend_actions"]),
        finish_active_plan_before_install=True,
    )
    old_trace = replay(
        old, frames_by_tick, plans, float(start["rtc_max_result_age_ms"])
    )
    new_trace = replay(
        new, frames_by_tick, plans, float(start["rtc_max_result_age_ms"])
    )
    old_tuned_trace = replay(
        old_tuned,
        frames_by_tick,
        plans,
        float(start["rtc_max_result_age_ms"]),
    )
    old_metrics = metrics(old_trace, old_policy)
    old_tuned_metrics = metrics(old_tuned_trace, old_tuned_policy)
    new_metrics = metrics(new_trace, new_policy)
    result = {
        "schema": "assembly_finish_active_boundary_bridge_offline_validation_v1",
        "strictly_hardware_free": True,
        "source_log": str(args.run_log),
        "counterfactual_limitation": (
            "Recorded closed-loop observations and chunks are reused; this isolates "
            "command stitching and is not a counterfactual policy rollout."
        ),
        "old_finish_active": old_metrics,
        "old_finish_active_with_tuned_accel": old_tuned_metrics,
        "new_boundary_bridge": new_metrics,
        "proposed_command_filter": {
            "max_joint_command_step": new_max_step,
            "max_joint_command_accel": new_max_accel,
        },
        "improvement": {
            "boundary_step_rms_mean_reduction_fraction": float(
                1.0
                - new_metrics["boundary_step_rms_deg_mean"]
                / old_metrics["boundary_step_rms_deg_mean"]
            ),
            "boundary_step_rms_max_reduction_fraction": float(
                1.0
                - new_metrics["boundary_step_rms_deg_max"]
                / old_metrics["boundary_step_rms_deg_max"]
            ),
        },
        "gates": {
            "all_boundaries_have_requested_bridge": (
                new_metrics["boundary_blend_actions"] == [int(args.bridge_actions)]
            ),
            "driver_acceleration_respects_configured_limit": (
                new_metrics["max_abs_driver_acceleration"]
                <= new_max_accel + 1e-5
            ),
            "boundary_acceleration_mean_reduced_by_at_least_50_percent": (
                new_metrics["boundary_accel_rms_deg_mean"]
                <= 0.5 * old_metrics["boundary_accel_rms_deg_mean"]
            ),
            "bridge_boundary_acceleration_peak_below_tuned_old_controller": (
                new_metrics["boundary_accel_rms_deg_max"]
                < old_tuned_metrics["boundary_accel_rms_deg_max"]
            ),
            "bridge_global_acceleration_peak_below_tuned_old_controller": (
                new_metrics["accel_rms_deg_max"]
                < old_tuned_metrics["accel_rms_deg_max"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
