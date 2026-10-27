#!/usr/bin/env python
"""Guardedly return the Piper follower to the median robot-training start pose.

The default ``check`` mode is observation-only. Controller enable is double-gated
behind ``--mode hold|probe|execute --allow-motion``. ``hold`` enables the
controller while repeatedly commanding only the measured pose, ``probe`` then
executes exactly one small arm waypoint, and execute mode is feedback closed-loop:
it sends one small waypoint, waits until measured joints reach it, and only then
advances. Incoherent feedback, a large feedback jump, reverse motion, a motor
fault, or lack of progress aborts the reset before another waypoint is issued.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from egomimic.robot.piper.piper_kinematics import PiperKinematicsSolver, raw_joint_to_rad

from lerobot.robots.piper.config_piper import PiperConfig
from lerobot.robots.piper.piper import Piper

DEFAULT_CALIBRATION = Path("/mnt/wd10/kerun/calibration/my_piper_follower.json")
DEFAULT_LOG_DIR = Path("/mnt/wd10/kerun/deploy_logs")

# Coordinate-wise median of frame 0 from all 70 original robot demonstrations
# in Data_Collect/teddy_0912_{1..7}. Units are Piper feedback units:
# joints 0..5 are 0.001 degree and the gripper is 0.001 mm.
DEFAULT_TARGET_RAW = np.asarray(
    [1157.5, 0.0, 0.0, -14604.5, 6182.5, -153.5, 99700.0], dtype=np.float64
)

DEFAULT_DATA_WORKSPACE_MIN = np.asarray([0.057787, -0.348144, 0.121358], dtype=np.float64)
DEFAULT_DATA_WORKSPACE_MAX = np.asarray([0.319566, 0.014069, 0.420660], dtype=np.float64)
# The current real arm has been observed safely at x=0.042696 m and the
# validated reset interpolation reaches x=0.042592 m. Keep a small measured
# margin without pretending the narrower demonstration envelope is a physical
# collision boundary.
DEFAULT_PHYSICAL_WORKSPACE_MIN = np.asarray([0.040000, -0.348144, 0.121358], dtype=np.float64)
DEFAULT_PHYSICAL_WORKSPACE_MAX = np.asarray([0.319566, 0.014069, 0.420660], dtype=np.float64)


def parse_vector(value: str, length: int, name: str) -> np.ndarray:
    try:
        result = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain comma-separated numbers") from exc
    if result.shape != (length,):
        raise argparse.ArgumentTypeError(f"{name} must contain exactly {length} values")
    if not np.isfinite(result).all():
        raise argparse.ArgumentTypeError(f"{name} must contain only finite values")
    return result


def load_calibration(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        calibration = json.load(stream)
    required = [f"joint_{index}" for index in range(6)] + ["gripper"]
    missing = [key for key in required if key not in calibration]
    if missing:
        raise ValueError(f"calibration is missing {missing}: {path}")
    return calibration


def raw_to_normalized(raw: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    normalized = np.empty(7, dtype=np.float64)
    for index in range(6):
        item = calibration[f"joint_{index}"]
        low = float(item["range_min"])
        high = float(item["range_max"])
        if high <= low:
            raise ValueError(f"invalid joint_{index} calibration range [{low}, {high}]")
        normalized[index] = (raw[index] - low) / (high - low) * 200.0 - 100.0
    gripper = calibration["gripper"]
    low = float(gripper["range_min"])
    high = float(gripper["range_max"])
    if high <= low:
        raise ValueError(f"invalid gripper calibration range [{low}, {high}]")
    normalized[6] = (raw[6] - low) / (high - low) * 100.0
    return normalized


def normalized_to_raw(normalized: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    raw = np.empty(7, dtype=np.float64)
    for index in range(6):
        item = calibration[f"joint_{index}"]
        low = float(item["range_min"])
        high = float(item["range_max"])
        raw[index] = low + (normalized[index] + 100.0) / 200.0 * (high - low)
    gripper = calibration["gripper"]
    low = float(gripper["range_min"])
    high = float(gripper["range_max"])
    raw[6] = low + normalized[6] / 100.0 * (high - low)
    return raw


def raw_status(robot: Piper) -> np.ndarray:
    status = robot.sdk.get_status()
    return np.asarray([status[f"joint_{index}.pos"] for index in range(7)], dtype=np.float64)


def stable_status(
    robot: Piper,
    calibration: dict[str, Any],
    samples: int,
    sample_interval_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median raw/normalized feedback and the normalized sample span.

    A single Piper SDK read can occasionally contain a discontinuous snapshot.
    Taking a short burst lets the caller reject a changing/incoherent burst
    instead of treating one sample as a real arm pose.
    """
    raw_samples = []
    for sample_index in range(samples):
        raw = raw_status(robot)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            raise RuntimeError(f"invalid Piper feedback: {raw.tolist()}")
        raw_samples.append(raw)
        if sample_index + 1 < samples and sample_interval_s > 0:
            time.sleep(sample_interval_s)
    raw_stack = np.stack(raw_samples)
    normalized_stack = np.stack(
        [raw_to_normalized(sample, calibration) for sample in raw_stack]
    )
    raw_median = np.median(raw_stack, axis=0)
    normalized_median = raw_to_normalized(raw_median, calibration)
    normalized_span = np.ptp(normalized_stack, axis=0)
    return raw_median, normalized_median, normalized_span


def validate_normalized_state(state: np.ndarray, context: str) -> None:
    if state.shape != (7,) or not np.isfinite(state).all():
        raise RuntimeError(f"{context}: invalid normalized state {state.tolist()}")
    if np.any(state[:6] < -110.0) or np.any(state[:6] > 110.0):
        raise RuntimeError(f"{context}: arm feedback is outside calibrated safety margin: {state[:6].tolist()}")
    if not -5.0 <= state[6] <= 105.0:
        raise RuntimeError(f"{context}: gripper feedback is outside safety margin: {state[6]}")


def motor_faults(robot: Piper) -> list[dict[str, Any]]:
    try:
        return robot.sdk.get_motor_faults()
    except Exception as exc:  # A diagnostic failure must not hide the original abort reason.
        return [{"diagnostic_error": f"{type(exc).__name__}: {exc}"}]


def active_motor_faults(faults: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fault_keys = (
        "voltage_too_low",
        "motor_overheating",
        "driver_overcurrent",
        "driver_overheating",
        "collision_protection_triggered",
        "driver_error",
        "stall_protection_triggered",
    )
    active = []
    for motor_index, fault in enumerate(faults):
        triggered = [key for key in fault_keys if fault.get(key) is True]
        if triggered:
            active.append({"motor_index": motor_index, "triggered": triggered, "status": fault})
    return active


def normalized_action(values: np.ndarray) -> dict[str, float]:
    values = commandable_normalized(values)
    return {f"joint_{index}.pos": float(values[index]) for index in range(7)}


def commandable_normalized(values: np.ndarray) -> np.ndarray:
    """Clamp feedback-derived values into the controller command domain.

    Real feedback can sit a little beyond a hand-measured calibration endpoint.
    Feeding that value back as a command can make the firmware reject the entire
    joint packet, which previously looked like a stalled reset.  This clamp is
    only for commands; the unmodified feedback remains in the audit log.
    """
    result = np.asarray(values, dtype=np.float64).copy()
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError(f"invalid normalized command: {result.tolist()}")
    result[:6] = np.clip(result[:6], -100.0, 100.0)
    result[6] = np.clip(result[6], 0.0, 100.0)
    return result


def planned_path(
    start_normalized: np.ndarray, target_normalized: np.ndarray, max_step: float
) -> np.ndarray:
    points = [start_normalized.copy()]
    current = start_normalized.copy()
    for _ in range(10000):
        delta = np.clip(target_normalized - current, -max_step, max_step)
        if np.max(np.abs(delta)) < 1e-12:
            return np.stack(points)
        current = current + delta
        points.append(current.copy())
    raise RuntimeError("planned path exceeded 10000 interpolation steps")


def path_xyz(path_normalized: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    raw = np.stack([normalized_to_raw(point, calibration) for point in path_normalized])
    solver = PiperKinematicsSolver(device="cpu")
    return solver.fk(raw_joint_to_rad(raw[:, :6]))[:, :3]


def validate_local_segment(
    start: np.ndarray,
    end: np.ndarray,
    calibration: dict[str, Any],
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    interpolation_step: float,
) -> np.ndarray:
    segment = planned_path(start, end, interpolation_step)
    xyz = path_xyz(segment, calibration)
    outside = np.any((xyz < workspace_min) | (xyz > workspace_max), axis=1)
    if outside.any():
        first = int(np.flatnonzero(outside)[0])
        raise RuntimeError(
            "next guarded reset segment leaves workspace at "
            f"point {first}: xyz={xyz[first].tolist()}"
        )
    return xyz


def print_plan(
    current_raw: np.ndarray,
    target_raw: np.ndarray,
    calibration: dict[str, Any],
    max_step: float,
    physical_workspace_min: np.ndarray,
    physical_workspace_max: np.ndarray,
    data_workspace_min: np.ndarray,
    data_workspace_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_normalized = raw_to_normalized(current_raw, calibration)
    target_normalized = raw_to_normalized(target_raw, calibration)
    if np.any(target_normalized[:6] < -100.0) or np.any(target_normalized[:6] > 100.0):
        raise RuntimeError(f"target arm joints exceed calibrated range: {target_normalized[:6].tolist()}")
    if not 0.0 <= target_normalized[6] <= 100.0:
        raise RuntimeError(f"target gripper exceeds calibrated range: {target_normalized[6]}")

    path = planned_path(current_normalized, target_normalized, max_step)
    xyz = path_xyz(path, calibration)
    physical_outside = np.any(
        (xyz < physical_workspace_min) | (xyz > physical_workspace_max), axis=1
    )
    data_outside = np.any(
        (xyz < data_workspace_min) | (xyz > data_workspace_max), axis=1
    )

    print(f"current_raw={current_raw.tolist()}")
    print(f"target_raw={target_raw.tolist()}")
    print(f"current_normalized={current_normalized.tolist()}")
    print(f"target_normalized={target_normalized.tolist()}")
    print(f"planned_steps={len(path) - 1}")
    print(f"path_xyz_min_m={xyz.min(axis=0).tolist()}")
    print(f"path_xyz_max_m={xyz.max(axis=0).tolist()}")
    print(f"data_workspace_outside_points={int(data_outside.sum())}")
    print(f"physical_workspace_rejected_points={int(physical_outside.sum())}")
    # Retain the old field as a compatibility alias, now explicitly meaning
    # only the physical hard envelope.
    print(f"workspace_rejected_points={int(physical_outside.sum())}")
    if physical_outside.any():
        first = int(np.flatnonzero(physical_outside)[0])
        raise RuntimeError(
            "planned reset path leaves physical workspace at "
            f"point {first}: xyz={xyz[first].tolist()}"
        )
    return current_normalized, target_normalized, xyz


class JsonlLog:
    def __init__(self, log_dir: Path, mode: str):
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = log_dir / f"reset-training-start-{mode}_{timestamp}.jsonl"
        self.stream = self.path.open("x")

    def write(self, record: dict[str, Any]) -> None:
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def abort_reset(
    log: JsonlLog,
    robot: Piper,
    reason: str,
    **details: Any,
) -> None:
    faults = motor_faults(robot)
    log.write(
        {
            "event": "abort",
            "reason": reason,
            "motor_faults": faults,
            "command_sent": False,
            **details,
        }
    )
    raise RuntimeError(f"guarded reset aborted: {reason}")


def firmware_quick_stop(robot: Piper) -> dict[str, Any]:
    """Clear any latched trajectory command while retaining motor torque."""
    sdk = getattr(robot, "sdk", None)
    if sdk is None or not getattr(sdk, "motion_enabled", False):
        return {"quick_stop_sent": False, "reason": "motion was not enabled"}
    hold = sdk.get_status_normalized()
    hold_vector = [float(hold[f"joint_{index}.pos"]) for index in range(7)]
    sdk.set_joint_positions(hold_vector)
    time.sleep(0.05)
    # Piper EmergencyStop(0x01) is a torque-holding quick stop. Never call
    # ResetPiper here: the SDK warns that reset cuts power and the arm may fall.
    sdk.piper.EmergencyStop(0x01)
    sdk.motion_enabled = False
    return {
        "quick_stop_sent": True,
        "emergency_stop_code": 1,
        "primed_hold_normalized": hold_vector,
    }


def guarded_feedback(
    robot: Piper,
    calibration: dict[str, Any],
    args: argparse.Namespace,
    log: JsonlLog,
    last_accepted: np.ndarray | None,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw, normalized, span = stable_status(
        robot,
        calibration,
        args.feedback_samples,
        args.feedback_sample_interval_s,
    )
    validate_normalized_state(normalized, context)
    if float(np.max(span[:6])) > args.feedback_span_tolerance:
        abort_reset(
            log,
            robot,
            "incoherent feedback burst",
            context=context,
            raw_state=raw.tolist(),
            normalized_state=normalized.tolist(),
            normalized_span=span.tolist(),
            limit=args.feedback_span_tolerance,
        )
    if last_accepted is not None:
        jump = np.abs(normalized[:6] - last_accepted[:6])
        if float(np.max(jump)) > args.max_feedback_jump:
            abort_reset(
                log,
                robot,
                "discontinuous arm feedback",
                context=context,
                previous_normalized=last_accepted.tolist(),
                normalized_state=normalized.tolist(),
                joint_abs_jump=jump.tolist(),
                limit=args.max_feedback_jump,
            )
    return raw, normalized


def execute_stage(
    *,
    robot: Piper,
    calibration: dict[str, Any],
    args: argparse.Namespace,
    log: JsonlLog,
    stage_name: str,
    active_indices: tuple[int, ...],
    target: np.ndarray,
    initial: np.ndarray,
    global_started: float,
) -> np.ndarray:
    """Move selected axes via reached-before-advance feedback waypoints."""
    current = initial.copy()
    last_accepted = current.copy()
    waypoint_index = 0
    control_period = 1.0 / args.fps

    log.write(
        {
            "event": "stage_start",
            "stage": stage_name,
            "active_indices": list(active_indices),
            "initial_normalized": current.tolist(),
            "target_normalized": target.tolist(),
            "command_sent": False,
        }
    )

    while (
        float(np.max(np.abs(target[list(active_indices)] - current[list(active_indices)])))
        > args.waypoint_tolerance
    ):
        if time.monotonic() - global_started >= args.timeout_s:
            abort_reset(
                log,
                robot,
                "global timeout",
                stage=stage_name,
                waypoint=waypoint_index,
                normalized_state=current.tolist(),
                target_normalized=target.tolist(),
                timeout_s=args.timeout_s,
            )

        faults = motor_faults(robot)
        active_faults = active_motor_faults(faults)
        if active_faults:
            abort_reset(
                log,
                robot,
                "motor fault before waypoint",
                stage=stage_name,
                waypoint=waypoint_index,
                active_faults=active_faults,
                normalized_state=current.tolist(),
            )

        waypoint = commandable_normalized(current)
        indices = list(active_indices)
        waypoint[indices] += np.clip(
            target[indices] - current[indices],
            -args.max_relative_target,
            args.max_relative_target,
        )
        if any(index < 6 for index in active_indices):
            segment_xyz = validate_local_segment(
                current,
                waypoint,
                calibration,
                args.physical_workspace_min,
                args.physical_workspace_max,
                max(args.max_relative_target / 5.0, 0.01),
            )
        else:
            segment_xyz = np.empty((0, 3), dtype=np.float64)

        waypoint_start = current.copy()
        desired_delta = waypoint - waypoint_start
        sent = robot.send_action(normalized_action(waypoint))
        sent_vector = waypoint.copy()
        for index in range(7):
            key = f"joint_{index}.pos"
            if key in sent:
                sent_vector[index] = float(sent[key])

        log.write(
            {
                "event": "waypoint_command",
                "stage": stage_name,
                "waypoint": waypoint_index,
                "elapsed_s": time.monotonic() - global_started,
                "start_normalized": waypoint_start.tolist(),
                "requested_normalized": waypoint.tolist(),
                "sent_normalized": sent_vector.tolist(),
                "segment_xyz_min_m": None if len(segment_xyz) == 0 else segment_xyz.min(axis=0).tolist(),
                "segment_xyz_max_m": None if len(segment_xyz) == 0 else segment_xyz.max(axis=0).tolist(),
                "command_sent": True,
            }
        )

        waypoint_started = time.monotonic()
        best_error = float("inf")
        stalled_frames = 0
        feedback_frame = 0
        while True:
            loop_started = time.monotonic()
            raw, observed = guarded_feedback(
                robot,
                calibration,
                args,
                log,
                last_accepted,
                f"{stage_name} waypoint {waypoint_index}",
            )
            last_accepted = observed.copy()
            active_error = np.abs(sent_vector[indices] - observed[indices])
            active_target_error = np.abs(target[indices] - observed[indices])
            target_error_at_waypoint = np.abs(target[indices] - sent_vector[indices])
            # A motor controller may cross a small intermediate waypoint and
            # settle closer to the final stage target in one feedback period.
            # That is forward progress, not a stall. Count an axis as reached
            # either when it is close to the waypoint or when it is at least
            # as close to the final target as the waypoint was (within the
            # same tolerance). This remains directional because moving the
            # wrong way increases distance to the final target; the explicit
            # reverse-motion guard below remains active as a second check.
            reached_waypoint = active_error <= args.waypoint_tolerance
            passed_toward_target = (
                active_target_error
                <= target_error_at_waypoint + args.waypoint_tolerance
            )
            active_reached = reached_waypoint | passed_toward_target
            active_residual_error = np.where(active_reached, 0.0, active_error)
            max_error = float(np.max(active_residual_error))

            actual_delta = observed - waypoint_start
            reverse_amounts = []
            for index in active_indices:
                if abs(desired_delta[index]) < 1e-9:
                    reverse_amounts.append(0.0)
                else:
                    reverse_amounts.append(
                        float(-np.sign(desired_delta[index]) * actual_delta[index])
                    )
            max_reverse = max(reverse_amounts, default=0.0)
            if max_reverse > args.reverse_tolerance:
                abort_reset(
                    log,
                    robot,
                    "motion opposite to commanded waypoint",
                    stage=stage_name,
                    waypoint=waypoint_index,
                    raw_state=raw.tolist(),
                    waypoint_start=waypoint_start.tolist(),
                    sent_normalized=sent_vector.tolist(),
                    observed_normalized=observed.tolist(),
                    reverse_amounts=reverse_amounts,
                    limit=args.reverse_tolerance,
                )

            log.write(
                {
                    "event": "waypoint_feedback",
                    "stage": stage_name,
                    "waypoint": waypoint_index,
                    "feedback_frame": feedback_frame,
                    "elapsed_s": time.monotonic() - global_started,
                    "raw_state": raw.tolist(),
                    "normalized_state": observed.tolist(),
                    "sent_normalized": sent_vector.tolist(),
                    "active_abs_error": active_error.tolist(),
                    "active_target_abs_error": active_target_error.tolist(),
                    "target_abs_error_at_waypoint": target_error_at_waypoint.tolist(),
                    "active_reached": active_reached.tolist(),
                    "active_residual_error": active_residual_error.tolist(),
                    "max_active_error": max_error,
                    "command_sent": False,
                }
            )

            if bool(np.all(active_reached)):
                current = observed
                log.write(
                    {
                        "event": "waypoint_reached",
                        "stage": stage_name,
                        "waypoint": waypoint_index,
                        "feedback_frames": feedback_frame + 1,
                        "normalized_state": observed.tolist(),
                        "max_active_residual_error": max_error,
                        "active_reached": active_reached.tolist(),
                        "command_sent": False,
                    }
                )
                break

            if best_error - max_error >= args.min_progress:
                best_error = max_error
                stalled_frames = 0
            else:
                stalled_frames += 1
            if stalled_frames >= args.stall_frames:
                abort_reset(
                    log,
                    robot,
                    "waypoint made no measurable progress",
                    stage=stage_name,
                    waypoint=waypoint_index,
                    raw_state=raw.tolist(),
                    normalized_state=observed.tolist(),
                    sent_normalized=sent_vector.tolist(),
                    max_active_error=max_error,
                    best_error=best_error,
                    stalled_frames=stalled_frames,
                )
            if time.monotonic() - waypoint_started >= args.waypoint_timeout_s:
                abort_reset(
                    log,
                    robot,
                    "waypoint timeout",
                    stage=stage_name,
                    waypoint=waypoint_index,
                    raw_state=raw.tolist(),
                    normalized_state=observed.tolist(),
                    sent_normalized=sent_vector.tolist(),
                    max_active_error=max_error,
                    waypoint_timeout_s=args.waypoint_timeout_s,
                )

            # Piper position control is normally latched, but refreshing the same
            # already-validated waypoint makes the intended hold explicit.
            robot.send_action(normalized_action(sent_vector))
            feedback_frame += 1
            delay = control_period - (time.monotonic() - loop_started)
            if delay > 0:
                time.sleep(delay)

        waypoint_index += 1
        if getattr(args, "mode", "execute") == "probe":
            break

    log.write(
        {
            "event": "stage_complete",
            "stage": stage_name,
            "waypoints": waypoint_index,
            "final_normalized": current.tolist(),
            "target_reached": bool(
                np.max(np.abs(target[list(active_indices)] - current[list(active_indices)]))
                <= args.waypoint_tolerance
            ),
            "command_sent": False,
        }
    )
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("check", "hold", "probe", "execute"), default="check"
    )
    parser.add_argument("--can-port", default="can_follower")
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--target-raw",
        type=lambda value: parse_vector(value, 7, "target-raw"),
        default=DEFAULT_TARGET_RAW.copy(),
        help="Seven Piper raw values: six joints in 0.001deg and gripper in 0.001mm",
    )
    parser.add_argument(
        "--physical-workspace-min",
        "--workspace-min",
        dest="physical_workspace_min",
        type=lambda value: parse_vector(value, 3, "workspace-min"),
        default=DEFAULT_PHYSICAL_WORKSPACE_MIN.copy(),
        help="Hard physical safety envelope; --workspace-min is a legacy alias.",
    )
    parser.add_argument(
        "--physical-workspace-max",
        "--workspace-max",
        dest="physical_workspace_max",
        type=lambda value: parse_vector(value, 3, "workspace-max"),
        default=DEFAULT_PHYSICAL_WORKSPACE_MAX.copy(),
        help="Hard physical safety envelope; --workspace-max is a legacy alias.",
    )
    parser.add_argument(
        "--data-workspace-min",
        type=lambda value: parse_vector(value, 3, "data-workspace-min"),
        default=DEFAULT_DATA_WORKSPACE_MIN.copy(),
        help="Training-data envelope lower bound; diagnostic only.",
    )
    parser.add_argument(
        "--data-workspace-max",
        type=lambda value: parse_vector(value, 3, "data-workspace-max"),
        default=DEFAULT_DATA_WORKSPACE_MAX.copy(),
        help="Training-data envelope upper bound; diagnostic only.",
    )
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--max-relative-target", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--waypoint-timeout-s", type=float, default=4.0)
    parser.add_argument("--tolerance", type=float, default=0.75)
    parser.add_argument("--waypoint-tolerance", type=float, default=0.20)
    parser.add_argument("--settle-frames", type=int, default=10)
    parser.add_argument("--stall-frames", type=int, default=60)
    parser.add_argument("--min-progress", type=float, default=0.005)
    parser.add_argument("--reverse-tolerance", type=float, default=0.75)
    parser.add_argument("--max-feedback-jump", type=float, default=4.0)
    parser.add_argument("--initial-samples", type=int, default=9)
    parser.add_argument("--feedback-samples", type=int, default=3)
    parser.add_argument("--feedback-sample-interval-s", type=float, default=0.005)
    parser.add_argument("--feedback-span-tolerance", type=float, default=0.50)
    parser.add_argument("--hold-check-s", type=float, default=1.0)
    parser.add_argument("--hold-drift-tolerance", type=float, default=1.0)
    parser.add_argument("--allow-motion", action="store_true")
    args = parser.parse_args()

    if args.mode == "check" and args.allow_motion:
        parser.error("--allow-motion is only valid with --mode hold, probe, or execute")
    if args.mode in ("hold", "probe", "execute") and not args.allow_motion:
        parser.error("hold, probe, and execute modes require --allow-motion")
    if not 0 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [0, 100]")
    if (
        args.max_relative_target <= 0
        or args.fps <= 0
        or args.timeout_s <= 0
        or args.waypoint_timeout_s <= 0
    ):
        parser.error("max-relative-target, fps, timeout-s, and waypoint-timeout-s must be positive")
    if args.waypoint_tolerance >= args.max_relative_target:
        parser.error("waypoint-tolerance must be smaller than max-relative-target")
    if (
        args.tolerance <= 0
        or args.waypoint_tolerance <= 0
        or args.min_progress <= 0
        or args.reverse_tolerance <= 0
        or args.max_feedback_jump <= 0
        or args.feedback_span_tolerance <= 0
        or args.hold_check_s <= 0
        or args.hold_drift_tolerance <= 0
    ):
        parser.error("all tolerance/progress safety values must be positive")
    if args.settle_frames <= 0 or args.stall_frames <= 0:
        parser.error("settle-frames and stall-frames must be positive")
    if args.initial_samples < 3 or args.feedback_samples < 3:
        parser.error("initial-samples and feedback-samples must be at least 3")
    if args.feedback_sample_interval_s < 0:
        parser.error("feedback-sample-interval-s must be non-negative")
    if np.any(args.physical_workspace_max <= args.physical_workspace_min):
        parser.error(
            "physical-workspace-max must be greater than physical-workspace-min "
            "on every axis"
        )
    if np.any(args.data_workspace_max <= args.data_workspace_min):
        parser.error(
            "data-workspace-max must be greater than data-workspace-min on every axis"
        )
    return args


def main() -> None:
    args = parse_args()
    calibration = load_calibration(args.calibration_json)
    execute = args.mode in ("hold", "probe", "execute")
    robot = Piper(
        PiperConfig(
            id=args.calibration_json.stem,
            calibration_dir=args.calibration_json.parent,
            port=args.can_port,
            speed_percent=args.speed_percent,
            max_relative_target=args.max_relative_target,
            # Always connect observation-only.  All plan, workspace, feedback,
            # and motor-fault checks must finish before motion is enabled.
            # enable_motion_hold_current() below then latches the measured pose
            # as the first command, preventing a stale firmware target from
            # moving the arm during connection.
            enable_motion_on_connect=False,
            home_on_disconnect=False,
            disable_torque_on_disconnect=False,
        )
    )
    log = JsonlLog(args.log_dir, args.mode)
    connected = False
    reset_complete = False
    try:
        robot.connect(calibrate=False)
        connected = True
        initial_samples = args.initial_samples if execute else max(args.initial_samples, 5)
        current_raw, sampled_normalized, initial_span = stable_status(
            robot,
            calibration,
            initial_samples,
            args.feedback_sample_interval_s,
        )
        validate_normalized_state(sampled_normalized, "initial feedback")
        if float(np.max(initial_span[:6])) > args.feedback_span_tolerance:
            abort_reset(
                log,
                robot,
                "initial feedback is not stable",
                raw_state=current_raw.tolist(),
                normalized_state=sampled_normalized.tolist(),
                normalized_span=initial_span.tolist(),
                limit=args.feedback_span_tolerance,
            )
        current_normalized, target_normalized, xyz = print_plan(
            current_raw,
            args.target_raw,
            calibration,
            args.max_relative_target,
            args.physical_workspace_min,
            args.physical_workspace_max,
            args.data_workspace_min,
            args.data_workspace_max,
        )
        log.write(
            {
                "event": "plan",
                "mode": args.mode,
                "current_raw": current_raw.tolist(),
                "target_raw": args.target_raw.tolist(),
                "current_normalized": current_normalized.tolist(),
                "target_normalized": target_normalized.tolist(),
                "initial_feedback_span": initial_span.tolist(),
                "path_xyz_min_m": xyz.min(axis=0).tolist(),
                "path_xyz_max_m": xyz.max(axis=0).tolist(),
                "motor_faults": motor_faults(robot),
                "command_sent": False,
            }
        )
        if not execute:
            print("CHECK PASSED: observation-only; no motion command was sent.")
            return

        faults = motor_faults(robot)
        active_faults = active_motor_faults(faults)
        if active_faults:
            abort_reset(
                log,
                robot,
                "motor fault before reset",
                active_faults=active_faults,
                normalized_state=current_normalized.tolist(),
            )

        held_position = robot.enable_motion_hold_current()
        log.write(
            {
                "event": "motion_enable_hold_current",
                "held_position": held_position,
                "command_sent": True,
            }
        )

        # Re-anchor the trajectory after enabling.  The hold-current command
        # should leave the pose unchanged; a different stable pose means the
        # original plan is stale and must not be executed.
        enabled_raw, enabled_normalized = guarded_feedback(
            robot,
            calibration,
            args,
            log,
            current_normalized,
            "post-enable hold",
        )
        current_normalized = enabled_normalized
        log.write(
            {
                "event": "post_enable_feedback",
                "raw_state": enabled_raw.tolist(),
                "normalized_state": enabled_normalized.tolist(),
                "command_sent": False,
            }
        )

        if args.mode == "hold":
            hold_reference = enabled_normalized.copy()
            hold_last = enabled_normalized.copy()
            hold_started = time.monotonic()
            hold_frames = 0
            while time.monotonic() - hold_started < args.hold_check_s:
                hold_raw, hold_observed = guarded_feedback(
                    robot,
                    calibration,
                    args,
                    log,
                    hold_last,
                    "hold-only validation",
                )
                hold_last = hold_observed.copy()
                drift = np.abs(hold_observed[:6] - hold_reference[:6])
                max_drift = float(np.max(drift))
                log.write(
                    {
                        "event": "hold_feedback",
                        "frame": hold_frames,
                        "raw_state": hold_raw.tolist(),
                        "normalized_state": hold_observed.tolist(),
                        "joint_abs_drift": drift.tolist(),
                        "max_joint_abs_drift": max_drift,
                        "command_sent": False,
                    }
                )
                if max_drift > args.hold_drift_tolerance:
                    abort_reset(
                        log,
                        robot,
                        "enabled hold drifted from measured pose",
                        normalized_state=hold_observed.tolist(),
                        hold_reference=hold_reference.tolist(),
                        joint_abs_drift=drift.tolist(),
                        limit=args.hold_drift_tolerance,
                    )
                robot.send_action(normalized_action(hold_reference))
                hold_frames += 1
                time.sleep(1.0 / args.fps)
            log.write(
                {
                    "event": "hold_complete",
                    "frames": hold_frames,
                    "hold_reference": hold_reference.tolist(),
                    "final_normalized": hold_last.tolist(),
                    "command_sent": False,
                }
            )
            print(
                "HOLD CHECK COMPLETE: controller enable preserved the measured pose; "
                "no displacement target was commanded."
            )
            return

        print(
            "EXECUTING feedback-guarded reset. Keep emergency stop accessible; "
            "an aborted exit requests a firmware quick-stop; successful completion "
            "leaves the verified target latched."
        )
        started = time.monotonic()

        # Phase 1 moves only the six arm axes while holding the gripper at its
        # measured opening. Phase 2 adjusts the gripper after the arm is settled.
        arm_target = target_normalized.copy()
        arm_target[6] = current_normalized[6]
        current_normalized = execute_stage(
            robot=robot,
            calibration=calibration,
            args=args,
            log=log,
            stage_name="arm",
            active_indices=(0, 1, 2, 3, 4, 5),
            target=arm_target,
            initial=current_normalized,
            global_started=started,
        )
        if args.mode == "probe":
            log.write(
                {
                    "event": "probe_complete",
                    "elapsed_s": time.monotonic() - started,
                    "final_normalized": current_normalized.tolist(),
                    "target_normalized": target_normalized.tolist(),
                    "command_sent": False,
                }
            )
            print("PROBE COMPLETE: exactly one guarded arm waypoint was reached.")
            print(f"final_normalized={current_normalized.tolist()}")
            print("Full reset was not started; torque remains enabled.")
            return
        current_normalized = execute_stage(
            robot=robot,
            calibration=calibration,
            args=args,
            log=log,
            stage_name="gripper",
            active_indices=(6,),
            target=target_normalized,
            initial=current_normalized,
            global_started=started,
        )

        settled = 0
        final_raw = None
        final_normalized = current_normalized
        last_accepted = current_normalized.copy()
        while settled < args.settle_frames:
            if time.monotonic() - started >= args.timeout_s:
                abort_reset(
                    log,
                    robot,
                    "global timeout during final settle",
                    normalized_state=final_normalized.tolist(),
                    target_normalized=target_normalized.tolist(),
                    settled_frames=settled,
                )
            final_raw, final_normalized = guarded_feedback(
                robot,
                calibration,
                args,
                log,
                last_accepted,
                "final settle",
            )
            last_accepted = final_normalized.copy()
            max_error = float(np.max(np.abs(target_normalized - final_normalized)))
            settled = settled + 1 if max_error <= args.tolerance else 0
            log.write(
                {
                    "event": "settle",
                    "settled_frames": settled,
                    "raw_state": final_raw.tolist(),
                    "normalized_state": final_normalized.tolist(),
                    "target_normalized": target_normalized.tolist(),
                    "max_abs_error": max_error,
                    "command_sent": False,
                }
            )
            if settled < args.settle_frames:
                robot.send_action(normalized_action(target_normalized))
                time.sleep(1.0 / args.fps)

        log.write(
            {
                "event": "complete",
                "elapsed_s": time.monotonic() - started,
                "final_raw": final_raw.tolist(),
                "final_normalized": final_normalized.tolist(),
                "target_normalized": target_normalized.tolist(),
                "max_abs_error": float(np.max(np.abs(target_normalized - final_normalized))),
                "command_sent": False,
            }
        )
        # Leave a valid position target latched.  Normal completion deliberately
        # does not issue EmergencyStop: the next deployment process can connect
        # observation-only without first clearing a reset-induced stop state.
        robot.send_action(normalized_action(target_normalized))
        reset_complete = True
        print(f"RESET COMPLETE in {time.monotonic() - started:.2f}s")
        print(f"final_raw={final_raw.tolist()}")
        print(
            "Torque remains enabled by design; keep the arm supervised or follow the "
            "approved physical shutdown procedure."
        )
    finally:
        if connected:
            if execute and not reset_complete:
                try:
                    quick_stop = firmware_quick_stop(robot)
                    log.write(
                        {
                            "event": "exit_quick_stop",
                            **quick_stop,
                            "command_sent": bool(quick_stop["quick_stop_sent"]),
                        }
                    )
                except Exception as stop_exc:
                    log.write(
                        {
                            "event": "exit_quick_stop_failed",
                            "reason": repr(stop_exc),
                            "command_sent": False,
                        }
                    )
                    print(
                        "WARNING: firmware quick-stop failed; use the physical emergency stop."
                    )
            robot.disconnect()
        log.close()
        print(f"audit_log={log.path}")


if __name__ == "__main__":
    main()
