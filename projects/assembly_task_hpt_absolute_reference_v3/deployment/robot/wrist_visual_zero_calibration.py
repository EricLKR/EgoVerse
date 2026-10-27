#!/usr/bin/env python3
"""Bounded, model-free wrist-camera zero-offset calibration for PIPER.

The deployment arm is the original collection leader, whereas the policy was
trained on the follower.  Equal encoder angles on two physical arms need not
produce an identical camera pose.  This tool estimates a small hardware joint
offset by comparing the live D405 image with a training Home reference.

``check`` is strictly observation-only: it opens the CAN interface without
motion authority, captures the D405 image, and reports the visual residual.

``execute`` requires ``--allow-motion``.  It is intentionally restricted to a
small neighbourhood of the saved Home pose.  The program probes selected wrist
joints with central differences, builds a numerical image/J4-J6 Jacobian, and
uses damped least squares plus a line search.  It never edits deployment assets
or activates the resulting offset.  A successful run only writes a candidate
JSON for explicit review and later application.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    BUNDLE_ROOT / "assets/wrist_alignment_reference_train_ep0000_frame0.png"
)
DEFAULT_CALIBRATION = BUNDLE_ROOT / "assets/piper_original_leader_calibration.json"
DEFAULT_INITIAL_POSE = (
    BUNDLE_ROOT / "assets/initial_pose_follower_physical_on_original_leader.json"
)
DEFAULT_OUTPUT_ROOT = BUNDLE_ROOT / "validation/wrist_zero_calibration"
WRIST_SERIAL = "427622273943"
CONTROL_HZ = 30.0


@dataclass(frozen=True)
class SolverResult:
    step_deg: np.ndarray
    raw_step_deg: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition_number: float
    predicted_error_before: float
    predicted_error_after: float


def weighted_alignment_error(
    report: dict[str, Any], *, rotation_weight_px_per_deg: float
) -> np.ndarray:
    """Return the optimization residual in pixel-equivalent units."""
    values = np.asarray(
        [
            float(report["tx_px"]),
            float(report["ty_px"]),
            float(report["rotation_deg"]) * rotation_weight_px_per_deg,
        ],
        dtype=np.float64,
    )
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"invalid visual alignment residual: {values.tolist()}")
    return values


def damped_joint_update(
    error: Sequence[float],
    jacobian: np.ndarray,
    *,
    damping: float,
    max_step_deg: float,
    current_offset_deg: Sequence[float],
    max_total_offset_deg: float,
    minimum_rank: int = 2,
) -> SolverResult:
    """Solve a bounded damped least-squares image correction.

    ``jacobian`` maps active-joint degrees to pixel-equivalent visual error.
    The returned step minimizes ``||J dq + error||^2 + damping ||dq||^2``.
    Both the per-iteration infinity norm and the cumulative per-joint offset
    are bounded.
    """
    residual = np.asarray(error, dtype=np.float64)
    matrix = np.asarray(jacobian, dtype=np.float64)
    offset = np.asarray(current_offset_deg, dtype=np.float64)
    if residual.shape != (3,):
        raise ValueError(f"error must have shape (3,), got {residual.shape}")
    if matrix.ndim != 2 or matrix.shape[0] != 3:
        raise ValueError(f"jacobian must have shape (3,N), got {matrix.shape}")
    if offset.shape != (matrix.shape[1],):
        raise ValueError(
            f"current offset must have shape {(matrix.shape[1],)}, got {offset.shape}"
        )
    if not np.isfinite(residual).all() or not np.isfinite(matrix).all():
        raise ValueError("solver input contains non-finite values")
    if not np.isfinite(offset).all():
        raise ValueError("current offset contains non-finite values")
    for name, value in (
        ("damping", damping),
        ("max_step_deg", max_step_deg),
        ("max_total_offset_deg", max_total_offset_deg),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    if rank < minimum_rank:
        raise RuntimeError(
            "visual joint Jacobian is rank deficient: "
            f"rank={rank}, required={minimum_rank}, singular_values={singular_values.tolist()}"
        )
    smallest_observable = singular_values[rank - 1]
    condition_number = float(singular_values[0] / smallest_observable)

    lhs = matrix.T @ matrix + float(damping) * np.eye(matrix.shape[1])
    rhs = -(matrix.T @ residual)
    raw_step = np.linalg.solve(lhs, rhs)
    largest = float(np.max(np.abs(raw_step)))
    step = raw_step.copy()
    if largest > max_step_deg:
        step *= max_step_deg / largest

    proposed_offset = offset + step
    total_largest = float(np.max(np.abs(proposed_offset)))
    if total_largest > max_total_offset_deg:
        # Move only to the cumulative trust-region boundary; do not clip axes
        # independently because that can reverse the solver's direction.
        allowed = max_total_offset_deg - float(np.max(np.abs(offset)))
        if allowed <= 0:
            raise RuntimeError(
                "visual calibration reached the total joint-offset limit "
                f"({max_total_offset_deg} deg)"
            )
        step_largest = float(np.max(np.abs(step)))
        step *= allowed / step_largest

    before = float(np.linalg.norm(residual))
    after = float(np.linalg.norm(residual + matrix @ step))
    if after >= before:
        raise RuntimeError(
            "damped visual solver predicts no improvement: "
            f"before={before:.3f}, after={after:.3f}"
        )
    return SolverResult(
        step_deg=step,
        raw_step_deg=raw_step,
        singular_values=singular_values,
        rank=rank,
        condition_number=condition_number,
        predicted_error_before=before,
        predicted_error_after=after,
    )


def alignment_converged(
    report: dict[str, Any],
    *,
    translation_tolerance_px: float,
    rotation_tolerance_deg: float,
) -> bool:
    return bool(
        float(report["translation_norm_px"]) <= translation_tolerance_px
        and abs(float(report["rotation_deg"])) <= rotation_tolerance_deg
    )


class WristCamera:
    """Persistent D405 stream used while the robot controller holds position."""

    def __init__(self, *, warmup_frames: int) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(WRIST_SERIAL)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self._pipeline.start(config)
        try:
            for _ in range(warmup_frames):
                frame = self._pipeline.wait_for_frames(5000).get_color_frame()
                if not frame:
                    raise RuntimeError("D405 returned no RGB frame during warmup")
        except BaseException:
            self._pipeline.stop()
            raise

    def capture(
        self,
        *,
        sample_frames: int,
        hold_callback: Callable[[], None] | None = None,
    ) -> np.ndarray:
        frames: list[np.ndarray] = []
        for _ in range(sample_frames):
            frame = self._pipeline.wait_for_frames(5000).get_color_frame()
            if not frame:
                raise RuntimeError("D405 returned no RGB frame during sampling")
            frames.append(np.asanyarray(frame.get_data()).copy())
            if hold_callback is not None:
                hold_callback()
        return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

    def close(self) -> None:
        self._pipeline.stop()


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("x", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self._stream.write(json.dumps(payload, allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def parse_active_joints(value: str) -> tuple[int, ...]:
    try:
        joints = tuple(int(item.strip()) - 1 for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("active joints must be comma-separated integers") from exc
    if not joints or len(set(joints)) != len(joints) or any(joint not in (3, 4, 5) for joint in joints):
        raise argparse.ArgumentTypeError(
            "automatic wrist calibration is restricted to unique joints 4,5,6"
        )
    return joints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "execute"), default="check")
    parser.add_argument("--can-port", required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--initial-pose-json", type=Path, default=DEFAULT_INITIAL_POSE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--active-joints", type=parse_active_joints, default=parse_active_joints("4,5,6"))
    parser.add_argument("--probe-step-deg", type=float, default=0.5)
    parser.add_argument("--max-update-step-deg", type=float, default=1.0)
    parser.add_argument("--max-total-offset-deg", type=float, default=8.0)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--damping", type=float, default=0.25)
    parser.add_argument("--rotation-weight-px-per-deg", type=float, default=10.0)
    parser.add_argument("--translation-tolerance-px", type=float, default=3.0)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--minimum-actual-improvement-px", type=float, default=0.5)
    parser.add_argument("--start-pose-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--joint-limit-margin-deg", type=float, default=2.0)
    parser.add_argument("--move-velocity-deg-s", type=float, default=2.0)
    parser.add_argument("--move-acceleration-deg-s2", type=float, default=5.0)
    parser.add_argument("--move-tolerance-deg", type=float, default=0.2)
    parser.add_argument("--move-timeout-s", type=float, default=30.0)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--max-relative-target", type=float, default=0.25)
    parser.add_argument("--max-gripper-relative-target", type=float, default=100.0)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--sample-frames", type=int, default=9)
    parser.add_argument(
        "--return-to-start-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-motion", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.mode == "execute" and not args.allow_motion:
        parser.error("execute mode requires --allow-motion")
    if args.mode != "execute" and args.allow_motion:
        parser.error("--allow-motion is valid only with --mode execute")
    for path in (args.reference, args.calibration_json, args.initial_pose_json):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if not 1 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [1,100]")
    if args.max_iterations < 1 or args.warmup_frames < 1 or args.sample_frames < 1:
        parser.error("iteration and frame counts must be positive")
    positive = (
        "probe_step_deg",
        "max_update_step_deg",
        "max_total_offset_deg",
        "damping",
        "rotation_weight_px_per_deg",
        "translation_tolerance_px",
        "rotation_tolerance_deg",
        "minimum_actual_improvement_px",
        "start_pose_tolerance_deg",
        "joint_limit_margin_deg",
        "move_velocity_deg_s",
        "move_acceleration_deg_s2",
        "move_tolerance_deg",
        "move_timeout_s",
        "settle_s",
        "max_relative_target",
        "max_gripper_relative_target",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.probe_step_deg > 1.0:
        parser.error("--probe-step-deg is safety-limited to at most 1 degree")
    if args.max_update_step_deg > 2.0:
        parser.error("--max-update-step-deg is safety-limited to at most 2 degrees")
    if args.max_total_offset_deg > 10.0:
        parser.error("--max-total-offset-deg is safety-limited to at most 10 degrees")


def _alignment_module() -> Any:
    tools_dir = BUNDLE_ROOT / "tools"
    sys.path.insert(0, str(tools_dir))
    import check_wrist_visual_alignment

    return check_wrist_visual_alignment


def _home_module() -> Any:
    robot_dir = BUNDLE_ROOT / "robot"
    sys.path.insert(0, str(robot_dir))
    import home_assembly_new1

    return home_assembly_new1


def _validate_goal_limits(
    goal_deg: np.ndarray,
    joint_limits: Sequence[tuple[float, float]],
    margin_deg: float,
) -> None:
    for index, (value, (low, high)) in enumerate(zip(goal_deg, joint_limits, strict=True)):
        if not low + margin_deg <= value <= high - margin_deg:
            raise RuntimeError(
                f"calibration probe joint {index + 1}={value:.3f} deg violates "
                f"the {margin_deg:.3f} deg limit margin [{low:.3f},{high:.3f}]"
            )


def _move_to_joint_pose(
    sdk: Any,
    *,
    goal_deg: np.ndarray,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
    args: argparse.Namespace,
    home: Any,
    log: JsonlLogger,
    label: str,
) -> None:
    _validate_goal_limits(goal_deg, joint_limits, args.joint_limit_margin_deg)
    start_pose = home.raw_feedback_pose(sdk.get_status())
    start = np.asarray(start_pose.joints_deg, dtype=np.float64)
    duration = home.quintic_joint_duration(
        tuple(start),
        tuple(goal_deg),
        args.move_velocity_deg_s,
        args.move_acceleration_deg_s2,
    )
    if duration > args.move_timeout_s:
        raise RuntimeError(
            f"calibration move {label} requires {duration:.2f}s, exceeding "
            f"the {args.move_timeout_s:.2f}s timeout"
        )
    started = time.perf_counter()
    tick = 0
    while True:
        tick_started = time.perf_counter()
        if tick_started - started > args.move_timeout_s:
            raise RuntimeError(f"calibration move {label} timed out")
        home.verify_health(sdk, require_enabled=True, check_controller=False)
        elapsed = tick_started - started
        blend = home.quintic_blend(elapsed / duration)
        target_deg = start + (goal_deg - start) * blend
        target = home.PhysicalPose(tuple(float(value) for value in target_deg), start_pose.gripper_mm)
        command = home.bounded_send(
            sdk,
            home.physical_pose_to_normalized(target, joint_limits, gripper_stroke_mm),
            max_arm_relative_target=args.max_relative_target,
            max_gripper_relative_target=args.max_gripper_relative_target,
        )
        measured = home.raw_feedback_pose(sdk.get_status())
        error = float(
            np.max(np.abs(np.asarray(measured.joints_deg, dtype=np.float64) - goal_deg))
        )
        if tick % 30 == 0:
            log.write(
                {
                    "event": "calibration_move_progress",
                    "label": label,
                    "goal_joints_deg": goal_deg.tolist(),
                    "measured_joints_deg": list(measured.joints_deg),
                    "max_error_deg": error,
                    "command": command,
                    "command_sent": True,
                }
            )
        if elapsed >= duration and error <= args.move_tolerance_deg:
            break
        tick += 1
        time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - tick_started)))

    settle_started = time.perf_counter()
    goal_pose = home.PhysicalPose(tuple(float(value) for value in goal_deg), start_pose.gripper_mm)
    goal_normalized = home.physical_pose_to_normalized(
        goal_pose, joint_limits, gripper_stroke_mm
    )
    while time.perf_counter() - settle_started < args.settle_s:
        tick_started = time.perf_counter()
        home.verify_health(sdk, require_enabled=True, check_controller=False)
        home.bounded_send(
            sdk,
            goal_normalized,
            max_arm_relative_target=args.max_relative_target,
            max_gripper_relative_target=args.max_gripper_relative_target,
        )
        time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - tick_started)))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / f"{args.mode}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log = JsonlLogger(run_dir / "events.jsonl")
    alignment = _alignment_module()
    home = _home_module()
    reference_rgb = np.asarray(Image.open(args.reference).convert("RGB"))
    joint_limits, gripper_stroke_mm = home.load_calibration(args.calibration_json)
    saved_home = home.load_initial_pose(
        args.initial_pose_json, joint_limits, gripper_stroke_mm
    )
    sdk: Any | None = None
    camera: WristCamera | None = None
    motion_enabled = False
    start_deg: np.ndarray | None = None
    current_center: np.ndarray | None = None

    def hold_current_center() -> None:
        if sdk is None or current_center is None or not motion_enabled:
            return
        measured = home.raw_feedback_pose(sdk.get_status())
        pose = home.PhysicalPose(
            tuple(float(value) for value in current_center), measured.gripper_mm
        )
        home.bounded_send(
            sdk,
            home.physical_pose_to_normalized(pose, joint_limits, gripper_stroke_mm),
            max_arm_relative_target=args.max_relative_target,
            max_gripper_relative_target=args.max_gripper_relative_target,
        )

    def capture_report(label: str) -> tuple[dict[str, Any], np.ndarray]:
        if camera is None:
            raise RuntimeError("wrist camera is not open")
        image = camera.capture(
            sample_frames=args.sample_frames,
            hold_callback=hold_current_center if motion_enabled else None,
        )
        if image.shape != reference_rgb.shape:
            raise RuntimeError(
                f"image shape mismatch: reference={reference_rgb.shape}, current={image.shape}"
            )
        report = alignment.estimate_alignment(reference_rgb, image)
        # ``estimate_alignment`` is also used by a camera-only diagnostic and
        # therefore defaults these audit fields to false. This program has an
        # observation-only CAN connection in check mode and sends bounded hold
        # commands during execute captures, so overwrite them truthfully here.
        report["robot_or_can_opened"] = sdk is not None
        report["motion_command_sent"] = bool(motion_enabled)
        Image.fromarray(image).save(run_dir / f"{label}.png")
        log.write({"event": "visual_measurement", "label": label, **report})
        return report, weighted_alignment_error(
            report, rotation_weight_px_per_deg=args.rotation_weight_px_per_deg
        )

    try:
        sdk = home.PiperSDKInterface(
            port=args.can_port,
            gripper_stroke_mm=gripper_stroke_mm,
            speed_percent=args.speed_percent,
            joint_limits_deg=joint_limits,
            enable_motion=False,
        )
        measured = home.raw_feedback_pose(sdk.get_status())
        start_deg = np.asarray(measured.joints_deg, dtype=np.float64)
        saved_home_deg = np.asarray(saved_home.joints_deg, dtype=np.float64)
        start_error = np.abs(start_deg - saved_home_deg)
        motors = home.verify_health(sdk, require_enabled=False, check_controller=False)
        status = sdk.piper.GetArmStatus().arm_status
        camera = WristCamera(warmup_frames=args.warmup_frames)
        current_center = start_deg.copy()
        baseline_report, baseline_error = capture_report("baseline")
        preflight = {
            "event": "wrist_zero_calibration_preflight",
            "mode": args.mode,
            "can_port": args.can_port,
            "active_joints_1_based": [joint + 1 for joint in args.active_joints],
            "current_joints_deg": start_deg.tolist(),
            "saved_home_joints_deg": saved_home_deg.tolist(),
            "max_start_home_error_deg": float(np.max(start_error)),
            "motor_enabled": [bool(motor["driver_enabled"]) for motor in motors],
            "motor_voltage_v": [float(motor["voltage_v"]) for motor in motors],
            "controller_status": str(status),
            "baseline_alignment": baseline_report,
            "command_sent": False,
        }
        log.write(preflight)
        (run_dir / "preflight.json").write_text(
            json.dumps(preflight, indent=2), encoding="utf-8"
        )
        print(json.dumps(preflight, indent=2))
        if args.mode == "check":
            print(
                "CHECK PASSED: observation-only; camera and CAN feedback were read, "
                f"no enable or motion command was sent. output={run_dir}"
            )
            return

        if float(np.max(start_error)) > args.start_pose_tolerance_deg:
            raise RuntimeError(
                "automatic wrist calibration requires the arm at saved Home: "
                f"max_error={float(np.max(start_error)):.3f} deg exceeds "
                f"{args.start_pose_tolerance_deg:.3f} deg"
            )
        if status.err_code or status.arm_status or status.ctrl_mode not in (0, 1):
            raise RuntimeError(
                "automatic wrist calibration requires NORMAL STANDBY/CAN controller "
                f"state; got {status}"
            )
        enabled = [bool(motor["driver_enabled"]) for motor in motors]
        if all(enabled):
            home.adopt_already_enabled_motion_hold_current(sdk)
            transition = "adopt_already_enabled_without_quick_stop"
        elif not any(enabled):
            home.enable_disabled_motion_hold_current(sdk)
            transition = "enable_from_all_disabled_without_quick_stop"
        else:
            raise RuntimeError(
                "refusing automatic calibration from a partially enabled arm: "
                f"driver_enabled={enabled}"
            )
        motion_enabled = True
        # Remove normal tracking residual from the estimated zero offset. The
        # optimization origin is the exact saved *commanded* model Home, not a
        # noisy encoder sample that happened to be within the preflight gate.
        _move_to_joint_pose(
            sdk,
            goal_deg=saved_home_deg,
            joint_limits=joint_limits,
            gripper_stroke_mm=gripper_stroke_mm,
            args=args,
            home=home,
            log=log,
            label="exact_saved_home_origin",
        )
        current_center = saved_home_deg.copy()
        start_deg = saved_home_deg.copy()
        baseline_report, baseline_error = capture_report("execute_baseline")
        log.write(
            {
                "event": "motion_enable",
                "transition": transition,
                "held_joints_deg": current_center.tolist(),
                "command_sent": True,
            }
        )

        report = baseline_report
        error = baseline_error
        success = alignment_converged(
            report,
            translation_tolerance_px=args.translation_tolerance_px,
            rotation_tolerance_deg=args.rotation_tolerance_deg,
        )
        best_metric = float(np.linalg.norm(error))
        best_pose = current_center.copy()
        history: list[dict[str, Any]] = []
        for iteration in range(args.max_iterations):
            if success:
                break
            center = current_center.copy()
            columns: list[np.ndarray] = []
            probes: list[dict[str, Any]] = []
            for joint in args.active_joints:
                plus = center.copy()
                minus = center.copy()
                plus[joint] += args.probe_step_deg
                minus[joint] -= args.probe_step_deg
                _move_to_joint_pose(
                    sdk,
                    goal_deg=plus,
                    joint_limits=joint_limits,
                    gripper_stroke_mm=gripper_stroke_mm,
                    args=args,
                    home=home,
                    log=log,
                    label=f"iter{iteration}_j{joint + 1}_plus",
                )
                current_center = plus
                plus_report, plus_error = capture_report(
                    f"iter{iteration:02d}_j{joint + 1}_plus"
                )
                _move_to_joint_pose(
                    sdk,
                    goal_deg=minus,
                    joint_limits=joint_limits,
                    gripper_stroke_mm=gripper_stroke_mm,
                    args=args,
                    home=home,
                    log=log,
                    label=f"iter{iteration}_j{joint + 1}_minus",
                )
                current_center = minus
                minus_report, minus_error = capture_report(
                    f"iter{iteration:02d}_j{joint + 1}_minus"
                )
                derivative = (plus_error - minus_error) / (2.0 * args.probe_step_deg)
                columns.append(derivative)
                probes.append(
                    {
                        "joint_1_based": joint + 1,
                        "plus_alignment": plus_report,
                        "minus_alignment": minus_report,
                        "derivative_per_deg": derivative.tolist(),
                    }
                )
                _move_to_joint_pose(
                    sdk,
                    goal_deg=center,
                    joint_limits=joint_limits,
                    gripper_stroke_mm=gripper_stroke_mm,
                    args=args,
                    home=home,
                    log=log,
                    label=f"iter{iteration}_return_center_j{joint + 1}",
                )
                current_center = center

            jacobian = np.column_stack(columns)
            active_offset = center[list(args.active_joints)] - start_deg[list(args.active_joints)]
            solution = damped_joint_update(
                error,
                jacobian,
                damping=args.damping,
                max_step_deg=args.max_update_step_deg,
                current_offset_deg=active_offset,
                max_total_offset_deg=args.max_total_offset_deg,
            )
            iteration_record: dict[str, Any] = {
                "iteration": iteration,
                "center_joints_deg": center.tolist(),
                "alignment_before": report,
                "error_before": error.tolist(),
                "jacobian": jacobian.tolist(),
                "probes": probes,
                "raw_step_deg": solution.raw_step_deg.tolist(),
                "bounded_step_deg": solution.step_deg.tolist(),
                "singular_values": solution.singular_values.tolist(),
                "rank": solution.rank,
                "condition_number": solution.condition_number,
                "predicted_error_before": solution.predicted_error_before,
                "predicted_error_after": solution.predicted_error_after,
                "line_search": [],
            }

            accepted = False
            for factor in (1.0, 0.5, 0.25):
                candidate = center.copy()
                candidate[list(args.active_joints)] += solution.step_deg * factor
                _move_to_joint_pose(
                    sdk,
                    goal_deg=candidate,
                    joint_limits=joint_limits,
                    gripper_stroke_mm=gripper_stroke_mm,
                    args=args,
                    home=home,
                    log=log,
                    label=f"iter{iteration}_candidate_{factor}",
                )
                current_center = candidate
                candidate_report, candidate_error = capture_report(
                    f"iter{iteration:02d}_candidate_{factor:.2f}"
                )
                candidate_metric = float(np.linalg.norm(candidate_error))
                actual_improvement = float(np.linalg.norm(error)) - candidate_metric
                iteration_record["line_search"].append(
                    {
                        "factor": factor,
                        "candidate_joints_deg": candidate.tolist(),
                        "alignment": candidate_report,
                        "metric": candidate_metric,
                        "actual_improvement": actual_improvement,
                    }
                )
                if actual_improvement >= args.minimum_actual_improvement_px:
                    report = candidate_report
                    error = candidate_error
                    accepted = True
                    if candidate_metric < best_metric:
                        best_metric = candidate_metric
                        best_pose = candidate.copy()
                    break
                _move_to_joint_pose(
                    sdk,
                    goal_deg=center,
                    joint_limits=joint_limits,
                    gripper_stroke_mm=gripper_stroke_mm,
                    args=args,
                    home=home,
                    log=log,
                    label=f"iter{iteration}_reject_{factor}",
                )
                current_center = center

            iteration_record["accepted"] = accepted
            history.append(iteration_record)
            log.write({"event": "solver_iteration", **iteration_record})
            if not accepted:
                raise RuntimeError(
                    "visual calibration line search found no measured improvement; "
                    "the selected wrist joints cannot safely explain the camera offset"
                )
            success = alignment_converged(
                report,
                translation_tolerance_px=args.translation_tolerance_px,
                rotation_tolerance_deg=args.rotation_tolerance_deg,
            )

        if not success:
            raise RuntimeError(
                "visual calibration did not reach tolerance after "
                f"{args.max_iterations} iterations; best_metric={best_metric:.3f}"
            )

        final_measured_pose = np.asarray(
            home.raw_feedback_pose(sdk.get_status()).joints_deg, dtype=np.float64
        )
        if current_center is None:
            raise RuntimeError("calibration completed without a commanded center")
        offset = np.zeros(6, dtype=np.float64)
        offset[list(args.active_joints)] = (
            current_center[list(args.active_joints)]
            - saved_home_deg[list(args.active_joints)]
        )
        candidate = {
            "schema": "assembly_new1_wrist_visual_zero_offset_candidate_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "candidate_not_activated",
            "model_training_home_joints_deg": saved_home_deg.tolist(),
            "hardware_visual_home_command_joints_deg": current_center.tolist(),
            "hardware_visual_home_measured_joints_deg": final_measured_pose.tolist(),
            "hardware_minus_model_joint_offset_deg": offset.tolist(),
            "active_joints_1_based": [joint + 1 for joint in args.active_joints],
            "baseline_alignment": baseline_report,
            "final_alignment": report,
            "translation_tolerance_px": args.translation_tolerance_px,
            "rotation_tolerance_deg": args.rotation_tolerance_deg,
            "history": history,
            "deployment_files_modified": False,
        }
        candidate_path = run_dir / "candidate_joint_zero_offset.json"
        candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        home.latch_current_pose(sdk)
        log.write(
            {
                "event": "calibration_complete",
                "candidate": candidate,
                "candidate_path": str(candidate_path),
                "command_sent": True,
            }
        )
        print(json.dumps(candidate, indent=2))
        print(
            "CALIBRATION CANDIDATE COMPLETE: pose is held, but deployment mapping "
            f"was not changed. candidate={candidate_path}"
        )
    except BaseException as error:
        log.write(
            {
                "event": "calibration_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "command_sent": bool(motion_enabled),
            }
        )
        if sdk is not None and motion_enabled:
            if (
                args.return_to_start_on_failure
                and start_deg is not None
                and current_center is not None
            ):
                try:
                    _move_to_joint_pose(
                        sdk,
                        goal_deg=start_deg,
                        joint_limits=joint_limits,
                        gripper_stroke_mm=gripper_stroke_mm,
                        args=args,
                        home=home,
                        log=log,
                        label="failure_return_to_start",
                    )
                    current_center = start_deg.copy()
                    log.write(
                        {
                            "event": "failure_return_complete",
                            "joints_deg": start_deg.tolist(),
                            "command_sent": True,
                        }
                    )
                except BaseException as return_error:
                    log.write(
                        {
                            "event": "failure_return_failed",
                            "error": repr(return_error),
                            "command_sent": False,
                        }
                    )
            try:
                held = home.latch_current_pose(sdk)
                log.write(
                    {
                        "event": "current_pose_latched",
                        "normalized_state": held,
                        "command_sent": True,
                    }
                )
            except BaseException as hold_error:
                log.write(
                    {
                        "event": "current_pose_latch_failed",
                        "error": repr(hold_error),
                        "command_sent": False,
                    }
                )
        raise
    finally:
        if camera is not None:
            camera.close()
        if sdk is not None:
            sdk.disconnect(home_first=False, disable_torque=False)
        log.close()


if __name__ == "__main__":
    main()
