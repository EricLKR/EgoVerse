#!/usr/bin/env python3
"""Guarded, model-free replay of an exported Assembly New1 episode.

``check`` reads only the NPZ and calibration. ``live-dry-run`` additionally
opens observation-only CAN feedback. ``execute`` requires an already enabled
arm at the recorded original-leader start pose and explicit ``--allow-motion``.
No checkpoint, camera, FK, IK, or policy is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = BUNDLE_ROOT / "assets/piper_original_leader_calibration.json"
DEFAULT_MODEL_CALIBRATION = BUNDLE_ROOT / "assets/piper_follower_calibration.json"
DEFAULT_ROLE_MAPPING = BUNDLE_ROOT / "assets/reset_control.json"
DEFAULT_LOG_DIR = BUNDLE_ROOT / "logs/episode_replay"
CONTROL_HZ = 30.0
J2_INDEX = 1
J3_INDEX = 2
J5_INDEX = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayTrajectory:
    metadata: dict[str, Any]
    frame_index: np.ndarray
    action_percent: np.ndarray
    model_follower_joint_deg: np.ndarray
    hardware_joint_deg: np.ndarray
    recorded_follower_gripper_mm: np.ndarray
    gripper_percent: np.ndarray


class JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("x", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self._stream.write(json.dumps(payload, allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def firmware_feedback_to_replay_joints(
    joints_deg: np.ndarray | tuple[float, ...] | list[float],
    j2_feedback_offset_deg: float,
    j2_command_offset_deg: float = 0.0,
    j3_command_offset_deg: float = 0.0,
    j5_command_offset_deg: float = 0.0,
) -> np.ndarray:
    """Convert firmware joint feedback into the recorded replay coordinate frame.

    ``j2_feedback_offset_deg`` is added to the raw firmware J2 reading.  For the
    observed post-recalibration ``+6 deg`` firmware zero shift, pass ``-6``.
    When replay actions instead receive a command-only offset, subtract that
    offset for start/tracking comparisons while preserving the separately
    logged raw firmware feedback unchanged.
    """

    corrected = np.asarray(joints_deg, dtype=np.float64).copy()
    if corrected.shape != (6,):
        raise ValueError(f"expected six joint values, got {corrected.shape}")
    corrected[J2_INDEX] += float(j2_feedback_offset_deg) - float(
        j2_command_offset_deg
    )
    corrected[J3_INDEX] -= float(j3_command_offset_deg)
    corrected[J5_INDEX] -= float(j5_command_offset_deg)
    return corrected


def replay_actions_to_firmware_commands(
    joints_deg: np.ndarray | tuple[float, ...] | list[float],
    j2_command_offset_deg: float,
    j3_command_offset_deg: float = 0.0,
    j5_command_offset_deg: float = 0.0,
) -> np.ndarray:
    """Apply command-only joint offsets without modifying replay actions in place."""

    commanded = np.asarray(joints_deg, dtype=np.float64).copy()
    if commanded.shape[-1:] != (6,) or commanded.ndim not in (1, 2):
        raise ValueError(f"expected shape (6,) or (N,6), got {commanded.shape}")
    commanded[..., J2_INDEX] += float(j2_command_offset_deg)
    commanded[..., J3_INDEX] += float(j3_command_offset_deg)
    commanded[..., J5_INDEX] += float(j5_command_offset_deg)
    if not np.isfinite(commanded).all():
        raise ValueError("non-finite joint command after applying joint offsets")
    return commanded


def validate_firmware_command_limits(
    commanded_joints_deg: np.ndarray,
    calibration_path: Path,
    *,
    max_overshoot_deg: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and clamp small, explicitly allowed command-limit overshoots.

    Exported follower feedback can differ slightly from the current arm's
    calibrated endpoints.  Values within ``max_overshoot_deg`` are clamped to
    the current hardware range; larger incompatibilities remain a hard error.
    """

    if not math.isfinite(max_overshoot_deg) or max_overshoot_deg < 0.0:
        raise ValueError("max_overshoot_deg must be finite and non-negative")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    commanded = np.asarray(commanded_joints_deg, dtype=np.float64)
    if commanded.ndim != 2 or commanded.shape[1] != 6:
        raise ValueError(f"expected firmware commands with shape (N,6), got {commanded.shape}")
    lows = np.asarray(
        [float(calibration[f"joint_{index}"]["range_min"]) / 1000.0 for index in range(6)],
        dtype=np.float64,
    )
    highs = np.asarray(
        [float(calibration[f"joint_{index}"]["range_max"]) / 1000.0 for index in range(6)],
        dtype=np.float64,
    )
    overshoot = np.maximum(
        np.maximum(lows[None, :] - commanded, 0.0),
        np.maximum(commanded - highs[None, :], 0.0),
    )
    maximum = float(overshoot.max(initial=0.0))
    if maximum > max_overshoot_deg + 1e-9:
        flat_index = int(np.argmax(overshoot))
        _, joint_index = np.unravel_index(flat_index, overshoot.shape)
        raise RuntimeError(
            f"replay command exceeds J{joint_index + 1} hardware calibration limits: "
            f"range=[{lows[joint_index]:.3f},{highs[joint_index]:.3f}] deg, "
            f"max_overshoot={maximum:.3f} deg, "
            f"allowed={max_overshoot_deg:.3f} deg"
        )
    bounded = np.clip(commanded, lows[None, :], highs[None, :])
    report: dict[str, Any] = {}
    for index in range(6):
        name = f"j{index + 1}"
        joint_overshoot = overshoot[:, index]
        report.update(
            {
                f"{name}_firmware_command_limit_deg": [lows[index], highs[index]],
                f"{name}_firmware_command_limit_overshoot_count": int(
                    np.count_nonzero(joint_overshoot)
                ),
                f"{name}_firmware_command_max_limit_overshoot_deg": float(
                    joint_overshoot.max(initial=0.0)
                ),
            }
        )
    report.update(
        {
            "firmware_command_limit_allowed_overshoot_deg": max_overshoot_deg,
            "firmware_command_limit_clamp_count": int(np.count_nonzero(overshoot)),
            "firmware_command_limit_clamped_tick_count": int(
                np.count_nonzero(np.any(overshoot > 0.0, axis=1))
            ),
            "firmware_command_max_limit_overshoot_deg": maximum,
        }
    )
    return bounded, report


def load_trajectory(path: Path, calibration_path: Path) -> ReplayTrajectory:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        trajectory = ReplayTrajectory(
            metadata=metadata,
            frame_index=np.asarray(payload["frame_index"], dtype=np.int64),
            action_percent=np.asarray(payload["action_percent"], dtype=np.float64),
            model_follower_joint_deg=np.asarray(
                payload["model_follower_joint_deg"], dtype=np.float64
            ),
            hardware_joint_deg=np.asarray(
                payload["hardware_joint_deg"], dtype=np.float64
            ),
            recorded_follower_gripper_mm=np.asarray(
                payload["recorded_follower_gripper_mm"], dtype=np.float64
            ),
            gripper_percent=np.asarray(payload["gripper_percent"], dtype=np.float64),
        )
    count = len(trajectory.frame_index)
    expected_shapes = {
        "action_percent": (count, 7),
        "model_follower_joint_deg": (count, 6),
        "hardware_joint_deg": (count, 6),
        "recorded_follower_gripper_mm": (count,),
        "gripper_percent": (count,),
    }
    for name, shape in expected_shapes.items():
        value = getattr(trajectory, name)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid {name}: shape={value.shape}, expected={shape}")
    if metadata.get("schema") != "assembly_new1_physical_joint_episode_replay_v2":
        raise ValueError(f"unsupported replay schema: {metadata.get('schema')!r}")
    expected_hash = metadata.get("leader_calibration_sha256")
    actual_hash = sha256_file(calibration_path)
    if expected_hash != actual_hash:
        raise RuntimeError(
            "replay trajectory calibration differs from deployment hardware calibration: "
            f"trajectory={expected_hash}, current={actual_hash}"
        )
    if not np.array_equal(trajectory.action_percent[:, :6], np.clip(
        trajectory.action_percent[:, :6], -100.0, 100.0
    )):
        raise ValueError("replay joint percentiles exceed [-100,100]")
    if np.any(trajectory.gripper_percent < 0.0) or np.any(
        trajectory.gripper_percent > 100.0
    ):
        raise ValueError("hardware gripper percentage exceeds [0,100]")
    return trajectory


def select_frames(
    trajectory: ReplayTrajectory,
    start_frame: int,
    end_frame: int | None,
    *,
    gripper_command_mode: str = "recorded-aperture",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stop = len(trajectory.frame_index) if end_frame is None else end_frame + 1
    if start_frame < 0 or stop <= start_frame or stop > len(trajectory.frame_index):
        raise ValueError(
            f"invalid replay frame range [{start_frame},{stop}); trajectory has "
            f"{len(trajectory.frame_index)} frames"
        )
    if gripper_command_mode == "recorded-aperture":
        gripper = trajectory.gripper_percent
    elif gripper_command_mode == "binary-action":
        # The seventh action element is the operator's intended gripper
        # command.  Use it for open/close replay instead of thresholding the
        # measured aperture, which can stop halfway when an object is held.
        gripper = trajectory.action_percent[:, 6]
    else:
        raise ValueError(f"unsupported gripper command mode: {gripper_command_mode}")
    return (
        trajectory.hardware_joint_deg[start_frame:stop],
        gripper[start_frame:stop],
        trajectory.frame_index[start_frame:stop],
    )


def resample_bounded(
    joints_deg: np.ndarray,
    gripper_percent: np.ndarray,
    source_frames: np.ndarray,
    *,
    time_scale: float,
    max_joint_step_deg: float,
    max_gripper_step_percent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(joints_deg) == 0:
        raise ValueError("cannot resample an empty trajectory")
    output_joints = [np.asarray(joints_deg[0], dtype=np.float64)]
    output_gripper = [float(gripper_percent[0])]
    output_source = [float(source_frames[0])]
    minimum_substeps = max(1, int(math.ceil(time_scale)))
    for index in range(1, len(joints_deg)):
        joint_delta = np.asarray(joints_deg[index] - joints_deg[index - 1])
        gripper_delta = float(gripper_percent[index] - gripper_percent[index - 1])
        substeps = max(
            minimum_substeps,
            int(math.ceil(float(np.max(np.abs(joint_delta))) / max_joint_step_deg)),
            int(math.ceil(abs(gripper_delta) / max_gripper_step_percent)),
        )
        for substep in range(1, substeps + 1):
            fraction = substep / substeps
            output_joints.append(joints_deg[index - 1] + fraction * joint_delta)
            output_gripper.append(gripper_percent[index - 1] + fraction * gripper_delta)
            source_delta = float(source_frames[index] - source_frames[index - 1])
            output_source.append(source_frames[index - 1] + fraction * source_delta)
    result_joints = np.asarray(output_joints, dtype=np.float64)
    result_gripper = np.asarray(output_gripper, dtype=np.float64)
    result_source = np.asarray(output_source, dtype=np.float64)
    if len(result_joints) > 1:
        observed_max = float(np.max(np.abs(np.diff(result_joints, axis=0))))
        if observed_max > max_joint_step_deg + 1e-9:
            raise AssertionError(f"resampler exceeded joint step bound: {observed_max}")
    return result_joints, result_gripper, result_source


def trajectory_report(
    trajectory: ReplayTrajectory,
    selected_joints: np.ndarray,
    selected_gripper: np.ndarray,
    replay_joints: np.ndarray,
    firmware_command_joints: np.ndarray,
    replay_gripper: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_delta = np.abs(np.diff(selected_joints, axis=0))
    replay_delta = np.abs(np.diff(replay_joints, axis=0))
    return {
        "event": "replay_preflight",
        "mode": args.mode,
        "trajectory": str(args.trajectory),
        "episode_index": trajectory.metadata["episode_index"],
        "source_frame_range": [args.start_frame, args.end_frame],
        "replay_direction": "reverse" if args.reverse else "forward",
        "joint_mapping_mode": args.joint_mapping_mode,
        "j2_feedback_offset_deg": args.j2_feedback_offset_deg,
        "j2_action_offset_deg": args.j2_command_offset_deg,
        "j3_action_offset_deg": args.j3_command_offset_deg,
        "j5_action_offset_deg": args.j5_command_offset_deg,
        "source_frame_count": int(len(selected_joints)),
        "replay_tick_count": int(len(replay_joints)),
        "time_scale": args.time_scale,
        "estimated_duration_s": len(replay_joints) / CONTROL_HZ,
        "first_hardware_joints_deg": selected_joints[0].tolist(),
        "first_firmware_command_joints_deg": firmware_command_joints[0].tolist(),
        "first_gripper_percent": float(selected_gripper[0]),
        "last_hardware_joints_deg": selected_joints[-1].tolist(),
        "last_firmware_command_joints_deg": firmware_command_joints[-1].tolist(),
        "last_gripper_percent": float(selected_gripper[-1]),
        "gripper_command_mode": (
            "hold_closed_zero_percent"
            if args.hold_gripper_closed
            else args.gripper_command_mode
        ),
        "gripper_binary_threshold_percent": args.gripper_binary_threshold_percent,
        "gripper_effort_sdk_units": args.gripper_effort,
        "gripper_effort_nm": args.gripper_effort * 0.001,
        "first_commanded_gripper_percent": commanded_gripper_percent(
            float(selected_gripper[0]),
            hold_gripper_closed=args.hold_gripper_closed,
            command_mode=args.gripper_command_mode,
            binary_threshold_percent=args.gripper_binary_threshold_percent,
        ),
        "last_commanded_gripper_percent": commanded_gripper_percent(
            float(selected_gripper[-1]),
            hold_gripper_closed=args.hold_gripper_closed,
            command_mode=args.gripper_command_mode,
            binary_threshold_percent=args.gripper_binary_threshold_percent,
        ),
        "source_max_joint_step_deg": (
            float(selected_delta.max()) if len(selected_delta) else 0.0
        ),
        "replay_max_joint_step_deg": (
            float(replay_delta.max()) if len(replay_delta) else 0.0
        ),
        "replay_max_gripper_step_percent": (
            float(np.max(np.abs(np.diff(replay_gripper))))
            if len(replay_gripper) > 1
            else 0.0
        ),
        "checkpoint_loaded": False,
        "policy_constructed": False,
        "camera_opened": False,
        "command_sent": False,
    }


def import_hardware_helpers():
    import home_assembly_new1 as home

    return home


def commanded_gripper_percent(
    recorded_percent: float,
    *,
    hold_gripper_closed: bool,
    command_mode: str = "recorded-aperture",
    binary_threshold_percent: float = 50.0,
) -> float:
    """Convert recorded intent to the actual gripper command.

    ``binary-action`` always emits exactly 0% (closed) or 100% (open).  This
    deliberately does not replay measured contact aperture.
    """
    if hold_gripper_closed:
        return 0.0
    if command_mode == "recorded-aperture":
        return float(recorded_percent)
    if command_mode == "binary-action":
        return 100.0 if recorded_percent >= binary_threshold_percent else 0.0
    raise ValueError(f"unsupported gripper command mode: {command_mode}")


def latch_replay_pose(
    home,
    sdk,
    *,
    hold_gripper_closed: bool,
    commanded_gripper_target_percent: float | None = None,
) -> list[float]:
    """Latch joints while preserving an explicit open/close target."""
    current = home.normalized_feedback(sdk)
    if hold_gripper_closed:
        current[6] = 0.0
    elif commanded_gripper_target_percent is not None:
        current[6] = float(commanded_gripper_target_percent)
    sdk.set_joint_positions(current.tolist())
    return current.tolist()


def run_live(
    args: argparse.Namespace,
    trajectory: ReplayTrajectory,
    replay_joints: np.ndarray,
    firmware_command_joints: np.ndarray,
    replay_gripper: np.ndarray,
    replay_source: np.ndarray,
    report: dict[str, Any],
    log: JsonlLogger,
) -> None:
    home = import_hardware_helpers()
    joint_limits, gripper_stroke_mm = home.load_calibration(args.calibration_json)
    sdk = None
    motion_adopted = False
    try:
        sdk = home.PiperSDKInterface(
            port=args.can_port,
            gripper_stroke_mm=gripper_stroke_mm,
            gripper_effort=args.gripper_effort,
            speed_percent=args.speed_percent,
            joint_limits_deg=joint_limits,
            enable_motion=False,
        )
        current = home.raw_feedback_pose(sdk.get_status())
        current_raw_joints = np.asarray(current.joints_deg, dtype=np.float64)
        current_replay_joints = firmware_feedback_to_replay_joints(
            current_raw_joints,
            args.j2_feedback_offset_deg,
            args.j2_command_offset_deg,
            args.j3_command_offset_deg,
            args.j5_command_offset_deg,
        )
        start_error = np.abs(
            current_replay_joints - replay_joints[0]
        )
        live_report = {
            **report,
            "current_firmware_joints_deg": current_raw_joints.tolist(),
            "current_replay_frame_joints_deg": current_replay_joints.tolist(),
            "current_gripper_mm": current.gripper_mm,
            "gripper_effort_sdk_units": args.gripper_effort,
            "gripper_effort_nm": args.gripper_effort * 0.001,
            "gripper_command_mode": (
                "hold_closed_zero_percent"
                if args.hold_gripper_closed
                else args.gripper_command_mode
            ),
            "max_held_gripper_aperture_mm": args.max_held_gripper_aperture_mm,
            "start_joint_error_deg": start_error.tolist(),
            "start_max_joint_error_deg": float(start_error.max()),
            "controller_status": str(sdk.piper.GetArmStatus().arm_status),
        }
        log.write(live_report)
        print(json.dumps(live_report, indent=2))
        if args.mode == "live-dry-run":
            print(f"LIVE DRY RUN PASSED: observation-only; no command sent. log={log.path}")
            return
        if float(start_error.max()) > args.max_start_joint_error_deg:
            raise RuntimeError(
                "current pose is not at the replay start: "
                f"max_error={float(start_error.max()):.3f} deg, "
                f"limit={args.max_start_joint_error_deg:.3f} deg"
            )
        if (
            args.hold_gripper_closed
            and current.gripper_mm > args.max_held_gripper_aperture_mm
        ):
            raise RuntimeError(
                "closed-gripper replay requires a confirmed grasp aperture; "
                f"feedback={current.gripper_mm:.3f} mm exceeds "
                f"limit={args.max_held_gripper_aperture_mm:.3f} mm"
            )
        motors = home.verify_health(sdk, require_enabled=False, check_controller=False)
        enabled = [bool(motor["driver_enabled"]) for motor in motors]
        if not all(enabled):
            raise RuntimeError(
                "episode replay requires the all-enabled state left by Home; "
                f"driver_enabled={enabled}"
            )
        home.adopt_already_enabled_motion_hold_current(sdk)
        motion_adopted = True
        bad_tracking_cycles = 0
        last_command_gripper_pct = commanded_gripper_percent(
            float(replay_gripper[0]),
            hold_gripper_closed=args.hold_gripper_closed,
            command_mode=args.gripper_command_mode,
            binary_threshold_percent=args.gripper_binary_threshold_percent,
        )
        for tick, (joints, command_joints, gripper_pct, source_frame) in enumerate(
            zip(
                replay_joints,
                firmware_command_joints,
                replay_gripper,
                replay_source,
                strict=True,
            )
        ):
            tick_started = time.perf_counter()
            home.verify_health(sdk, require_enabled=True, check_controller=False)
            command_gripper_pct = commanded_gripper_percent(
                float(gripper_pct),
                hold_gripper_closed=args.hold_gripper_closed,
                command_mode=args.gripper_command_mode,
                binary_threshold_percent=args.gripper_binary_threshold_percent,
            )
            last_command_gripper_pct = command_gripper_pct
            goal = home.PhysicalPose(
                tuple(float(value) for value in command_joints),
                float(command_gripper_pct / 100.0 * gripper_stroke_mm),
            )
            normalized = home.physical_pose_to_normalized(
                goal, joint_limits, gripper_stroke_mm
            )
            command = home.bounded_send(
                sdk,
                normalized,
                max_arm_relative_target=args.max_relative_target,
                max_gripper_relative_target=args.max_gripper_relative_target,
            )
            observed = home.raw_feedback_pose(sdk.get_status())
            observed_firmware_joints = np.asarray(
                observed.joints_deg, dtype=np.float64
            )
            observed_replay_joints = firmware_feedback_to_replay_joints(
                observed_firmware_joints,
                args.j2_feedback_offset_deg,
                args.j2_command_offset_deg,
                args.j3_command_offset_deg,
                args.j5_command_offset_deg,
            )
            tracking_error = float(
                np.max(np.abs(observed_replay_joints - joints))
            )
            bad_tracking_cycles = (
                bad_tracking_cycles + 1
                if tracking_error > args.max_tracking_error_deg
                else 0
            )
            log.write(
                {
                    "event": "replay_tick",
                    "tick": tick,
                    "source_frame": float(source_frame),
                    "target_replay_frame_joints_deg": joints.tolist(),
                    "target_firmware_joints_deg": command_joints.tolist(),
                    "recorded_target_gripper_percent": float(gripper_pct),
                    "commanded_target_gripper_percent": command_gripper_pct,
                    "observed_firmware_joints_deg": observed_firmware_joints.tolist(),
                    "observed_replay_frame_joints_deg": observed_replay_joints.tolist(),
                    "observed_gripper_mm": observed.gripper_mm,
                    "max_tracking_error_deg": tracking_error,
                    "command": command,
                    "command_sent": True,
                }
            )
            if bad_tracking_cycles >= args.max_tracking_error_cycles:
                raise RuntimeError(
                    "replay tracking error remained excessive: "
                    f"error={tracking_error:.3f} deg for {bad_tracking_cycles} cycles"
                )
            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - tick_started)))
        held = latch_replay_pose(
            home,
            sdk,
            hold_gripper_closed=args.hold_gripper_closed,
            commanded_gripper_target_percent=last_command_gripper_pct,
        )
        log.write(
            {
                "event": "replay_complete",
                "ticks": len(replay_joints),
                "normalized_hold": held,
                "command_sent": True,
            }
        )
        print(f"REPLAY COMPLETE: final pose latched. log={log.path}")
    except BaseException as error:
        log.write(
            {
                "event": "replay_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "command_sent": False,
            }
        )
        if sdk is not None and motion_adopted:
            try:
                latch_replay_pose(
                    home,
                    sdk,
                    hold_gripper_closed=args.hold_gripper_closed,
                    commanded_gripper_target_percent=(
                        last_command_gripper_pct if motion_adopted else None
                    ),
                )
            except BaseException as hold_error:
                log.write(
                    {
                        "event": "replay_abort_hold_failed",
                        "error": repr(hold_error),
                        "command_sent": False,
                    }
                )
        raise
    finally:
        if sdk is not None:
            sdk.disconnect(home_first=False, disable_torque=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "live-dry-run", "execute"), default="check")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--model-calibration-json", type=Path, default=DEFAULT_MODEL_CALIBRATION
    )
    parser.add_argument("--role-mapping-json", type=Path, default=DEFAULT_ROLE_MAPPING)
    parser.add_argument(
        "--joint-mapping-mode",
        choices=("physical-identity", "recorded-percentile-affine"),
        default="physical-identity",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--can-port")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="execute the selected inclusive frame range from end to start",
    )
    parser.add_argument("--time-scale", type=float, default=4.0)
    parser.add_argument("--max-joint-step-deg", type=float, default=0.25)
    parser.add_argument("--max-gripper-step-percent", type=float, default=5.0)
    parser.add_argument("--max-start-joint-error-deg", type=float, default=1.0)
    parser.add_argument(
        "--max-trajectory-joint-limit-overshoot-deg",
        type=float,
        default=0.0,
        help=(
            "maximum source-command endpoint mismatch that may be clamped to "
            "the current hardware calibration; zero keeps strict validation"
        ),
    )
    parser.add_argument("--max-tracking-error-deg", type=float, default=8.0)
    parser.add_argument("--max-tracking-error-cycles", type=int, default=15)
    parser.add_argument(
        "--j2-feedback-offset-deg",
        type=float,
        default=0.0,
        help=(
            "value added to raw firmware J2 feedback before replay comparisons; "
            "replay action commands are unchanged (use -6 to subtract 6 deg from feedback)"
        ),
    )
    parser.add_argument(
        "--j2-command-offset-deg",
        type=float,
        default=0.0,
        help=(
            "value added only to replay J2 immediately before firmware command "
            "generation; raw firmware feedback is unchanged (use 6 to command J2 +6 deg)"
        ),
    )
    parser.add_argument(
        "--j3-command-offset-deg",
        type=float,
        default=0.0,
        help=(
            "value added only to replay J3 immediately before firmware command "
            "generation; raw feedback is inverse-mapped for replay comparisons"
        ),
    )
    parser.add_argument(
        "--j5-command-offset-deg",
        type=float,
        default=0.0,
        help=(
            "value added only to replay J5 immediately before firmware command "
            "generation; raw feedback is inverse-mapped for replay comparisons"
        ),
    )
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--max-relative-target", type=float, default=0.5)
    parser.add_argument("--max-gripper-relative-target", type=float, default=100.0)
    parser.add_argument(
        "--gripper-command-mode",
        choices=("recorded-aperture", "binary-action"),
        default="recorded-aperture",
        help=(
            "record measured aperture, or threshold the recorded seventh "
            "action element and emit only fully closed/open commands"
        ),
    )
    parser.add_argument(
        "--gripper-binary-threshold-percent",
        type=float,
        default=50.0,
        help="binary-action values below this close; values at/above it open",
    )
    parser.add_argument(
        "--gripper-effort",
        type=int,
        default=5000,
        help="PIPER gripper torque in 0.001 N*m SDK units (0..5000)",
    )
    parser.add_argument(
        "--hold-gripper-closed",
        action="store_true",
        help=(
            "ignore recorded contact aperture and command 0%% throughout this "
            "replay range; use only after visually confirming an object grasp"
        ),
    )
    parser.add_argument(
        "--max-held-gripper-aperture-mm",
        type=float,
        default=80.0,
        help="refuse --hold-gripper-closed above this measured aperture",
    )
    parser.add_argument("--allow-motion", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if not args.trajectory.is_file():
        parser.error(f"trajectory does not exist: {args.trajectory}")
    if not args.calibration_json.is_file():
        parser.error(f"calibration does not exist: {args.calibration_json}")
    if not args.model_calibration_json.is_file():
        parser.error(f"model calibration does not exist: {args.model_calibration_json}")
    if not args.role_mapping_json.is_file():
        parser.error(f"role mapping does not exist: {args.role_mapping_json}")
    if args.mode != "check" and not args.can_port:
        parser.error(f"{args.mode} requires --can-port")
    if args.mode == "execute" and not args.allow_motion:
        parser.error("execute mode requires --allow-motion")
    if args.mode != "execute" and args.allow_motion:
        parser.error("--allow-motion is valid only in execute mode")
    for name in (
        "time_scale",
        "max_joint_step_deg",
        "max_gripper_step_percent",
        "max_start_joint_error_deg",
        "max_tracking_error_deg",
        "max_relative_target",
        "max_gripper_relative_target",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if (
        not math.isfinite(args.max_trajectory_joint_limit_overshoot_deg)
        or args.max_trajectory_joint_limit_overshoot_deg < 0.0
    ):
        parser.error(
            "--max-trajectory-joint-limit-overshoot-deg must be finite and non-negative"
        )
    if args.max_tracking_error_cycles < 1:
        parser.error("--max-tracking-error-cycles must be positive")
    if (
        not math.isfinite(args.j2_feedback_offset_deg)
        or abs(args.j2_feedback_offset_deg) > 15.0
    ):
        parser.error("--j2-feedback-offset-deg must be finite and within [-15,15]")
    if (
        not math.isfinite(args.j2_command_offset_deg)
        or abs(args.j2_command_offset_deg) > 15.0
    ):
        parser.error("--j2-command-offset-deg must be finite and within [-15,15]")
    if (
        not math.isfinite(args.j3_command_offset_deg)
        or abs(args.j3_command_offset_deg) > 15.0
    ):
        parser.error("--j3-command-offset-deg must be finite and within [-15,15]")
    if (
        not math.isfinite(args.j5_command_offset_deg)
        or abs(args.j5_command_offset_deg) > 15.0
    ):
        parser.error("--j5-command-offset-deg must be finite and within [-15,15]")
    if args.j2_feedback_offset_deg != 0.0 and args.j2_command_offset_deg != 0.0:
        parser.error(
            "do not combine J2 feedback and command offsets in one replay; "
            "this experiment requires raw feedback"
        )
    if not 0 <= args.gripper_effort <= 5000:
        parser.error("--gripper-effort must be in [0,5000]")
    if not 0.0 <= args.gripper_binary_threshold_percent <= 100.0:
        parser.error("--gripper-binary-threshold-percent must be in [0,100]")
    if (
        not math.isfinite(args.max_held_gripper_aperture_mm)
        or args.max_held_gripper_aperture_mm <= 0
    ):
        parser.error("--max-held-gripper-aperture-mm must be finite and positive")
    if not 1 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [1,100]")
    return args


def main() -> None:
    args = parse_args()
    trajectory = load_trajectory(args.trajectory, args.calibration_json)
    selected_joints, selected_gripper, selected_frames = select_frames(
        trajectory,
        args.start_frame,
        args.end_frame,
        gripper_command_mode=args.gripper_command_mode,
    )
    if args.joint_mapping_mode != "physical-identity":
        from joint_space_adapter import JointSpaceAdapter

        adapter = JointSpaceAdapter.from_files(
            mapping_path=args.role_mapping_json,
            model_calibration_path=args.model_calibration_json,
            hardware_calibration_path=args.calibration_json,
            mapping_mode=args.joint_mapping_mode,
        )
        selected_joints = np.asarray(
            [
                adapter.model_degrees_to_hardware_degrees(joints)
                for joints in selected_joints
            ],
            dtype=np.float64,
        )
    if args.reverse:
        selected_joints = selected_joints[::-1].copy()
        selected_gripper = selected_gripper[::-1].copy()
        selected_frames = selected_frames[::-1].copy()
    # A binary gripper changes command atomically.  It must not stretch the
    # arm timeline by inserting aperture interpolation ticks.
    resample_gripper_step = (
        100.0
        if args.gripper_command_mode == "binary-action"
        else args.max_gripper_step_percent
    )
    replay_joints, replay_gripper, replay_source = resample_bounded(
        selected_joints,
        selected_gripper,
        selected_frames,
        time_scale=args.time_scale,
        max_joint_step_deg=args.max_joint_step_deg,
        max_gripper_step_percent=resample_gripper_step,
    )
    firmware_command_joints = replay_actions_to_firmware_commands(
        replay_joints,
        args.j2_command_offset_deg,
        args.j3_command_offset_deg,
        args.j5_command_offset_deg,
    )
    firmware_command_joints, command_limit_report = validate_firmware_command_limits(
        firmware_command_joints,
        args.calibration_json,
        max_overshoot_deg=args.max_trajectory_joint_limit_overshoot_deg,
    )
    report = trajectory_report(
        trajectory,
        selected_joints,
        selected_gripper,
        replay_joints,
        firmware_command_joints,
        replay_gripper,
        args,
    )
    report.update(command_limit_report)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = JsonlLogger(args.log_dir / f"episode_replay_{args.mode}_{timestamp}.jsonl")
    try:
        log.write(report)
        if args.mode == "check":
            print(json.dumps(report, indent=2))
            print(f"CHECK PASSED: offline only; no robot or CAN opened. log={log.path}")
            return
        run_live(
            args,
            trajectory,
            replay_joints,
            firmware_command_joints,
            replay_gripper,
            replay_source,
            report,
            log,
        )
    finally:
        log.close()


if __name__ == "__main__":
    main()
