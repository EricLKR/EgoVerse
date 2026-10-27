#!/usr/bin/env python
"""Hold all six PIPER joints and operate only the gripper.

This model-free utility is intended for supervised setup between replay runs.
``check`` is observation-only. ``execute`` adopts an already-enabled arm
without quick-stop, continuously holds the measured six-joint pose, and opens
or closes the gripper with an explicit SDK torque setting.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import home_assembly_new1 as home


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = BUNDLE_ROOT / "logs/gripper_only"
CONTROL_HZ = 30.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "execute"), default="check")
    parser.add_argument("--can-port", required=True)
    parser.add_argument("--action", choices=("open", "close"), required=True)
    parser.add_argument(
        "--calibration-json", type=Path, default=home.DEFAULT_CALIBRATION
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument(
        "--gripper-effort",
        type=int,
        default=5000,
        help="PIPER motor torque in 0.001 N*m SDK units; 5000 = 5.0 N*m",
    )
    parser.add_argument("--gripper-velocity-mm-s", type=float, default=40.0)
    parser.add_argument("--timeout-s", type=float, default=8.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=1.5)
    parser.add_argument("--stall-frames", type=int, default=15)
    parser.add_argument("--stall-motion-tolerance-mm", type=float, default=0.4)
    parser.add_argument("--minimum-close-travel-mm", type=float, default=5.0)
    parser.add_argument("--allow-motion", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.mode == "execute" and not args.allow_motion:
        parser.error("execute mode requires --allow-motion")
    if not 0 <= args.gripper_effort <= 5000:
        parser.error("--gripper-effort must be in [0,5000]")
    if not 1 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [1,100]")
    for name in (
        "gripper_velocity_mm_s",
        "timeout_s",
        "position_tolerance_mm",
        "stall_motion_tolerance_mm",
        "minimum_close_travel_mm",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.stall_frames < 2:
        parser.error("--stall-frames must be at least 2")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    joint_limits, gripper_stroke_mm = home.load_calibration(args.calibration_json)
    target_mm = gripper_stroke_mm if args.action == "open" else 0.0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = home.JsonlLogger(args.log_dir / f"gripper_{args.action}_{args.mode}_{timestamp}.jsonl")
    sdk = None
    adopted = False
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
        motors = home.verify_health(
            sdk, require_enabled=False, check_controller=False
        )
        status = sdk.piper.GetArmStatus().arm_status
        report = {
            "event": "gripper_only_preflight",
            "mode": args.mode,
            "action": args.action,
            "current_joints_deg": list(current.joints_deg),
            "current_gripper_mm": current.gripper_mm,
            "target_gripper_mm": target_mm,
            "gripper_effort_sdk_units": args.gripper_effort,
            "gripper_effort_nm": args.gripper_effort * 0.001,
            "motor_enabled": [bool(motor["driver_enabled"]) for motor in motors],
            "motor_voltage_v": [float(motor["voltage_v"]) for motor in motors],
            "controller_status": str(status),
            "command_sent": False,
        }
        log.write(report)
        print(json.dumps(report, indent=2))
        if args.mode == "check":
            print(f"CHECK PASSED: observation-only; no command sent. log={log.path}")
            return

        enabled = [bool(motor["driver_enabled"]) for motor in motors]
        if not all(enabled):
            raise RuntimeError(
                "gripper-only execute requires all six arm motors already enabled; "
                f"driver_enabled={enabled}"
            )
        held = home.adopt_already_enabled_motion_hold_current(sdk)
        adopted = True
        held_joints = [float(held[f"joint_{index}.pos"]) for index in range(6)]
        start_mm = home.raw_feedback_pose(sdk.get_status()).gripper_mm
        direction = 1.0 if target_mm >= start_mm else -1.0
        started = time.perf_counter()
        recent = deque(maxlen=args.stall_frames)
        outcome = "timeout"
        last_mm = start_mm

        while time.perf_counter() - started <= args.timeout_s:
            tick_started = time.perf_counter()
            home.verify_health(sdk, require_enabled=True, check_controller=False)
            observed = home.raw_feedback_pose(sdk.get_status())
            last_mm = observed.gripper_mm
            recent.append(last_mm)
            elapsed = tick_started - started
            ramp_mm = start_mm + direction * args.gripper_velocity_mm_s * elapsed
            if direction > 0:
                requested_mm = min(target_mm, ramp_mm)
            else:
                requested_mm = max(target_mm, ramp_mm)
            requested_norm = max(
                0.0, min(100.0, requested_mm / gripper_stroke_mm * 100.0)
            )
            sdk.set_joint_positions([*held_joints, requested_norm])

            if abs(last_mm - target_mm) <= args.position_tolerance_mm:
                outcome = "target_reached"
                break
            if (
                args.action == "open"
                and last_mm - start_mm >= args.minimum_close_travel_mm
                and last_mm >= 0.9 * gripper_stroke_mm
                and len(recent) == recent.maxlen
                and max(recent) - min(recent) <= args.stall_motion_tolerance_mm
            ):
                # The calibration range can exceed the gripper's repeatable
                # mechanical open endpoint by a few millimetres. Treat a
                # stable aperture above 90% stroke as fully open instead of
                # reporting a false timeout.
                outcome = "open_endpoint_stall"
                break
            if (
                args.action == "close"
                and start_mm - last_mm >= args.minimum_close_travel_mm
                and len(recent) == recent.maxlen
                and max(recent) - min(recent) <= args.stall_motion_tolerance_mm
            ):
                outcome = "object_contact"
                break
            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - tick_started)))

        # Leave the firmware holding the requested endpoint. In particular,
        # after object contact the close target must remain zero so the selected
        # effort continues to clamp the object after this process disconnects.
        final_norm = target_mm / gripper_stroke_mm * 100.0
        for _ in range(5):
            sdk.set_joint_positions([*held_joints, final_norm])
            time.sleep(1.0 / CONTROL_HZ)
        final = home.raw_feedback_pose(sdk.get_status())
        result = {
            "event": "gripper_only_complete",
            "action": args.action,
            "outcome": outcome,
            "start_gripper_mm": start_mm,
            "final_gripper_mm": final.gripper_mm,
            "target_gripper_mm": target_mm,
            "gripper_effort_sdk_units": args.gripper_effort,
            "gripper_effort_nm": args.gripper_effort * 0.001,
            "held_joints_normalized": held_joints,
            "command_sent": True,
        }
        log.write(result)
        print(json.dumps(result, indent=2))
        if outcome == "timeout":
            raise RuntimeError(
                f"gripper {args.action} timed out at {final.gripper_mm:.3f} mm"
            )
        print(f"GRIPPER {args.action.upper()} COMPLETE: log={log.path}")
    except BaseException as error:
        log.write(
            {
                "event": "gripper_only_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "command_sent": bool(adopted),
            }
        )
        raise
    finally:
        if sdk is not None:
            sdk.disconnect(home_first=False, disable_torque=False)
        log.close()


if __name__ == "__main__":
    main()
