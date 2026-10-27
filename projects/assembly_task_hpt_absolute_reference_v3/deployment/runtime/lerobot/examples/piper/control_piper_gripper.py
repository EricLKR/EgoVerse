#!/usr/bin/env python
"""Open or close only the PIPER gripper while holding the six arm joints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lerobot.robots.piper.piper_sdk_interface import PiperSDKInterface


DEFAULT_CALIBRATION = Path("/mnt/wd10/kerun/calibration/my_piper_follower.json")


def load_calibration(path: Path) -> tuple[list[tuple[float, float]], float]:
    with path.open() as stream:
        calibration = json.load(stream)
    joint_limits = [
        (
            float(calibration[f"joint_{index}"]["range_min"]) / 1000.0,
            float(calibration[f"joint_{index}"]["range_max"]) / 1000.0,
        )
        for index in range(6)
    ]
    gripper_stroke_mm = float(calibration["gripper"]["range_max"]) / 1000.0
    return joint_limits, gripper_stroke_mm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("open", "close"), required=True)
    parser.add_argument("--can-port", default="can_follower")
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=4.0)
    parser.add_argument("--settle-frames", type=int, default=5)
    parser.add_argument(
        "--allow-motion",
        action="store_true",
        help="required acknowledgement that the gripper will move",
    )
    args = parser.parse_args()
    if not args.allow_motion:
        parser.error("--allow-motion is required")
    if not 0 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [0,100]")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    if args.settle_frames <= 0:
        parser.error("--settle-frames must be positive")
    return args


def main() -> None:
    args = parse_args()
    joint_limits, gripper_stroke_mm = load_calibration(args.calibration_json)
    target_pct = 100.0 if args.target == "open" else 0.0
    sdk = None
    last_feedback = None
    reached_frames = 0
    try:
        # Connect observation-only first. The guarded promotion below primes
        # the measured raw joint targets before enabling the controller.
        sdk = PiperSDKInterface(
            port=args.can_port,
            gripper_stroke_mm=gripper_stroke_mm,
            speed_percent=args.speed_percent,
            joint_limits_deg=joint_limits,
            enable_motion=False,
        )
        held = sdk.enable_motion_hold_current()
        arm_hold = [held[f"joint_{index}.pos"] for index in range(6)]
        deadline = time.monotonic() + args.timeout_s
        while time.monotonic() < deadline:
            sdk.set_joint_positions([*arm_hold, target_pct])
            last_feedback = float(sdk.get_status_normalized()["joint_6.pos"])
            reached = last_feedback >= 98.0 if args.target == "open" else last_feedback <= 2.0
            reached_frames = reached_frames + 1 if reached else 0
            if reached_frames >= args.settle_frames:
                break
            time.sleep(1.0 / 30.0)
        print(
            json.dumps(
                {
                    "target": args.target,
                    "target_pct": target_pct,
                    "feedback_pct": last_feedback,
                    "reached": reached_frames >= args.settle_frames,
                    "arm_target": "latched_measured_pose",
                },
                indent=2,
            )
        )
        if reached_frames < args.settle_frames:
            raise RuntimeError(
                f"gripper did not reach {args.target} before timeout; feedback={last_feedback}"
            )
    finally:
        if sdk is not None:
            # Latch exact measured feedback, then leave the controller in a
            # torque-holding quick-stop. Never disable torque here.
            raw = sdk.get_status()
            sdk.piper.JointCtrl(*(int(raw[f"joint_{index}.pos"]) for index in range(6)))
            sdk.piper.GripperCtrl(int(raw["joint_6.pos"]), 1000, 0x01, 0)
            sdk.piper.EmergencyStop(0x01)
            sdk.motion_enabled = False
            sdk.piper.DisconnectPort()


if __name__ == "__main__":
    main()
