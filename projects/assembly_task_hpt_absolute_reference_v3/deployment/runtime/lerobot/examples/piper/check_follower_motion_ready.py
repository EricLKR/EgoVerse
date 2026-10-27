#!/usr/bin/env python

"""Resume/enable a Piper without commanding displacement, verify readiness, then quick-stop.

This is deliberately separate from policy deployment.  It uses the same production
``PiperSDKInterface`` startup handshake, latches the measured pose as the only joint target,
prints controller/driver state, and returns the firmware to quick-stop.  It never disables
torque and never commands a pose different from measured feedback.
"""

import argparse
import json

from lerobot.robots.piper.piper_sdk_interface import PiperSDKInterface


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="can_follower")
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument(
        "--allow-resume",
        action="store_true",
        help="Required acknowledgement: resume and enable the controller, but command no displacement.",
    )
    args = parser.parse_args()
    if not args.allow_resume:
        parser.error("--allow-resume is required")
    if not 0 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [0, 100]")

    sdk = None
    try:
        sdk = PiperSDKInterface(
            port=args.port,
            speed_percent=args.speed_percent,
            enable_motion=True,
        )
        status = sdk.piper.GetArmStatus().arm_status
        normalized = sdk.get_status_normalized()
        faults = sdk.get_motor_faults()
        bad_faults = []
        for index, fault in enumerate(faults):
            bad = {
                key: value
                for key, value in fault.items()
                if (key == "driver_enabled" and value is False)
                or (key != "driver_enabled" and isinstance(value, bool) and value)
            }
            if bad:
                bad_faults.append({"joint": index, **bad})
        report = {
            "port": args.port,
            "arm_status": repr(status.arm_status),
            "ctrl_mode": repr(status.ctrl_mode),
            "motion_status": repr(status.motion_status),
            "normalized_state": normalized,
            "motor_faults": faults,
            "bad_faults": bad_faults,
            "ready": status.arm_status == 0 and not bad_faults,
            "motion_command_sent": False,
        }
        print(json.dumps(report, indent=2))
        if not report["ready"]:
            raise RuntimeError("Piper did not pass the motion-ready status/fault checks")
        print("READY CHECK PASSED: controller resumed; measured-pose hold only; no displacement commanded.")
    finally:
        if sdk is not None:
            # Prime the exact measured raw pose, then return to torque-holding
            # quick-stop. Never call DisablePiper/ResetPiper here.
            raw = sdk.get_status()
            sdk.piper.JointCtrl(*(int(raw[f"joint_{i}.pos"]) for i in range(6)))
            sdk.piper.GripperCtrl(int(raw["joint_6.pos"]), 1000, 0x01, 0)
            sdk.piper.EmergencyStop(0x01)
            sdk.motion_enabled = False
            sdk.piper.DisconnectPort()
            print("Returned controller to firmware quick-stop; torque was not disabled.")


if __name__ == "__main__":
    main()
