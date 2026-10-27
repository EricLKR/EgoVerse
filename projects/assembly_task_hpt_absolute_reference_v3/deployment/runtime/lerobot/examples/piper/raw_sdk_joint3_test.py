#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Isolation test: exercises the same nudge-and-check pattern as verify_follower.py, but using
*only* raw `piper_sdk` calls -- no lerobot code (no Piper/PiperSDKInterface/_INVERTED_JOINTS/
normalization) is involved anywhere. Exists to determine whether joint4's (AgileX numbering; our
joint_3) direction-dependent misbehavior is something in our lerobot integration, or something in
piper_sdk / the arm's own firmware -- if it reproduces here, it's conclusively not our code.

Only tests joint4 (motor_4 / AgileX's own numbering), by design -- that's the joint in question.

Usage:
    python examples/piper/raw_sdk_joint3_test.py --port=can0 --speed-percent=15
Press Enter to nudge, watch the arm, Ctrl+C anytime.
"""

import argparse
import time

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError as e:
    raise ImportError("piper_sdk is not installed. Install it with `pip install lerobot[piper]`.") from e

# Real, hand-confirmed limits for joint4 on this specific unit (see find_real_joint_limits.py),
# in raw 0.001deg units (what JointCtrl/GetArmJointMsgs use directly).
JOINT4_MIN_RAW = -124_200
JOINT4_MAX_RAW = 124_900


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="can0")
    parser.add_argument("--speed-percent", type=int, default=15)
    parser.add_argument("--nudge-deg", type=float, default=20.0, help="Test displacement, in real degrees.")
    parser.add_argument("--hold-s", type=float, default=3.0)
    args = parser.parse_args()
    nudge_raw = int(args.nudge_deg * 1000)

    print(f"Connecting on {args.port}...", flush=True)
    piper = C_PiperInterface_V2(args.port)
    piper.ConnectPort()
    time.sleep(0.1)
    print("Connected. Enabling (this is where it silently hung before if the arm won't enable)...", flush=True)

    deadline = time.time() + 10.0
    while not piper.EnablePiper():
        if time.time() > deadline:
            raise RuntimeError(
                "EnablePiper() did not succeed within 10s -- check GetArmStatus()/GetArmLowSpdInfoMsgs() "
                "for a fault, check the physical CAN connection to the arm, or power-cycle it."
            )
        time.sleep(0.01)
    piper.MotionCtrl_2(0x01, 0x01, args.speed_percent, 0x00)

    print(f"Connected on {args.port}, enabled, speed_percent={args.speed_percent}.")
    print(f"Will nudge joint4 by {args.nudge_deg:+.1f} real degrees, hold {args.hold_s}s, return.\n")

    try:
        while True:
            input("Press Enter to test joint4 (Ctrl+C to stop) ...")

            joint_state = piper.GetArmJointMsgs().joint_state
            before_raw = joint_state.joint_4
            print(f"  before: {before_raw / 1000.0:+.2f} deg")

            target_raw = before_raw + nudge_raw
            target_raw = max(JOINT4_MIN_RAW, min(JOINT4_MAX_RAW, target_raw))
            # Hold the other 5 joints at their current position; only joint4 moves.
            j1 = piper.GetArmJointMsgs().joint_state
            piper.JointCtrl(j1.joint_1, j1.joint_2, j1.joint_3, target_raw, j1.joint_5, j1.joint_6)
            time.sleep(args.hold_s)

            after_raw = piper.GetArmJointMsgs().joint_state.joint_4
            print(f"  after:  {after_raw / 1000.0:+.2f} deg (expected ~{target_raw / 1000.0:+.2f} deg)")

            low_spd = piper.GetArmLowSpdInfoMsgs().motor_4
            print(
                f"  motor_4 driver: {low_spd.vol * 0.1:.1f}V, motor {low_spd.motor_temp}C, "
                f"stall={low_spd.foc_status.stall_status}, collision={low_spd.foc_status.collision_status}, "
                f"error={low_spd.foc_status.driver_error_status}, enabled={low_spd.foc_status.driver_enable_status}"
            )

            # Return joint4 only, hold everything else.
            j2 = piper.GetArmJointMsgs().joint_state
            piper.JointCtrl(j2.joint_1, j2.joint_2, j2.joint_3, before_raw, j2.joint_5, j2.joint_6)
            time.sleep(args.hold_s)
            print(f"  returned joint4 to {before_raw / 1000.0:+.2f} deg\n")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        while piper.DisablePiper():
            time.sleep(0.01)
        print("Disabled (torque released, no homing move -- same caution as always applies).")


if __name__ == "__main__":
    main()
