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

"""Connects to a PIPER follower and prints the gripper's raw reading in mm, continuously, so you can
open it fully by hand and read off the real max stroke -- to check against PiperConfig's
`gripper_stroke_mm` default (100mm for PIPER-L, 70mm for standard PIPER; see
piper/piper_sdk_interface.py for sourcing).

No motion is ever commanded by this script -- but `Piper.connect()` itself enables torque as part of
its normal connect sequence (same as any other robot connect), which holds the gripper rigid under
active control, NOT freely backdrivable. So right after connecting, this disables torque again
(matching how find_real_joint_limits.py does it) specifically so you *can* move the gripper by hand.

Usage:
    python examples/piper/read_gripper_stroke.py --port=can0
"""

import argparse
import time

from lerobot.robots.piper import Piper, PiperConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="can0", help="CAN channel the arm is on.")
    parser.add_argument("--hz", type=float, default=5.0)
    args = parser.parse_args()

    robot = Piper(PiperConfig(port=args.port, id="read_gripper_stroke"))
    robot.connect()

    if not robot.is_connected:
        raise RuntimeError("Failed to connect.")

    # Release torque so the gripper (and arm) are actually free to move by hand -- connect() alone
    # leaves them enabled/rigid.
    while robot.sdk.piper.DisablePiper():
        time.sleep(0.01)

    print(f"Connected on {args.port}, torque released. Open/close the gripper by hand and watch the max mm reached.")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            mm = robot.sdk.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000
            print(f"gripper opening: {mm:6.2f} mm")
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
