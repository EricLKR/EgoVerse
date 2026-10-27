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

"""Empirically discovers each joint's REAL mechanical range by hand, safely -- torque is disabled
immediately on connect (no motor force involved at any point), and this just logs the min/max raw
angle seen per joint while you manually walk each one through its full range of motion.

Exists because `GetAllMotorAngleLimitMaxSpd()` was found to report generic/uncalibrated defaults
(symmetric +-170 deg for every joint) on this unit rather than real per-joint limits -- see
piper_set_init_default.py in agilexrobotics/piper_sdk for what those defaults are. This script
doesn't touch that firmware-side calibration at all; it just tells you, empirically and safely,
what the real numbers probably are, so you have something trustworthy before considering writing
new calibration values (a persistent flash write per MotorAngleLimitMaxSpdSet's own docstring --
not something to do without being sure).

A PIPER arm configured as master (e.g. a piper_leader that's had set_master_on_connect fire, which
is persistent until explicitly changed) only broadcasts "control frame" messages (GetArmJointCtrl,
CAN IDs 0x155-157) -- it does NOT send the normal feedback messages (GetArmJointMsgs, CAN IDs
0x2A5-2A7) at all (see agilexrobotics/piper_sdk's double_piper.MD). Reading the wrong channel for
a master-configured arm gives all-zero readings, indistinguishable at a glance from "not moving."
Use --source=ctrl for a leader/master-configured arm, --source=msgs (default) for a normal
follower/slave-configured one.

Usage:
    python examples/piper/find_real_joint_limits.py --port=can0
    python examples/piper/find_real_joint_limits.py --port=can1 --source=ctrl   # for a leader
Move each joint slowly, one at a time, through its complete range (gently -- to the mechanical
stop, not into it). Ctrl+C when done with all 6 to see the summary.
"""

import argparse
import time

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError as e:
    raise ImportError("piper_sdk is not installed. Install it with `pip install lerobot[piper]`.") from e

_JOINT_NAMES = [f"joint_{i}" for i in range(6)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="can0", help="CAN channel the arm is on.")
    parser.add_argument(
        "--source",
        choices=["msgs", "ctrl"],
        default="msgs",
        help="'msgs' (GetArmJointMsgs) for a normal/slave arm, 'ctrl' (GetArmJointCtrl) for a "
        "master-configured leader -- see module docstring.",
    )
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    piper = C_PiperInterface_V2(args.port)
    piper.ConnectPort()
    time.sleep(0.1)

    # Disable torque immediately -- this script never commands motion, only reads, so the arm
    # should be freely backdrivable by hand for the whole run.
    while piper.DisablePiper():
        time.sleep(0.01)

    print(f"Connected on {args.port}, torque disabled -- arm should move freely by hand.")
    print("Move each joint slowly through its full range, one at a time. Ctrl+C when done.\n")

    seen_min = [None] * 6
    seen_max = [None] * 6

    try:
        while True:
            if args.source == "ctrl":
                joint_state = piper.GetArmJointCtrl().joint_ctrl
            else:
                joint_state = piper.GetArmJointMsgs().joint_state
            raw = [
                joint_state.joint_1,
                joint_state.joint_2,
                joint_state.joint_3,
                joint_state.joint_4,
                joint_state.joint_5,
                joint_state.joint_6,
            ]
            line_parts = []
            for i, r in enumerate(raw):
                deg = r / 1000.0  # raw is 0.001deg
                seen_min[i] = deg if seen_min[i] is None else min(seen_min[i], deg)
                seen_max[i] = deg if seen_max[i] is None else max(seen_max[i], deg)
                line_parts.append(f"{_JOINT_NAMES[i]}={deg:+7.2f}")
            print(" | ".join(line_parts), end="\r")
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass

    print("\n\nDiscovered ranges (move each joint fully to trust its number):")
    print(f"{'joint':<10}{'min (deg)':>12}{'max (deg)':>12}")
    for i in range(6):
        lo = f"{seen_min[i]:.1f}" if seen_min[i] is not None else "?"
        hi = f"{seen_max[i]:.1f}" if seen_max[i] is not None else "?"
        print(f"{_JOINT_NAMES[i]:<10}{lo:>12}{hi:>12}")


if __name__ == "__main__":
    main()
