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

"""Prints the raw min/max joint angle limits (from `GetAllMotorAngleLimitMaxSpd`) that
`PiperSDKInterface` fetches on connect -- read-only, no motion commanded.

Checks for degenerate (min == max, or min > max) limits, which would make
`raw_deg001_to_norm` always return 0.0 for that joint regardless of its real position --
indistinguishable from "never moves" without looking at these raw numbers directly.

Usage:
    python examples/piper/check_joint_limits.py --port=can0
"""

import argparse

from lerobot.robots.piper import Piper, PiperConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="can0", help="CAN channel the arm is on.")
    args = parser.parse_args()

    robot = Piper(PiperConfig(port=args.port, home_on_disconnect=False, id="check_joint_limits"))
    robot.connect()

    if not robot.is_connected:
        raise RuntimeError("Failed to connect.")

    print(f"Connected on {args.port}.\n")
    print(f"{'joint':<10}{'min (0.1deg)':>15}{'max (0.1deg)':>15}{'range':>10}")
    for i in range(6):
        lo, hi = robot.sdk.min_pos[i], robot.sdk.max_pos[i]
        flag = "  <-- DEGENERATE" if hi <= lo else ""
        print(f"joint_{i:<4}{lo:>15.1f}{hi:>15.1f}{hi - lo:>10.1f}{flag}")

    robot.disconnect()
    print("\nDisconnected.")


if __name__ == "__main__":
    main()
