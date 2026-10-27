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

"""Low-speed, one-joint-at-a-time motion check for `piper` (the follower), no leader involved.

For each joint in turn: waits for you to press Enter, nudges *that joint only* by a small amount
from wherever it currently is, holds briefly, then returns it -- so you can confirm each joint
moves the expected direction/amount (i.e. sanity-check `_INVERTED_JOINTS` in
piper_sdk_interface.py) before ever pairing this with a leader.

Runs at a configurable speed cap (default 15%) -- see PiperConfig.speed_percent, which didn't
exist before this script surfaced the need for it. Be ready to physically stop the arm; this still
commands real motion.

Usage:
    python examples/piper/verify_follower.py --port=can0
"""

import argparse
import json
import time

from lerobot.robots.piper import Piper, PiperConfig

_JOINT_NAMES = [f"joint_{i}" for i in range(7)]  # joint_0..5 = arm, joint_6 = gripper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="can0", help="CAN channel the follower arm is on.")
    parser.add_argument("--speed-percent", type=int, default=15, help="Firmware speed cap, 0-100.")
    parser.add_argument("--nudge", type=float, default=8.0, help="Test displacement, in [-100,100] units.")
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument(
        "--joint-limits-deg",
        default=None,
        help="Override for PiperConfig.joint_limits_deg -- 6 [min,max] pairs as a JSON string, e.g. "
        "'[[-62.0,151.9],[0.9,186.0],[-170.1,-1.0],[-124.2,124.9],[-85.0,87.7],[-168.4,168.4]]'. "
        "Leave unset to trust the arm's own live-queried limits (see find_real_joint_limits.py if "
        "those turn out to be wrong for your unit).",
    )
    args = parser.parse_args()

    joint_limits_deg = None
    if args.joint_limits_deg is not None:
        parsed = json.loads(args.joint_limits_deg)
        joint_limits_deg = [tuple(pair) for pair in parsed]

    # home_on_disconnect off, deliberately: this script exists to find joints that might be
    # stalling/faulting, so auto-commanding every joint to move on every disconnect (the library's
    # normal default) is exactly the wrong behavior while that's still an open question. Once a
    # joint's been cleared, PiperConfig's default (home_on_disconnect=True) is fine for normal use.
    robot = Piper(
        PiperConfig(
            port=args.port,
            speed_percent=args.speed_percent,
            home_on_disconnect=False,
            joint_limits_deg=joint_limits_deg,
            id="verify_follower",
        )
    )
    robot.connect()

    if not robot.is_connected:
        raise RuntimeError("Follower failed to connect.")

    print(f"Connected on {args.port} at speed_percent={args.speed_percent}.")
    print(f"Each joint will nudge by {args.nudge:+.1f} (of [-100,100]) then return. Watch closely.\n")

    try:
        for name in _JOINT_NAMES:
            input(f"Press Enter to test {name} (Ctrl+C to stop here) ...")

            present = {
                k.removesuffix(".pos"): v for k, v in robot.sdk.get_status_normalized().items()
            }
            print(f"  before: {name}={present[name]:+.2f}")

            nudged = dict(present)
            nudged[name] = max(-100.0, min(100.0, present[name] + args.nudge))
            robot.send_action({f"{k}.pos": v for k, v in nudged.items()})
            time.sleep(args.hold_s)

            after = {k.removesuffix(".pos"): v for k, v in robot.sdk.get_status_normalized().items()}
            print(f"  after:  {name}={after[name]:+.2f} (expected ~{nudged[name]:+.2f})")

            tripped = {}
            if name != "joint_6":  # gripper has no per-motor fault feedback like the 6 arm joints
                idx = int(name.removeprefix("joint_"))
                faults = robot.sdk.get_motor_faults()[idx]
                # `driver_enabled` is True when healthy -- opposite polarity from every other flag
                # here, where True means bad. Keep it out of the "bad if True" fault check, and
                # check it separately (bad if *False*).
                bad_if_true = {k: v for k, v in faults.items() if k != "driver_enabled" and v is True}
                tripped = dict(bad_if_true)
                if not faults["driver_enabled"]:
                    tripped["driver_disabled"] = True
                if tripped:
                    print(f"  !! {name} driver fault flags set: {tripped}")
                print(
                    f"  {name} driver: {faults['voltage_v']:.1f}V, "
                    f"motor {faults['motor_temp_c']}C, driver {faults['driver_temp_c']}C, "
                    f"enabled={faults['driver_enabled']}"
                )

            if tripped:
                # Don't auto-return and don't move to the next joint: if the firmware's own
                # protection tripped, sending it *another* motion command (even "return to start")
                # without knowing why is exactly the wrong instinct. Stop here, inspect physically,
                # decide what's actually going on before commanding this joint again.
                print(
                    f"  Stopping here -- {name} tripped a fault flag. Inspect the arm before "
                    "commanding it again (this script will NOT auto-return it or continue)."
                )
                break

            robot.send_action({f"{k}.pos": v for k, v in present.items()})
            time.sleep(args.hold_s)
            print(f"  returned {name} to {present[name]:+.2f}\n")
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        robot.disconnect()
        print("Disconnected (torque released, no auto-homing move -- see note above).")


if __name__ == "__main__":
    main()
