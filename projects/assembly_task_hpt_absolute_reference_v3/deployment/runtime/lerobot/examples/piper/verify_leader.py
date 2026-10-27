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

"""Read-only sanity check for `piper_leader`, no follower involved -- verifies that
`GetArmJointCtrl()` returns live, sane data while you backdrive the arm by hand, and lets you read
off each joint's sign before trusting `PiperLeaderConfig.mirror_sign`.

No motion is ever commanded here; this only reads. Safe to run with the arm free to move by hand.

Usage:
    python examples/piper/verify_leader.py --port=can0
"""

import argparse
import time

from lerobot.teleoperators.piper_leader import PiperLeader, PiperLeaderConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="can0", help="CAN channel the leader arm is on.")
    parser.add_argument("--hz", type=float, default=5.0)
    args = parser.parse_args()

    leader = PiperLeader(PiperLeaderConfig(port=args.port, id="verify_leader"))
    leader.connect()

    if not leader.is_connected:
        raise RuntimeError("Leader failed to connect.")

    print(f"Connected on {args.port}. Move each joint by hand, one at a time, and watch its sign.")
    print("Expected: moving a joint one physical direction should read a *consistently* growing or")
    print("shrinking number -- not jump around or stay flat. Ctrl+C to stop.\n")

    try:
        while True:
            action = leader.get_action()
            line = " | ".join(f"{k}={v:+7.2f}" for k, v in sorted(action.items()))
            print(line)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        leader.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
