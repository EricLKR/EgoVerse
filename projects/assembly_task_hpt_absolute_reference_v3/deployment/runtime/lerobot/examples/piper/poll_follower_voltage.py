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

"""Continuously polls per-motor voltage/enable status via GetArmLowSpdInfoMsgs(), to check whether a
single early 0.0V reading was stale/uninitialized data vs. a real fault. No EnablePiper() call --
read-only, safe regardless of arm state.

Usage:
    python examples/piper/poll_follower_voltage.py --port=can_follower
"""

import argparse
import time

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError as e:
    raise ImportError("piper_sdk is not installed. Install it with `pip install lerobot[piper]`.") from e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="can_follower")
    parser.add_argument("--n", type=int, default=30, help="number of samples")
    parser.add_argument("--interval", type=float, default=0.3, help="seconds between samples")
    args = parser.parse_args()

    piper = C_PiperInterface_V2(args.port)
    piper.ConnectPort()
    time.sleep(0.5)

    print(f"Connected on {args.port}. Polling {args.n} samples every {args.interval}s (read-only).\n")

    for i in range(args.n):
        time.sleep(args.interval)
        info = piper.GetArmLowSpdInfoMsgs()
        motors = [info.motor_1, info.motor_2, info.motor_3, info.motor_4, info.motor_5, info.motor_6]
        parts = []
        for j, m in enumerate(motors, start=1):
            vol = m.vol * 0.1
            enabled = m.foc_status.driver_enable_status
            parts.append(f"m{j}={vol:4.1f}V/{'EN' if enabled else 'dis'}")
        print(f"{i:2d}: " + "  ".join(parts))

    print("\nDone.")


if __name__ == "__main__":
    main()
