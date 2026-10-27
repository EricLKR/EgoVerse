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

"""Same purpose as read_gripper_stroke.py, but connects with raw piper_sdk only -- no
EnablePiper() call at all -- so it works against a master-configured (leader/teaching-mode) arm,
where going through our Piper (follower-style) class's normal connect() sequence can hang. Reading
gripper feedback doesn't need the arm enabled either way.

Same master/slave channel split as find_real_joint_limits.py applies to the gripper too: a
master-configured arm only broadcasts GetArmGripperCtrl (CAN 0x159), not the normal
GetArmGripperMsgs feedback (CAN 0x2A8) -- reading the wrong one gives a flat 0 regardless of the
gripper's real position, even while it moves freely by hand. Use --source=ctrl for a leader,
--source=msgs (default) for a normal follower.

Usage:
    python examples/piper/read_gripper_stroke_raw.py --port=can_leader --source=ctrl
"""

import argparse
import time

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError as e:
    raise ImportError("piper_sdk is not installed. Install it with `pip install lerobot[piper]`.") from e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="can0", help="CAN channel the arm is on.")
    parser.add_argument(
        "--source",
        choices=["msgs", "ctrl"],
        default="msgs",
        help="'msgs' for a normal/slave arm, 'ctrl' for a master-configured leader.",
    )
    parser.add_argument("--hz", type=float, default=5.0)
    args = parser.parse_args()

    piper = C_PiperInterface_V2(args.port)
    piper.ConnectPort()
    time.sleep(0.5)

    print(f"Connected on {args.port} (not enabled -- read-only). Open/close the gripper by hand.")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            if args.source == "ctrl":
                mm = piper.GetArmGripperCtrl().gripper_ctrl.grippers_angle / 1000
            else:
                mm = piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000
            print(f"gripper opening: {mm:6.2f} mm")
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
