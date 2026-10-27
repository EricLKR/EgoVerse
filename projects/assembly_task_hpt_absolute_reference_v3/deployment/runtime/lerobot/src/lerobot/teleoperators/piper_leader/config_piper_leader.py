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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PiperLeaderConfig(TeleoperatorConfig):
    # CAN channel this leader arm is bound to, e.g. "can1" if the follower is on "can0".
    port: str = "can0"

    # Sends `MasterSlaveConfig(0xFA, 0, 0, 0)` on connect to configure this arm as PIPER's native
    # "teaching input arm" role (see agilexrobotics/piper_sdk's `piper_set_master.py` demo). Turn
    # this off if you've already configured master/slave roles yourself (e.g. via AgileX's own
    # assistant tool) and don't want this to re-issue the command on every connect.
    set_master_on_connect: bool = True

    # Gripper full-open stroke in mm -- this is THIS arm's own stroke, not the follower's. A
    # leader/follower pair is usually the same model, but measure both independently (see
    # examples/piper/read_gripper_stroke.py) rather than assuming they match -- the follower's own
    # measured stroke turned out ~3mm off from the datasheet default, so don't assume identical
    # units are identical without checking. Default (100mm) is from AgileX's PIPER-L datasheet.
    gripper_stroke_mm: float = 100.0

    # Per-joint sign flip applied to raw readings before reporting them as actions. Two PIPER units
    # in a leader/follower pair are usually mounted so their joints move in the same physical
    # direction, so this defaults to no inversion -- unlike the follower's driver, which flips
    # joints 1/4/6 to match how *that* arm's motors are wired. Verify at low speed before trusting
    # this; get it wrong and the follower will move opposite to what you expect.
    mirror_sign: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])

    # Optional override for THIS arm's per-joint (min, max) limits in real degrees, 6 pairs,
    # joint_0..joint_5. None (default) trusts this arm's own live GetAllMotorAngleLimitMaxSpd()
    # query. Set this if that query reports wrong values for your unit -- found on the follower to
    # report generic, uncalibrated defaults (symmetric +-170deg for every joint) instead of each
    # joint's real range. The leader is a separate physical unit from the follower and may have the
    # same problem independently -- measure it separately with find_real_joint_limits.py, don't
    # assume the follower's numbers apply here. A mismatch between leader and follower limits means
    # the same normalized value maps to different real angles on each arm, which shows up as exactly
    # the kind of "slight motion mismatch during teleoperation" this field exists to fix.
    joint_limits_deg: list[tuple[float, float]] | None = None
