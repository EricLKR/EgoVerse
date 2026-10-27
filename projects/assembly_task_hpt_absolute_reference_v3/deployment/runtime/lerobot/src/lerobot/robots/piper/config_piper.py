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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    # CAN interface the arm is bound to, e.g. "can0". Activate it beforehand with
    # `piper_sdk/can_activate.sh can0 1000000`.
    port: str = "can0"

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety
    # purposes. Set this to a positive scalar to have the same value for all joints, or a dictionary
    # that maps joint names to the max_relative_target value for that joint.
    max_relative_target: float | dict[str, float] | None = None

    # Firmware-side speed cap, 0-100 (passed to MotionCtrl_2's move_spd_rate_ctrl). Was previously
    # hardcoded to 100 with no way to reduce it -- turn this down (e.g. 10-20) for initial bring-up
    # / hardware verification, where you want the arm moving slowly enough to react if something's
    # wrong (inverted joint, bad mirror_sign, etc.) before it does.
    speed_percent: int = 100

    # False creates an observation-only CAN connection: feedback and cameras
    # are available, but the SDK wrapper must not call EmergencyStop(resume),
    # EnablePiper, MotionCtrl_2, JointCtrl, or GripperCtrl. This is the mode
    # used for live deployment dry-runs before any real motion is authorized.
    enable_motion_on_connect: bool = True

    # Gripper full-open stroke in mm. Not discoverable via the SDK (unlike joint limits). Default
    # (100mm) is from AgileX's PIPER-L datasheet ("Opening Range: 0-100mm"); standard PIPER is
    # 70mm -- pass that instead if that's your hardware. Verify against your specific unit if
    # precise gripper control matters.
    gripper_stroke_mm: float = 100.0

    # PIPER GripperCtrl effort argument, in 0.001 N*m SDK units. The firmware
    # accepts 0..5000; 5000 therefore requests the maximum 5.0 N*m motor
    # torque. This is torque, not a directly calibrated fingertip force in N.
    gripper_effort: int = 5000

    # On disconnect, gently move all 6 joints to their normalized midpoint (gripper left alone)
    # before cutting torque. Defaults OFF after a real-hardware test: cutting torque (DisablePiper)
    # can still make the arm drop, abruptly, from *any* pose -- including a freshly-homed one --
    # if PIPER-L has no passive brakes at the loaded joints and the home move hadn't fully settled
    # (this used a fixed 2s sleep, not a "did it actually arrive" check) or simply isn't a
    # gravity-neutral pose. Don't enable this unless the arm is resting on something that can
    # physically hold its weight (a stand, the table, a hand) regardless of what pose it ends up in
    # -- software cannot make disabling torque safe on its own on this hardware.
    home_on_disconnect: bool = False

    # If false, close the CAN connection without sending DisablePiper or a
    # gripper command. This avoids an automatic torque cut/drop at the end of
    # a guarded rollout. It intentionally leaves the arm enabled; a human must
    # then use the approved physical shutdown procedure. Observation-only
    # connections always take the no-command disconnect path regardless.
    disable_torque_on_disconnect: bool = True

    # Optional override for per-joint (min, max) limits in real degrees, 6 pairs, joint_0..joint_5.
    # None (default) trusts the arm's own GetAllMotorAngleLimitMaxSpd() live query. Set this if that
    # query reports wrong values for your unit -- found on a real PIPER-L to report generic,
    # uncalibrated defaults (symmetric +-170deg for every joint) instead of each joint's real,
    # very different range. Confirm real values by hand first: see
    # examples/piper/find_real_joint_limits.py (torque-disabled, safe, no motion commanded) --
    # don't guess these from a datasheet alone, per-unit manufacturing variance is real. Once
    # confirmed, trim a few degrees in from each discovered hard-stop as margin rather than using
    # the exact number, so normal operation never commands all the way into the physical stop.
    joint_limits_deg: list[tuple[float, float]] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
