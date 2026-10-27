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

"""PIPER-as-leader teleoperator: reads a second PIPER arm's commanded joint positions and reports
them normalized to the same [-100, 100] (joints) / [0, 100] (gripper) frame `Piper.send_action`
expects, so a `piper_leader` can drive a `piper` follower directly through `lerobot-teleoperate`.

Grounded in agilexrobotics/piper_sdk's own demos (`piper_sdk/demo/V2/`):
  - `piper_set_master.py`: `MasterSlaveConfig(0xFA, 0, 0, 0)` configures an arm as the
    "teaching input arm" (leader) role.
  - `piper_read_joint_ctrl.py`: `GetArmJointCtrl()` reads what the leader arm is currently
    commanding, which is what a human backdriving it produces -- per the upstream discussion in
    huggingface/lerobot#1335, this is the channel to poll, not the regular joint-state feedback.
No known LeRobot PR implements this; it's new code written for this integration.
"""

import logging
import time
from typing import Any

from lerobot.motors.motors_bus import MotorCalibration
from lerobot.robots.piper.piper_sdk_interface import raw_deg001_to_norm, raw_mm001_to_pct
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_piper_leader import PiperLeaderConfig

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    C_PiperInterface_V2 = None

logger = logging.getLogger(__name__)

_JOINT_NAMES = [f"joint_{i}" for i in range(6)]


class PiperLeader(Teleoperator):
    """A PIPER arm used as a leader, its joints backdriven by hand and read via `GetArmJointCtrl()`.

    Untested against real hardware -- verify `mirror_sign` and motion at low speed before trusting it.
    """

    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig):
        super().__init__(config)
        self.config = config
        self.piper = None
        self.min_pos: list[float] | None = None
        self.max_pos: list[float] | None = None
        self._warned_stale = False

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in [*_JOINT_NAMES, "joint_6"]}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.piper is not None

    def _joint_limits_deg_from_calibration(self) -> list[tuple[float, float]] | None:
        if not all(k in self.calibration for k in _JOINT_NAMES):
            return None
        return [
            (self.calibration[k].range_min / 1000.0, self.calibration[k].range_max / 1000.0)
            for k in _JOINT_NAMES
        ]

    def _gripper_stroke_mm_from_calibration(self) -> float | None:
        cal = self.calibration.get("gripper")
        return None if cal is None else cal.range_max / 1000.0

    @property
    def is_calibrated(self) -> bool:
        return all(k in self.calibration for k in _JOINT_NAMES) and "gripper" in self.calibration

    def calibrate(self) -> None:
        """Same purpose/format as Piper.calibrate() (see its docstring) -- populates and saves
        self.calibration, loaded automatically on future connects. This arm is already freely
        backdrivable (master/teaching mode), so unlike the follower's version, no torque
        disable/enable is needed here.
        """
        if self.calibration:
            user_input = input(
                f"Press ENTER to use the saved calibration for id {self.id}, or type 'c' and press "
                "ENTER to recalibrate: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using saved calibration for {self}.")
                return

        if self.config.joint_limits_deg is not None and self.calibration == {}:
            logger.info(f"Seeding calibration for {self} from config.joint_limits_deg (no hands-on step).")
            self.calibration = {
                k: MotorCalibration(
                    id=i, drive_mode=0, homing_offset=0,
                    range_min=int(lo * 1000), range_max=int(hi * 1000),
                )
                for i, (k, (lo, hi)) in enumerate(zip(_JOINT_NAMES, self.config.joint_limits_deg))
            }
            self.calibration["gripper"] = MotorCalibration(
                id=6, drive_mode=0, homing_offset=0,
                range_min=0, range_max=int(self.config.gripper_stroke_mm * 1000),
            )
            self._save_calibration()
            print(f"Calibration saved to {self.calibration_fpath}")
            return

        logger.info(f"\nRunning hands-on calibration of {self}")
        print(
            f"\n{self}: move each of the 6 joints, one at a time, slowly through its full range of "
            "motion (gently to the mechanical stop, not into it -- and watch for workspace "
            "obstructions, not just the joint's own limit)."
        )
        input("Press Enter to start recording (~15s to move all 6 joints)...")
        seen_min = [None] * 6
        seen_max = [None] * 6
        deadline = time.time() + 15.0
        while time.time() < deadline:
            joint_ctrl = self.piper.GetArmJointCtrl().joint_ctrl
            raw = [
                joint_ctrl.joint_1, joint_ctrl.joint_2, joint_ctrl.joint_3,
                joint_ctrl.joint_4, joint_ctrl.joint_5, joint_ctrl.joint_6,
            ]
            for i, v in enumerate(raw):
                seen_min[i] = v if seen_min[i] is None else min(seen_min[i], v)
                seen_max[i] = v if seen_max[i] is None else max(seen_max[i], v)
            time.sleep(0.05)
        for i, k in enumerate(_JOINT_NAMES):
            print(f"  {k}: {seen_min[i] / 1000:.1f} to {seen_max[i] / 1000:.1f} deg")

        print("\nNow open/close the gripper fully by hand.")
        input("Press Enter to start recording (~5s)...")
        gripper_max = 0
        deadline = time.time() + 5.0
        while time.time() < deadline:
            gripper_max = max(gripper_max, self.piper.GetArmGripperCtrl().gripper_ctrl.grippers_angle)
            time.sleep(0.05)
        print(f"  gripper: {gripper_max / 1000:.1f} mm")

        self.calibration = {
            k: MotorCalibration(id=i, drive_mode=0, homing_offset=0, range_min=seen_min[i], range_max=seen_max[i])
            for i, k in enumerate(_JOINT_NAMES)
        }
        self.calibration["gripper"] = MotorCalibration(
            id=6, drive_mode=0, homing_offset=0, range_min=0, range_max=gripper_max
        )
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

        joint_limits_deg = self._joint_limits_deg_from_calibration()
        for i in range(6):
            self.min_pos[i] = joint_limits_deg[i][0] * 10.0
            self.max_pos[i] = joint_limits_deg[i][1] * 10.0
        self.config.gripper_stroke_mm = self._gripper_stroke_mm_from_calibration()

    def _assert_master_role(self) -> None:
        """Send `MasterSlaveConfig(0xFA, ...)` and verify it actually took, retrying with a
        slave-then-master bounce if not.

        Needed because this role has been observed to get stuck in some other state -- e.g.
        leftover from Piper_sdk_ui's own master/slave toggle (that tool's git history has a real
        commit titled, in translation, "...fix master/slave conversion abnormality" -- commit
        fbbd061e) -- or drift there again later with nothing intentionally sent to cause it (see
        the matching defensive re-assertion in piper_sdk_interface.py's follower `connect()`).
        agilexrobotics/piper_sdk#35 is the same class of bug, still open upstream as of writing;
        its only documented workaround is exactly this bounce -- master, then slave, then reset.

        Can't use `GetArmStatus().Hz` to check whether the role took -- a master-configured arm
        never sends that message at all (CAN ID 0x2A1 is a *feedback* message; master arms only
        send control frames, 0x155/156/157/159/151, per agilexrobotics/piper_sdk's
        `double_piper.MD`). `GetArmJointCtrl()`'s own `Hz` is the channel this class actually
        reads, so that's the one to check.
        """
        self.piper.MasterSlaveConfig(0xFA, 0, 0, 0)
        time.sleep(0.3)
        if self.piper.GetArmJointCtrl().Hz > 0:
            return

        logger.warning(f"{self}: master role didn't take on first try, retrying via slave/master bounce.")
        self.piper.MasterSlaveConfig(0xFC, 0, 0, 0)
        time.sleep(0.2)
        self.piper.MasterSlaveConfig(0xFA, 0, 0, 0)
        time.sleep(0.3)
        if self.piper.GetArmJointCtrl().Hz == 0:
            logger.warning(
                f"{self}: still no control-frame traffic after the bounce retry -- {self} may need "
                "a power cycle, a physical connection check, or is on a dead/misconfigured CAN "
                "link. get_action() will read stale zeros until this clears."
            )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        if C_PiperInterface_V2 is None:
            raise ImportError("piper_sdk is not installed. Install it with `pip install lerobot[piper]`.")

        self.piper = C_PiperInterface_V2(self.config.port)
        self.piper.ConnectPort()
        time.sleep(0.1)

        if self.config.set_master_on_connect:
            self._assert_master_role()

        # Own limits, used to normalize this arm's readings the same way the follower normalizes
        # its own -- each physical unit may be homed/limited slightly differently. Precedence:
        # saved calibration (Robot.__init__ already loaded it into self.calibration) > explicit
        # config.joint_limits_deg > live GetAllMotorAngleLimitMaxSpd() query (can report wrong
        # generic defaults on this hardware -- see PiperLeaderConfig's docstring).
        joint_limits_deg = self._joint_limits_deg_from_calibration() or self.config.joint_limits_deg
        if joint_limits_deg is not None:
            if len(joint_limits_deg) != 6:
                raise ValueError(f"joint_limits_deg needs 6 (min, max) pairs, got {len(joint_limits_deg)}")
            self.min_pos = [lo * 10.0 for lo, _ in joint_limits_deg]
            self.max_pos = [hi * 10.0 for _, hi in joint_limits_deg]
        else:
            angle_status = self.piper.GetAllMotorAngleLimitMaxSpd()
            self.min_pos = [
                joint.min_angle_limit for joint in angle_status.all_motor_angle_limit_max_spd.motor[1:7]
            ]
            self.max_pos = [
                joint.max_angle_limit for joint in angle_status.all_motor_angle_limit_max_spd.motor[1:7]
            ]
        gripper_stroke_mm = self._gripper_stroke_mm_from_calibration()
        if gripper_stroke_mm is not None:
            self.config.gripper_stroke_mm = gripper_stroke_mm

        if not self.is_calibrated and calibrate:
            logger.info(
                f"No saved calibration for {self} (or config.joint_limits_deg was never set) -- "
                "running calibrate()."
            )
            self.calibrate()

        logger.info(f"{self} connected.")

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        joint_ctrl_msg = self.piper.GetArmJointCtrl()
        if joint_ctrl_msg.Hz == 0:
            if not self._warned_stale:
                logger.warning(
                    f"{self}: control-frame Hz dropped to 0 mid-session -- master role may have "
                    "drifted (see _assert_master_role's docstring); readings below are stale until "
                    "this clears. Won't repeat this warning until it recovers."
                )
                self._warned_stale = True
        else:
            self._warned_stale = False

        action = {}
        joint_ctrl = joint_ctrl_msg.joint_ctrl
        raw_joints = [
            joint_ctrl.joint_1,
            joint_ctrl.joint_2,
            joint_ctrl.joint_3,
            joint_ctrl.joint_4,
            joint_ctrl.joint_5,
            joint_ctrl.joint_6,
        ]
        for i, (name, raw) in enumerate(zip(_JOINT_NAMES, raw_joints)):
            norm = raw_deg001_to_norm(raw, self.min_pos[i], self.max_pos[i])
            action[f"{name}.pos"] = norm * self.config.mirror_sign[i]

        # GetArmGripperMsgs (feedback channel) reads as stale/zero on a master-configured arm, same
        # as joints -- GetArmGripperCtrl (control channel) is the one that's actually live here.
        gripper_ctrl = self.piper.GetArmGripperCtrl().gripper_ctrl
        action["joint_6.pos"] = raw_mm001_to_pct(gripper_ctrl.grippers_angle, self.config.gripper_stroke_mm)

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.piper = None
        self.min_pos = None
        self.max_pos = None
        logger.info(f"{self} disconnected.")
