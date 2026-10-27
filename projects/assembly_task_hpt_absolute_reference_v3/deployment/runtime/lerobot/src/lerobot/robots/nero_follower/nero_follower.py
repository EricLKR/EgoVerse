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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_nero_follower import NeroFollowerConfig

logger = logging.getLogger(__name__)

# NERO joint names (7 joints + 1 gripper)
NERO_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
NERO_MOTORS = [*NERO_JOINTS, "gripper"]


class NeroFollower(Robot):
    """AgileX/AGX NERO 7-DoF arm + AGX Gripper, driven over CAN via the `pyAgxArm` SDK.

    `pyAgxArm` is not published on PyPI; obtain it from the manufacturer and make sure it's
    importable (e.g. `pip install /path/to/pyAgxArm`) before using this robot.
    Joint angles are in radians; the gripper is normalized to [0, 100].
    """

    config_class = NeroFollowerConfig
    name = "nero_follower"

    def __init__(self, config: NeroFollowerConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._effector = None
        self._last_joint_targets: list[float] | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in NERO_MOTORS}

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        if self._arm is None:
            return False
        try:
            return self._arm.is_connected() and all(cam.is_connected for cam in self.cameras.values())
        except Exception:
            return False

    @property
    def is_calibrated(self) -> bool:
        # NERO has absolute encoders, no homing routine needed.
        return True

    def calibrate(self) -> None:
        logger.info(f"{self} uses absolute encoders, no calibration needed.")

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Imported lazily so `lerobot.robots` doesn't hard-require the (non-pip-installable) SDK.
        from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config

        arm_config = create_agx_arm_config(
            robot=ArmModel.NERO,
            comm="can",
            firmeware_version=self.config.firmeware_version,
            interface=self.config.interface,
            channel=self.config.channel,
            auto_set_motion_mode=self.config.auto_set_motion_mode,
            enable_joint_limits=self.config.enable_joint_limits,
        )
        self._arm = AgxArmFactory.create_arm(arm_config)
        self._arm.connect()

        logger.info(f"Enabling {self}...")
        while not self._arm.enable():
            time.sleep(0.01)

        self._effector = self._arm.init_effector(self._arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self._arm.set_speed_percent(self.config.speed_percent)

        # Lock JS (servo passthrough) mode for low-latency control; without this the SDK
        # auto-switches to J mode on every move_j call.
        self._arm.set_motion_mode(self._arm.OPTIONS.MOTION_MODE.JS)
        self._arm.set_auto_set_motion_mode_enabled(False)

        # Prime the target cache from live state so a partial action doesn't zero unspecified joints.
        self._last_joint_targets = self._read_joint_angles() or [0.0] * len(NERO_JOINTS)

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    def configure(self) -> None:
        self._arm.set_speed_percent(self.config.speed_percent)

    def _read_joint_angles(self) -> list[float] | None:
        joint_msg = self._arm.get_joint_angles()
        if joint_msg is None or joint_msg.msg is None:
            return None
        vals = [float(v) for v in list(joint_msg.msg)[: len(NERO_JOINTS)]]
        if len(vals) < len(NERO_JOINTS):
            vals += [0.0] * (len(NERO_JOINTS) - len(vals))
        return vals

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        joint_vals = self._read_joint_angles() or [0.0] * len(NERO_JOINTS)
        obs_dict = {f"{name}.pos": joint_vals[i] for i, name in enumerate(NERO_JOINTS)}

        obs_dict["gripper.pos"] = 0.0
        if self._effector is not None:
            try:
                gripper_msg = self._effector.get_gripper_status()
                if gripper_msg is not None and gripper_msg.msg is not None:
                    obs_dict["gripper.pos"] = float(gripper_msg.msg.value)
            except Exception as e:
                logger.debug(f"Failed to read gripper state: {e}")
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command the arm to a target joint configuration (`joint1.pos` ... `joint7.pos`, radians,
        plus `gripper.pos` in [0, 100]).

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            DeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the arm, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        joint_goals = {k: float(v) for k, v in goal_pos.items() if k in NERO_JOINTS}

        # Ignore actions with no joint/gripper content (e.g. stray keys) to avoid unintended zeroing.
        if not joint_goals and "gripper" not in goal_pos:
            return {}

        present_vals = self._read_joint_angles()
        base_targets = present_vals if present_vals is not None else list(
            self._last_joint_targets or [0.0] * len(NERO_JOINTS)
        )

        if self.config.max_relative_target is not None and present_vals is not None:
            present_dict = dict(zip(NERO_JOINTS, present_vals))
            goal_present = {k: (v, present_dict[k]) for k, v in joint_goals.items() if k in present_dict}
            if goal_present:
                joint_goals = ensure_safe_goal_position(goal_present, self.config.max_relative_target)

        for i, name in enumerate(NERO_JOINTS):
            if name in joint_goals:
                base_targets[i] = joint_goals[name]

        if joint_goals:
            self._arm.move_js(base_targets)
            self._last_joint_targets = list(base_targets)

        if "gripper" in goal_pos and self._effector is not None:
            try:
                self._effector.move_gripper_m(goal_pos["gripper"])
            except Exception as e:
                logger.warning(f"Failed to send gripper command: {e}")

        sent_action = {f"{motor}.pos": val for motor, val in joint_goals.items()}
        if "gripper" in goal_pos:
            sent_action["gripper.pos"] = float(goal_pos["gripper"])
        return sent_action

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            if self.config.disable_torque_on_disconnect:
                self._arm.disable()
        except Exception as e:
            logger.warning(f"Error disabling {self}: {e}")

        try:
            self._arm.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting {self}: {e}")

        self._arm = None
        self._effector = None
        self._last_joint_targets = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
