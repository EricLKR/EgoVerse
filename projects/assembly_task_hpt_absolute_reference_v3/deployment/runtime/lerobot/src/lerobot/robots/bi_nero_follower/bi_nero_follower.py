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
from lerobot.robots.nero_follower import NeroFollower
from lerobot.robots.nero_follower.config_nero_follower import NeroFollowerConfig

from ..robot import Robot
from .config_bi_nero_follower import BiNeroFollowerConfig

logger = logging.getLogger(__name__)


class BiNeroFollower(Robot):
    """Bimanual AgileX/AGX NERO follower, composed of two independent [NeroFollower][lerobot.robots.nero_follower.NeroFollower]
    arms addressed with `left_`/`right_` prefixes, mirroring how `BiSO100Follower` wraps two `SO100Follower`s.
    """

    config_class = BiNeroFollowerConfig
    name = "bi_nero_follower"

    def __init__(self, config: BiNeroFollowerConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = NeroFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            channel=config.left_arm_channel,
            interface=config.interface,
            firmeware_version=config.firmeware_version,
            auto_set_motion_mode=config.auto_set_motion_mode,
            enable_joint_limits=config.enable_joint_limits,
            speed_percent=config.speed_percent,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            cameras={},
        )
        right_arm_config = NeroFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            channel=config.right_arm_channel,
            interface=config.interface,
            firmeware_version=config.firmeware_version,
            auto_set_motion_mode=config.auto_set_motion_mode,
            enable_joint_limits=config.enable_joint_limits,
            speed_percent=config.speed_percent,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            cameras={},
        )

        self.left_arm = NeroFollower(left_arm_config)
        self.right_arm = NeroFollower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{key}": val for key, val in self.left_arm._motors_ft.items()} | {
            f"right_{key}": val for key, val in self.right_arm._motors_ft.items()
        }

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
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)

        return {f"left_{key}": value for key, value in sent_left.items()} | {
            f"right_{key}": value for key, value in sent_right.items()
        }

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()
