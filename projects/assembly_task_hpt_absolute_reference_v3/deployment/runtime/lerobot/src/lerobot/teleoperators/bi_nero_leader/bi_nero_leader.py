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
from functools import cached_property
from typing import Any

from lerobot.teleoperators.nero_leader.config_nero_leader import NeroLeaderConfig
from lerobot.teleoperators.nero_leader.nero_leader import NeroLeader

from ..teleoperator import Teleoperator
from .config_bi_nero_leader import BiNeroLeaderConfig

logger = logging.getLogger(__name__)


class BiNeroLeader(Teleoperator):
    """Bimanual AgileX/AGX NERO leader, composed of two independent [NeroLeader][lerobot.teleoperators.nero_leader.NeroLeader]
    arms addressed with `left_`/`right_` prefixes, mirroring how `BiSO100Leader` wraps two `SO100Leader`s.

    Mount the pair face-to-face for drag-teach and set `mirror_sign` on each underlying `NeroLeader`
    (via a custom config, if the defaults don't match your rig) to flip joints that oppose each other.
    """

    config_class = BiNeroLeaderConfig
    name = "bi_nero_leader"

    def __init__(self, config: BiNeroLeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = NeroLeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            can_interface=config.can_interface,
            firmware_version=config.firmware_version,
            enable_drag_teach=config.enable_drag_teach,
            reset_can_on_connect=config.reset_can_on_connect,
        )
        right_arm_config = NeroLeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            can_interface=config.can_interface,
            firmware_version=config.firmware_version,
            enable_drag_teach=config.enable_drag_teach,
            reset_can_on_connect=config.reset_can_on_connect,
        )

        self.left_arm = NeroLeader(left_arm_config)
        self.right_arm = NeroLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{key}": val for key, val in self.left_arm.action_features.items()} | {
            f"right_{key}": val for key, val in self.right_arm.action_features.items()
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def get_action(self) -> dict[str, Any]:
        action_dict = {}

        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
