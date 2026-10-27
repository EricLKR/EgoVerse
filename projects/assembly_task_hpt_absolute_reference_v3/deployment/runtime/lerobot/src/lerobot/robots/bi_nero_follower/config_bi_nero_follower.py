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


@RobotConfig.register_subclass("bi_nero_follower")
@dataclass
class BiNeroFollowerConfig(RobotConfig):
    # CAN channels, e.g. "can0" / "can1" if both arms are bridged to the same host.
    left_arm_channel: str
    right_arm_channel: str

    # Shared between both arms (mount two NEROs from the same interface family/firmware in practice)
    interface: str = "socketcan"
    firmeware_version: str = "default"
    auto_set_motion_mode: bool = True
    enable_joint_limits: bool = True
    speed_percent: int = 50

    # Optional
    left_arm_disable_torque_on_disconnect: bool = True
    left_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_disable_torque_on_disconnect: bool = True
    right_arm_max_relative_target: float | dict[str, float] | None = None

    # cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
