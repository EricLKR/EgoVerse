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


@TeleoperatorConfig.register_subclass("nero_leader")
@dataclass
class NeroLeaderConfig(TeleoperatorConfig):
    # CAN channel (e.g. "can0" on Linux)
    port: str = "can0"

    # Communication interface ("socketcan" on Linux, "agx_cando" on Windows, "slcan" on macOS)
    can_interface: str = "socketcan"

    # Firmware version: "DEFAULT" (<=1.10) or "V111" (>=1.11)
    firmware_version: str = "V111"

    # Drag-teach (gravity-compensated, zero-force) mode lets a human backdrive the leader arm.
    enable_drag_teach: bool = True

    # Reset (down/up) the CAN interface on connect; useful after a previous ungraceful disconnect.
    reset_can_on_connect: bool = True
    enable_retries: int = 50
    enable_retry_interval: float = 0.05

    # Joint names, in the order pyAgxArm reports/expects them for a 7-DoF NERO arm.
    joint_names: list[str] = field(default_factory=lambda: [f"joint{i}" for i in range(1, 8)])

    # Per-joint sign flip, useful when mounting a pair of leader arms face-to-face for bimanual
    # teleop (mirrors the corresponding follower's joint direction). 1 = keep, -1 = flip.
    mirror_sign: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1, 1])
