#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_widowx import WidowXConfig

logger = logging.getLogger(__name__)


class WidowX(Teleoperator):
    """
    [WidowX](https://www.trossenrobotics.com/widowx-250) developed by Trossen Robotics
    """

    config_class = WidowXConfig
    name = "widowx"

    def __init__(self, config: WidowXConfig):
        # raise NotImplementedError
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "waist": Motor(1, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "shoulder": Motor(2, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "shoulder_shadow": Motor(3, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "elbow": Motor(4, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "elbow_shadow": Motor(5, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "forearm_roll": Motor(6, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "wrist_angle": Motor(7, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "wrist_rotate": Motor(8, "xl430-w250", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(9, "xl430-w250", MotorNormMode.RANGE_0_100),
            },
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()

        # ⇩⇩⇩ ADD THIS BLOCK ⇩⇩⇩
        if getattr(self, "calibration", None):
            # sanity check: keys must match motor names
            expected = set(self.bus.motors.keys())
            have = set(self.calibration.keys())
            missing, extra = expected - have, have - expected
            if missing or extra:
                raise RuntimeError(f"Calibration keys mismatch. missing={missing}, extra={extra}")
            self.bus.write_calibration(self.calibration)
            # Some implementations gate on a flag; if not set by write_calibration, mark it explicitly
            # (No-op if your bus already updates this)
            try:
                self.bus._is_calibrated = True  # best-effort; ignore if attribute doesn't exist
            except Exception:
                pass

            # IMPORTANT: skip calling self.calibrate() if we already have a calibration
            calibrate = False
        # ⇧⇧⇧ ADD THIS BLOCK ⇧⇧⇧

        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated


    def calibrate(self) -> None:
        """
+        Interactive calibration for WidowX (Dynamixel):
+        - puts all joints in EXTENDED_POSITION
+        - computes homing offsets around the current mid pose (half-turn rounding)
+        - records motion ranges (except full-turn joints which are fixed to 0..4095)
+        - sets drive_mode=0 for all by default (safe; invert later if needed)
+        - writes + saves calibration
+        """
        logger.info(f"\nRunning calibration of {self}")
        # 1) Ensure torque is disabled and set operating mode
        self.bus.disable_torque()
        for motor_name in self.bus.motors:
            self.bus.write("Operating_Mode", motor_name, OperatingMode.EXTENDED_POSITION.value)

        # # 2) (Optional) set any known inverted joints here.
        # # If you KNOW a joint is inverted on your build, set to DriveMode.INVERTED (1).
        # # Otherwise keep everything non-inverted (0) and rely on later adjustments if needed.
        motor_inverted = ["shoulder_shadow", "elbow_shadow", "wrist_angle"]
        drive_modes = {motor: 1 if motor in motor_inverted else 0 for motor in self.bus.motors}
        # # Example: if your elbow needs inversion in your hardware, uncomment:
        # # drive_modes["elbow"] = DriveMode.INVERTED.value
        # for name, dm in drive_modes.items():
        #     self.bus.write("Drive_Mode", name, dm)

        # drive_modes = {name: 0 for name in self.bus.motors}
        for name, dm in drive_modes.items():
            # Safe to write even if not used by firmware; harmless if ignored.
            try:
                self.bus.write("Drive_Mode", name, dm)
            except Exception:
                print(f"Warning: could not set Drive_Mode for {name}; continuing...")
                pass

        # 3) Ask user to place arm at mid-range pose; compute homing offsets
        input("Move arm to a mid-range 'neutral' pose (all joints roughly centered) and press ENTER...")
        homing_offsets = self.bus.set_half_turn_homings()

        # 4) Record ranges of motion (skip full-turn joints that wrap)
        # Map your naming to the 'full turn' group:
        full_turn_motors = ["waist", "forearm_roll", "wrist_rotate"]  # pan and roll
        unknown_range_motors = [m for m in self.bus.motors if m not in full_turn_motors]
        print(
            f"Now slowly sweep EACH of these joints through their safe ranges while data is recorded:\n"
            f"{unknown_range_motors}\n"
            "Press ENTER to stop recording..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        # Set full-turn joints explicitly to full encoder span
        for m in full_turn_motors:
            range_mins[m] = 0
            range_maxes[m] = 4095

        # 5) Build calibration dict and persist
        self.calibration = {}
        for motor_name, m in self.bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=m.id,
                drive_mode=drive_modes[motor_name],
                homing_offset=homing_offsets[motor_name],
                range_min=range_mins[motor_name],
                range_max=range_maxes[motor_name],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    # def calibrate(self) -> None:
    #     raise NotImplementedError  # TODO(aliberts): adapt code below (copied from koch)
    #     logger.info(f"\nRunning calibration of {self}")
    #     self.bus.disable_torque()
    #     for motor in self.bus.motors:
    #         self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

    #     self.bus.write("Drive_Mode", "elbow_flex", DriveMode.INVERTED.value)
    #     drive_modes = {motor: 1 if motor == "elbow_flex" else 0 for motor in self.bus.motors}

    #     input("Move robot to the middle of its range of motion and press ENTER....")
    #     homing_offsets = self.bus.set_half_turn_homings()

    #     full_turn_motors = ["shoulder_pan", "wrist_roll"]
    #     unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
    #     print(
    #         f"Move all joints except {full_turn_motors} sequentially through their "
    #         "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
    #     )
    #     range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
    #     for motor in full_turn_motors:
    #         range_mins[motor] = 0
    #         range_maxes[motor] = 4095

    #     self.calibration = {}
    #     for motor, m in self.bus.motors.items():
    #         self.calibration[motor] = MotorCalibration(
    #             id=m.id,
    #             drive_mode=drive_modes[motor],
    #             homing_offset=homing_offsets[motor],
    #             range_min=range_mins[motor],
    #             range_max=range_maxes[motor],
    #         )

    #     self.bus.write_calibration(self.calibration)
    #     self._save_calibration()
    #     logger.info(f"Calibration saved to {self.calibration_fpath}")



    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()

        # Set secondary/shadow ID for shoulder and elbow. These joints have two motors.
        # As a result, if only one of them is required to move to a certain position,
        # the other will follow. This is to avoid breaking the motors.
        self.bus.write("Secondary_ID", "shoulder_shadow", 2)
        self.bus.write("Secondary_ID", "elbow_shadow", 4)

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
