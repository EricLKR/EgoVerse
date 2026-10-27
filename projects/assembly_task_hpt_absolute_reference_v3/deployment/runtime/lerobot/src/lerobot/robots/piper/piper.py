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
import math
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors.motors_bus import MotorCalibration
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_piper import PiperConfig
from .piper_sdk_interface import PiperSDKInterface

_JOINT_KEYS = [f"joint_{i}" for i in range(6)]

logger = logging.getLogger(__name__)


class Piper(Robot):
    """[AgileX PIPER](https://github.com/agilexrobotics/piper_sdk) 6-DoF arm + gripper, driven over CAN.

    Covers the whole PIPER family (PIPER, PIPER-L, ...) -- they share the same `piper_sdk` protocol
    and joint-angle limits are queried live from the connected arm, not hardcoded. The one thing
    that differs by variant and isn't discoverable via the SDK is gripper stroke -- see
    `PiperConfig.gripper_stroke_mm` (defaults to PIPER-L's 100mm; standard PIPER is 70mm).
    """

    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig):
        super().__init__(config)
        self.config = config
        self.sdk: PiperSDKInterface | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        return {f"joint_{i}.pos": float for i in range(7)}

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
        return self.sdk is not None and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # If a calibration file was loaded (by Robot.__init__, from calibration_fpath) use its
        # numbers as the starting point; config.joint_limits_deg/gripper_stroke_mm are only a
        # fallback for a brand new id with no saved calibration yet -- see calibrate() for how a
        # config override gets promoted into a saved calibration on first connect.
        joint_limits_deg = self._joint_limits_deg_from_calibration() or self.config.joint_limits_deg
        gripper_stroke_mm = self._gripper_stroke_mm_from_calibration() or self.config.gripper_stroke_mm

        connected_cameras = []
        try:
            # Validate all observation inputs before opening CAN or enabling motion.
            # A camera startup failure must never leave a half-started execute session.
            for cam in self.cameras.values():
                cam.connect()
                connected_cameras.append(cam)

            self.sdk = PiperSDKInterface(
                port=self.config.port,
                gripper_stroke_mm=gripper_stroke_mm,
                gripper_effort=self.config.gripper_effort,
                speed_percent=self.config.speed_percent,
                joint_limits_deg=joint_limits_deg,
                enable_motion=self.config.enable_motion_on_connect,
            )

            if not self.is_calibrated and calibrate and not self.config.enable_motion_on_connect:
                logger.warning(
                    "Skipping calibration because enable_motion_on_connect=False. Observation-only "
                    "mode never disables/enables torque or asks for hands-on joint motion."
                )
            elif not self.is_calibrated and calibrate:
                logger.info(
                    f"No saved calibration for {self} (or config.joint_limits_deg was never set) -- "
                    "running calibrate()."
                )
                self.calibrate()

            self.configure()
        except BaseException:
            if self.sdk is not None:
                try:
                    # Close CAN without issuing a home command or dropping torque.
                    self.sdk.disconnect(home_first=False, disable_torque=False)
                except Exception:
                    logger.exception("Failed to close Piper CAN after a startup error")
                self.sdk = None
            for cam in reversed(connected_cameras):
                if cam.is_connected:
                    try:
                        cam.disconnect()
                    except Exception:
                        logger.exception("Failed to close %s after a startup error", cam)
            raise

        logger.info(f"{self} connected.")

    def _joint_limits_deg_from_calibration(self) -> list[tuple[float, float]] | None:
        if not all(k in self.calibration for k in _JOINT_KEYS):
            return None
        # Stored in raw 0.001deg (see calibrate()); convert to plain degrees for PiperSDKInterface.
        return [
            (self.calibration[k].range_min / 1000.0, self.calibration[k].range_max / 1000.0)
            for k in _JOINT_KEYS
        ]

    def _gripper_stroke_mm_from_calibration(self) -> float | None:
        cal = self.calibration.get("gripper")
        return None if cal is None else cal.range_max / 1000.0  # stored in raw 0.001mm

    @property
    def is_calibrated(self) -> bool:
        return all(k in self.calibration for k in _JOINT_KEYS) and "gripper" in self.calibration

    def calibrate(self) -> None:
        """Populates and saves self.calibration (loaded automatically on future connects).

        Two paths:
          - If config.joint_limits_deg/gripper_stroke_mm were explicitly set (e.g. numbers you
            already hand-measured with find_real_joint_limits.py / read_gripper_stroke.py), those
            are saved directly -- no need to redo the hands-on procedure just to persist them.
          - Otherwise, runs that same hands-on procedure here: releases torque, has you move each
            joint (and the gripper) through its real range by hand, records min/max seen.
        """
        if not self.config.enable_motion_on_connect:
            raise RuntimeError("Calibration is disabled in observation-only mode.")
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
                for i, (k, (lo, hi)) in enumerate(
                    zip(_JOINT_KEYS, self.config.joint_limits_deg, strict=True)
                )
            }
            self.calibration["gripper"] = MotorCalibration(
                id=6, drive_mode=0, homing_offset=0,
                range_min=0, range_max=int(self.config.gripper_stroke_mm * 1000),
            )
            self._save_calibration()
            print(f"Calibration saved to {self.calibration_fpath}")
            return

        logger.info(f"\nRunning hands-on calibration of {self}")
        while self.sdk.piper.DisablePiper():
            time.sleep(0.01)
        print(
            f"\n{self}: torque released. Move each of the 6 joints, one at a time, slowly through "
            "its full range of motion (gently to the mechanical stop, not into it -- and watch for "
            "workspace obstructions, not just the joint's own limit)."
        )
        input("Press Enter to start recording (~15s to move all 6 joints)...")
        seen_min = [None] * 6
        seen_max = [None] * 6
        deadline = time.time() + 15.0
        while time.time() < deadline:
            raw = self.sdk.get_status()
            for i, k in enumerate(_JOINT_KEYS):
                v = raw[f"{k}.pos"]
                seen_min[i] = v if seen_min[i] is None else min(seen_min[i], v)
                seen_max[i] = v if seen_max[i] is None else max(seen_max[i], v)
            time.sleep(0.05)
        for i, k in enumerate(_JOINT_KEYS):
            print(f"  {k}: {seen_min[i] / 1000:.1f} to {seen_max[i] / 1000:.1f} deg")

        print("\nNow open/close the gripper fully by hand.")
        input("Press Enter to start recording (~5s)...")
        gripper_max = 0
        deadline = time.time() + 5.0
        while time.time() < deadline:
            gripper_max = max(gripper_max, self.sdk.get_status()["joint_6.pos"])
            time.sleep(0.05)
        print(f"  gripper: {gripper_max / 1000:.1f} mm")

        self.calibration = {
            k: MotorCalibration(id=i, drive_mode=0, homing_offset=0, range_min=seen_min[i], range_max=seen_max[i])
            for i, k in enumerate(_JOINT_KEYS)
        }
        self.calibration["gripper"] = MotorCalibration(
            id=6, drive_mode=0, homing_offset=0, range_min=0, range_max=gripper_max
        )
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

        # Apply what we just measured to the live sdk, and restore normal operating state.
        joint_limits_deg = self._joint_limits_deg_from_calibration()
        for i in range(6):
            self.sdk.min_pos[i] = joint_limits_deg[i][0] * 10.0
            self.sdk.max_pos[i] = joint_limits_deg[i][1] * 10.0
        self.sdk.gripper_stroke_mm = self._gripper_stroke_mm_from_calibration()
        while not self.sdk.piper.EnablePiper():
            time.sleep(0.01)

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict = self.sdk.get_status()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(
        self,
        action: dict[str, Any],
        *,
        present_position: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Command the arm to a target joint configuration (`joint_0.pos` ... `joint_6.pos`, the last
        being the gripper), each in [-100, 100].

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
        if not self.sdk.motion_enabled:
            raise RuntimeError(
                "Piper was connected observation-only; refusing to send an action. "
                "Run the guarded preflight and explicitly promote this connection first."
            )

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # NOTE: must compare/fall back to the *normalized* present position (same [-100, 100] /
        # [0, 100] frame as goal_pos), not sdk.get_status()'s raw sensor units -- otherwise the diff
        # (for max_relative_target) or the fallback (for a partial action, below) is meaningless.
        if present_position is None:
            present_pos = {
                key.removesuffix(".pos"): val
                for key, val in self.sdk.get_status_normalized().items()
            }
        else:
            # Deployment may pass the exact normalized feedback sample that was
            # used by its trajectory filter.  Re-reading feedback here would
            # apply the relative-target clamp against a different timestamp and
            # can turn harmless feedback motion into a false second clamp.
            present_pos = {
                key.removesuffix(".pos"): float(value)
                for key, value in present_position.items()
            }
            expected = {f"joint_{index}" for index in range(7)}
            if set(present_pos) != expected:
                raise ValueError(
                    "present_position must contain exactly joint_0..joint_6; "
                    f"got {sorted(present_pos)}"
                )
            if not all(math.isfinite(value) for value in present_pos.values()):
                raise ValueError("present_position contains NaN or Inf")

        if self.config.max_relative_target is not None:
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Any joint missing from the action (e.g. an action that only sets the gripper) holds its
        # current position instead of snapping to 0 -- see piper_sdk_interface.py's normalization
        # for why 0 would otherwise mean "joint's minimum limit", not "no change".
        positions = [goal_pos.get(f"joint_{i}", present_pos[f"joint_{i}"]) for i in range(7)]
        self.sdk.set_joint_positions(positions)
        return {f"{joint}.pos": val for joint, val in goal_pos.items()}

    def enable_motion_hold_current(self) -> dict[str, float]:
        """Enable motion only after deployment preflight and latch feedback."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self.sdk.enable_motion_hold_current()

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.sdk.disconnect(
            home_first=self.config.home_on_disconnect,
            disable_torque=self.config.disable_torque_on_disconnect,
        )
        self.sdk = None
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
