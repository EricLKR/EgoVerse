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

"""Thin wrapper around `piper_sdk` (https://github.com/agilexrobotics/piper_sdk) exposing the
subset of AgileX PIPER functionality needed by the LeRobot `Piper` robot class.

Unit reference, taken from `piper_sdk`'s own message definitions
(`piper_sdk/piper_msgs/msg_v2/{transmit,feedback}/*.py` in the agilexrobotics/piper_sdk repo):
  - `JointCtrl` / `GetArmJointMsgs` / `GetArmJointCtrl` joint angles: raw int, unit 0.001 degree.
  - `GetAllMotorAngleLimitMaxSpd` joint angle limits: raw int, unit 0.1 degree (100x coarser).
  - `GripperCtrl` / `GetArmGripperMsgs` gripper stroke: raw int, unit 0.001 mm.
"""

import logging
import time
from typing import Any

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    C_PiperInterface_V2 = None

logger = logging.getLogger(__name__)

# Verified against a real PIPER-L unit via examples/piper/verify_follower.py's before/after nudge
# test: joints 0 and 5 moved *opposite* the commanded direction while carried over from the original
# ported PR as "inverted" -- removed. Joint 3 (AgileX's own "joint4") was kept inverted for a while
# based on one ambiguous trial, but examples/piper/raw_sdk_joint3_test.py -- same joint, driven with
# *no* inversion at all, straight through piper_sdk with no lerobot code involved -- moved correctly
# and consistently across 5 repeated trials, including from starting positions where the (still)
# inverted lerobot-side code was giving wrong-direction results. That's strong evidence joint_3
# never needed inversion either; removed. So currently: no joints are believed to need inversion on
# this unit. Re-verify with examples/piper/verify_follower.py before fully trusting this.
_INVERTED_JOINTS = ()

# `GetAllMotorAngleLimitMaxSpd` limits are per-joint only (no gripper entry), so the gripper's
# open/close range isn't discoverable via the SDK the way joint limits are. AgileX's own PIPER-L
# datasheet ("Two-Finger Gripper Specifications") states Opening Range: 0-100mm -- this default.
# Cross-checked against the URDF (agilexrobotics/piper_isaac_sim): joint7/joint8 are the two jaws,
# each a symmetric prismatic joint limited to 0.05m, so total stroke = 2x that = 100mm, consistent.
# Standard (non-L) PIPER's datasheet/URDF gives 70mm instead -- pass gripper_stroke_mm=70.0 if
# that's your hardware. Still worth a quick sanity check against your specific unit (open it fully
# and read `GetArmGripperMsgs().gripper_state.grippers_angle`) in case of revision drift.
DEFAULT_GRIPPER_STROKE_MM = 100.0


def raw_deg001_to_norm(raw: float, min_deg01: float, max_deg01: float) -> float:
    """Map a raw 0.001deg joint reading to the [-100, 100] range used by `set_joint_positions`,
    using limits from `GetAllMotorAngleLimitMaxSpd` (unit 0.1deg, hence the /100)."""
    raw_deg01 = raw / 100.0
    if max_deg01 == min_deg01:
        return 0.0
    return (raw_deg01 - min_deg01) / (max_deg01 - min_deg01) * 200.0 - 100.0


def raw_mm001_to_pct(raw: float, stroke_mm: float) -> float:
    """Map a raw 0.001mm gripper reading to the [0, 100] range used by `set_joint_positions`."""
    if stroke_mm == 0:
        return 0.0
    return max(0.0, min(100.0, (raw / 1000.0) / stroke_mm * 100.0))


class PiperSDKInterface:
    """Talks to the physical arm over CAN via `piper_sdk`."""

    def __init__(
        self,
        port: str = "can0",
        gripper_stroke_mm: float = DEFAULT_GRIPPER_STROKE_MM,
        gripper_effort: int = 5000,
        speed_percent: int = 100,
        joint_limits_deg: list[tuple[float, float]] | None = None,
        enable_motion: bool = True,
    ):
        """`joint_limits_deg`: optional override, 6 (min, max) pairs in real degrees, one per arm
        joint. Bypasses `GetAllMotorAngleLimitMaxSpd()` for those joints entirely. Exists because
        that call was found (on a real PIPER-L unit) to report generic/uncalibrated defaults
        (symmetric +-170deg for every joint) rather than this unit's real, very different-per-joint
        limits -- see examples/piper/find_real_joint_limits.py for how those were empirically
        confirmed by hand. Leave as None to trust the live query (the original behavior, correct
        for units that report real calibrated limits).
        """
        if C_PiperInterface_V2 is None:
            raise ImportError("piper_sdk is not installed. Install it with `pip install lerobot[piper]`.")

        self.gripper_stroke_mm = gripper_stroke_mm
        if not 0 <= int(gripper_effort) <= 5000:
            raise ValueError(
                f"gripper_effort must be in [0,5000] SDK units, got {gripper_effort}"
            )
        self.gripper_effort = int(gripper_effort)
        self.speed_percent = speed_percent
        self.port = port
        self.motion_enabled = False

        self.piper = C_PiperInterface_V2(port)
        self.piper.ConnectPort()
        time.sleep(0.1)  # wait for the connection to establish

        # Force this arm into the "slave"/motion-output-arm hardware role via `MasterSlaveConfig`
        # (CAN ID 0x470), with feedback IDs at their default (unshifted) base. This is a *separate*
        # axis from lerobot's own leader/follower naming -- it's AgileX's own hardware master-slave
        # linkage feature (see piper_sdk's `MasterSlaveConfig`/`ReqMasterArmMoveToHome` docstrings).
        # Found the hard way: a real leader unit here was silently stuck in the "master"/teaching-
        # input-arm role (0xFA) -- almost certainly leftover from Piper_sdk_ui's own master/slave
        # toggle, whose git history has a commit literally titled "fixed master-slave switching
        # abnormality". In that role the arm sends *zero* CAN feedback at all (matches
        # agilexrobotics/piper_sdk's own README: "reading joint feedback requires the robot to be
        # in slave mode. If it is in master mode, the robot will not respond") -- which looks
        # *exactly* like unpowered motors from the outside: GetArmStatus() reports Hz=0.0 (stale,
        # zero-initialized cache, never actually updated by a real frame), every motor reads 0.0V,
        # and EnablePiper() times out -- even though the arm has power and its teach-button LED
        # responds normally. Also observed this state recur mid-session with nothing intentionally
        # sent to cause it, so don't treat one working connect as proof it'll stay that way --
        # this is sent unconditionally on every connect, not just as a one-time recovery step.
        self.piper.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
        time.sleep(0.2)  # let the first live feedback frames arrive before trusting GetArmStatus() below

        if enable_motion:
            # Resume the controller itself as well as its motion state.  On this
            # unit an emergency-stopped controller can still report
            # motion_status == 0 and EnablePiper() can still return True, while
            # silently rejecting every JointCtrl command.  Checking only
            # motion_status therefore produces a dangerous false-positive
            # "enabled" state.
            self._resume_controller_if_needed()

            enable_deadline = time.time() + 10.0
            enable_attempts = 0
            while not self.piper.EnablePiper():
                enable_attempts += 1
                if time.time() > enable_deadline:
                    low_spd = self.piper.GetArmLowSpdInfoMsgs()
                    voltages = [
                        low_spd.motor_1.vol, low_spd.motor_2.vol, low_spd.motor_3.vol,
                        low_spd.motor_4.vol, low_spd.motor_5.vol, low_spd.motor_6.vol,
                    ]
                    raise RuntimeError(
                        f"EnablePiper() on port {port!r} did not succeed after {enable_attempts} attempts "
                        f"(10s). Per-motor voltage (x0.1V, 0 usually means no motor power reaching the "
                        f"driver): {[v * 0.1 for v in voltages]}. If these are all 0, check the arm's main "
                        "DC power supply/adapter (separate from the CAN connection) is plugged in, "
                        "switched on, and its own indicator light is lit -- CAN comms working (this far) "
                        "does not mean the motor power rail is live."
                    )
                time.sleep(0.01)

            # Joint motion control at the configured speed cap (0-100).
            self.piper.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
            self.motion_enabled = True
            self._verify_controller_accepts_motion()
        else:
            logger.warning(
                "Piper connected in observation-only mode: motion enable/resume/control commands "
                "are disabled, and set_joint_positions will refuse all commands."
            )

        # Per-joint min/max limits (unit 0.1deg), used to scale the [-100, 100] normalized action
        # range onto real joint angles. `joint_limits_deg`, if given, overrides the live query --
        # see this method's docstring for why that's sometimes necessary.
        if joint_limits_deg is not None:
            if len(joint_limits_deg) != 6:
                raise ValueError(f"joint_limits_deg needs 6 (min, max) pairs, got {len(joint_limits_deg)}")
            self.min_pos = [lo * 10.0 for lo, _ in joint_limits_deg] + [0.0]
            self.max_pos = [hi * 10.0 for _, hi in joint_limits_deg] + [100.0]
        else:
            angle_status = self.piper.GetAllMotorAngleLimitMaxSpd()
            queried_min = [
                joint.min_angle_limit for joint in angle_status.all_motor_angle_limit_max_spd.motor[1:7]
            ]
            queried_max = [
                joint.max_angle_limit for joint in angle_status.all_motor_angle_limit_max_spd.motor[1:7]
            ]
            self.min_pos = queried_min + [0.0]
            self.max_pos = queried_max + [100.0]  # gripper's own [0, 100] range, not a joint limit.

    def _resume_controller_if_needed(self) -> bool:
        """Request resume when any controller state says motion is blocked.

        Returns True when a resume command was sent.  This method deliberately
        does not claim success; `_verify_controller_accepts_motion` performs
        the post-enable feedback check.
        """
        status = self.piper.GetArmStatus().arm_status
        emergency_stopped = status.arm_status == 1
        teaching = status.ctrl_mode == 2
        motion_blocked = status.motion_status != 0
        if not (emergency_stopped or teaching or motion_blocked):
            return False

        logger.warning(
            "Resuming Piper controller before motion enable "
            "(arm_status=%s, ctrl_mode=%s, motion_status=%s).",
            status.arm_status,
            status.ctrl_mode,
            status.motion_status,
        )
        self.piper.EmergencyStop(0x02)
        time.sleep(0.1)
        return True

    def _verify_controller_accepts_motion(self, timeout_s: float = 2.0) -> None:
        """Fail closed if feedback still reports emergency stop after enable."""
        deadline = time.time() + timeout_s
        status = self.piper.GetArmStatus().arm_status
        while status.arm_status == 1 and time.time() < deadline:
            time.sleep(0.02)
            status = self.piper.GetArmStatus().arm_status

        if status.arm_status == 1:
            self.motion_enabled = False
            raise RuntimeError(
                "Piper controller remained in EMERGENCY_STOP after resume and EnablePiper; "
                "refusing to report motion enabled or send policy commands."
            )

    def set_joint_positions(self, positions: list[float]) -> None:
        """`positions`: 7 floats in [-100, 100], the first 6 are joints and the 7th is the gripper."""
        if not self.motion_enabled:
            raise RuntimeError(
                "Piper is connected in observation-only mode; refusing JointCtrl/GripperCtrl. "
                "Reconnect explicitly with enable_motion=True only after guarded dry-run validation."
            )
        scaled = [
            100.0 * (self.min_pos[i] + (self.max_pos[i] - self.min_pos[i]) * (pos + 100) / 200)
            for i, pos in enumerate(positions[:6])
        ]
        # Gripper: [0, 100] -> [0, gripper_stroke_mm] -> raw 0.001mm units.
        gripper_mm = self.gripper_stroke_mm * max(0.0, min(100.0, positions[6])) / 100
        scaled.append(int(gripper_mm * 1_000))

        joints = [int(-v) if i in _INVERTED_JOINTS else int(v) for i, v in enumerate(scaled[:6])]
        self.piper.JointCtrl(*joints)
        self.piper.GripperCtrl(scaled[6], self.gripper_effort, 0x01, 0)

    def get_motor_faults(self) -> list[dict[str, Any]]:
        """Per-arm-joint (0-5) driver fault flags, from `GetArmLowSpdInfoMsgs` (CAN IDs 0x261-0x266,
        one per motor). Useful when a joint won't move under `set_joint_positions` -- e.g.
        `stall_status`/`collision_status` being True means the firmware's own protection is refusing
        to drive it, which looks identical to "nothing happened" from the normalized-position API
        alone. Note motors here are 1-indexed in the SDK (motor_1..motor_6); this returns them
        0-indexed to match our joint_0..joint_5 naming.
        """
        info = self.piper.GetArmLowSpdInfoMsgs()
        faults = []
        for i in range(1, 7):
            motor = getattr(info, f"motor_{i}")
            faults.append(
                {
                    "motor_temp_c": motor.motor_temp,
                    "driver_temp_c": motor.foc_temp,
                    "voltage_v": motor.vol * 0.1,
                    "voltage_too_low": motor.foc_status.voltage_too_low,
                    "motor_overheating": motor.foc_status.motor_overheating,
                    "driver_overcurrent": motor.foc_status.driver_overcurrent,
                    "driver_overheating": motor.foc_status.driver_overheating,
                    "collision_protection_triggered": motor.foc_status.collision_status,
                    "driver_error": motor.foc_status.driver_error_status,
                    "driver_enabled": motor.foc_status.driver_enable_status,
                    "stall_protection_triggered": motor.foc_status.stall_status,
                }
            )
        return faults

    def get_status(self) -> dict[str, Any]:
        """Raw sensor readout: joints in 0.001deg, gripper in 0.001mm (see module docstring)."""
        joint_state = self.piper.GetArmJointMsgs().joint_state
        gripper_state = self.piper.GetArmGripperMsgs().gripper_state
        return {
            "joint_0.pos": joint_state.joint_1,
            "joint_1.pos": joint_state.joint_2,
            "joint_2.pos": joint_state.joint_3,
            "joint_3.pos": joint_state.joint_4,
            "joint_4.pos": joint_state.joint_5,
            "joint_5.pos": joint_state.joint_6,
            "joint_6.pos": gripper_state.grippers_angle,
        }

    def get_status_normalized(self) -> dict[str, float]:
        """Same readout as `get_status`, but rescaled into the same [-100, 100] (joints) / [0, 100]
        (gripper) frame that `set_joint_positions` accepts -- use this (not `get_status`) whenever
        comparing against a commanded/goal position, e.g. for `max_relative_target` safety clamping."""
        raw = self.get_status()
        normalized = {
            f"joint_{i}.pos": raw_deg001_to_norm(raw[f"joint_{i}.pos"], self.min_pos[i], self.max_pos[i])
            for i in range(6)
        }
        normalized["joint_6.pos"] = raw_mm001_to_pct(raw["joint_6.pos"], self.gripper_stroke_mm)
        return normalized

    def _prime_measured_raw_target(self, raw: dict[str, Any]) -> None:
        """Load the measured pose into the firmware target registers.

        This deliberately uses raw feedback units and bypasses
        ``set_joint_positions`` because startup calls it while normal motion is
        still disabled.  Priming while quick-stopped prevents a resume command
        from briefly replaying a stale target left by an earlier process.
        """
        joints = [int(raw[f"joint_{index}.pos"]) for index in range(6)]
        gripper = int(raw["joint_6.pos"])
        self.piper.JointCtrl(*joints)
        self.piper.GripperCtrl(gripper, self.gripper_effort, 0x01, 0)

    def enable_motion_hold_current(self) -> dict[str, float]:
        """Promote an observation-only connection and immediately latch feedback.

        Deployment deliberately performs camera/model/IK/startup checks before
        enabling the arm.  Once those checks pass, this method snapshots the
        current normalized joint state, enables motion control, and sends that
        exact state as the first command.  This prevents the interval between
        ``EnablePiper`` and the first policy command from becoming an
        uncontrolled gravity drop or a replay of a stale controller target.
        """
        if self.motion_enabled:
            raise RuntimeError("Piper motion is already enabled on this connection")

        raw_hold = self.get_status()
        hold_position = {
            f"joint_{index}.pos": raw_deg001_to_norm(
                raw_hold[f"joint_{index}.pos"], self.min_pos[index], self.max_pos[index]
            )
            for index in range(6)
        }
        hold_position["joint_6.pos"] = raw_mm001_to_pct(
            raw_hold["joint_6.pos"], self.gripper_stroke_mm
        )

        # A controller in REACH_TARGET_POS_FAILED or emergency-stop can retain
        # the last JointCtrl target.  Resuming first creates a race in which
        # that stale target may execute before the measured-pose hold arrives.
        # Force a torque-holding quick-stop, preload the measured raw pose,
        # resume, and immediately preload it again before enabling motion.
        self.piper.EmergencyStop(0x01)
        time.sleep(0.05)
        self._prime_measured_raw_target(raw_hold)
        self.piper.EmergencyStop(0x02)
        time.sleep(0.05)
        self._prime_measured_raw_target(raw_hold)

        enable_deadline = time.time() + 10.0
        enable_attempts = 0
        while not self.piper.EnablePiper():
            enable_attempts += 1
            if time.time() > enable_deadline:
                low_spd = self.piper.GetArmLowSpdInfoMsgs()
                voltages = [
                    low_spd.motor_1.vol,
                    low_spd.motor_2.vol,
                    low_spd.motor_3.vol,
                    low_spd.motor_4.vol,
                    low_spd.motor_5.vol,
                    low_spd.motor_6.vol,
                ]
                raise RuntimeError(
                    f"EnablePiper() on port {self.port!r} did not succeed after "
                    f"{enable_attempts} attempts (10s). Per-motor voltage (x0.1V): "
                    f"{[v * 0.1 for v in voltages]}"
                )
            time.sleep(0.01)

        self._prime_measured_raw_target(raw_hold)
        self.piper.MotionCtrl_2(0x01, 0x01, self.speed_percent, 0x00)
        self.motion_enabled = True
        try:
            self.set_joint_positions(
                [hold_position[f"joint_{index}.pos"] for index in range(7)]
            )
            self._verify_controller_accepts_motion()
        except BaseException:
            # Do not leave an enabled controller waiting without a target if
            # the latch command itself fails.
            self.piper.EmergencyStop(0x01)
            raise
        return hold_position

    def disconnect(self, home_first: bool = True, disable_torque: bool = True) -> None:
        """Disable torque via `DisablePiper()`, matching agilexrobotics/piper_sdk's own
        `piper_ctrl_disable.py` demo -- not a `JointCtrl` motion command like the previous version
        of this method used (`JointCtrl(0, 0, 0, 0, 25000, 0)`, inherited from the original ported
        PR, which moved the arm rather than disabling it -- see the fix for that bug).

        If `home_first`, commands all 6 joints to their normalized midpoint (gripper left untouched)
        at the connection's configured speed and waits until they actually arrive (polled, not a
        fixed sleep) before cutting torque.

        SAFETY: even a completed, gentle home move does not make cutting torque safe. If this arm
        has no passive brakes at the loaded joints, `DisablePiper()` can still make it drop --
        abruptly -- from *whatever* pose it ends up in, home or not, if that pose isn't gravity-
        neutral. Only pass `home_first=True` (or call `disconnect()` at all) when the arm is resting
        on something that can physically hold its weight regardless of pose -- a stand, the table,
        a hand. This method cannot make disabling torque safe on its own.

        This also disables the gripper (separate enable state from the 6 arm joints -- see below),
        which means the SAME caution applies to it: whatever the gripper is holding, or the jaws
        themselves under gravity, can move/fall once its torque releases too. Support it, or
        whatever's in it, before disconnecting -- same as the arm.
        """
        if not self.motion_enabled or not disable_torque:
            self.piper.DisconnectPort()
            self.motion_enabled = False
            return

        if home_first:
            gripper_now = self.get_status_normalized()["joint_6.pos"]
            self.set_joint_positions([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_now])

            deadline = time.time() + 5.0  # don't wait forever if a joint never arrives (e.g. stuck)
            while time.time() < deadline:
                pos = self.get_status_normalized()
                if all(abs(pos[f"joint_{i}.pos"]) < 3.0 for i in range(6)):
                    break
                time.sleep(0.1)

        while self.piper.DisablePiper():
            time.sleep(0.01)

        # The gripper has its own, separate enable state from the 6 arm joints -- DisablePiper()
        # above does NOT touch it. Found this the hard way: every previous version of this method
        # left the gripper enabled/rigid indefinitely, even with no script running and the arm
        # joints properly released. status_code 0x00 = disable.
        self.piper.GripperCtrl(0, self.gripper_effort, 0x00, 0)
        self.piper.DisconnectPort()
        self.motion_enabled = False

    def soft_stop(self) -> None:
        """Untested alternative to `disconnect()`, NOT called automatically anywhere -- call it
        yourself, deliberately, with the arm physically supported (stand/table/hand), to observe
        what it actually does on your unit before ever relying on it.

        `DisablePiper()` (what `disconnect()` uses) is documented by AgileX's own SDK as cutting
        power to the motors immediately (their Q&A #1; `ResetPiper()`'s docstring is blunter still:
        "the arm will immediately lose power and fall"). Their demo table describes this command
        instead (`EmergencyStop(0x01)`, CAN ID 0x150, from `piper_ctrl_stop.py`) as letting the arm
        "descend slowly" -- a materially different, possibly gentler, description. Not confirmed to
        hold position; only that it's described differently from the immediate-power-cut path.

        Recovery is NOT a normal reconnect: per the same doc, you need `EmergencyStop(0x02)`
        (resume) and `EnablePiper()` called twice before the arm accepts normal control again --
        `PiperSDKInterface.__init__` only handles resuming from a non-idle `motion_status`, not
        specifically this state, so don't assume the next `connect()` will cleanly recover from it
        without checking.
        """
        self.piper.EmergencyStop(0x01)
