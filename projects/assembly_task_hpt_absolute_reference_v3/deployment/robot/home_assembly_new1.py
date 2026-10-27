#!/usr/bin/env python
"""Model-free guarded homing for the Assembly New1 PIPER task.

This entry point deliberately does not import torch, any policy class, a
checkpoint, or a camera driver.  It opens only the requested PIPER CAN port.

``check`` is observation-only and never sends an enable or motion command.
``execute`` requires ``--allow-motion`` and returns the physical deployment arm
(the original collection leader) to its saved pose corresponding to the
follower training Home, using a bounded quintic trajectory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from joint_space_adapter import load_joint_zero_offset


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION = BUNDLE_ROOT / "assets/piper_original_leader_calibration.json"
DEFAULT_INITIAL_POSE = (
    BUNDLE_ROOT / "assets/initial_pose_follower_physical_on_original_leader.json"
)
DEFAULT_JOINT_ZERO_OFFSET = BUNDLE_ROOT / "assets/joint_zero_offset.json"
DEFAULT_LOG_DIR = BUNDLE_ROOT / "logs/home"
CONTROL_HZ = 30.0
J2_INDEX = 1
J3_INDEX = 2
J5_INDEX = 4


def load_piper_sdk_interface_class() -> type:
    """Load only the thin CAN wrapper, bypassing LeRobot's torch-heavy package init."""
    source = (
        BUNDLE_ROOT
        / "runtime/lerobot/src/lerobot/robots/piper/piper_sdk_interface.py"
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"isolated PIPER runtime is missing: {source}; run tools/prepare_runtime.sh"
        )
    spec = importlib.util.spec_from_file_location(
        "assembly_new1_model_free_piper_sdk_interface", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PiperSDKInterface


PiperSDKInterface = load_piper_sdk_interface_class()


@dataclass(frozen=True)
class PhysicalPose:
    joints_deg: tuple[float, ...]
    gripper_mm: float


class JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("x")

    def write(self, record: dict[str, Any]) -> None:
        self._stream.write(json.dumps(record, allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def load_calibration(path: Path) -> tuple[list[tuple[float, float]], float]:
    with path.open() as stream:
        payload = json.load(stream)
    limits = [
        (
            float(payload[f"joint_{index}"]["range_min"]) / 1000.0,
            float(payload[f"joint_{index}"]["range_max"]) / 1000.0,
        )
        for index in range(6)
    ]
    gripper_stroke_mm = float(payload["gripper"]["range_max"]) / 1000.0
    if any(not math.isfinite(value) for pair in limits for value in pair):
        raise ValueError(f"non-finite joint calibration in {path}")
    if any(high <= low for low, high in limits):
        raise ValueError(f"invalid joint limits in {path}: {limits}")
    if not math.isfinite(gripper_stroke_mm) or gripper_stroke_mm <= 0:
        raise ValueError(f"invalid gripper stroke in {path}: {gripper_stroke_mm}")
    return limits, gripper_stroke_mm


def load_initial_pose(
    path: Path,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> PhysicalPose:
    with path.open() as stream:
        payload = json.load(stream)
    joints = tuple(float(value) for value in payload["joint_positions_deg"])
    gripper = float(payload["gripper_opening_mm"])
    if len(joints) != 6 or not all(math.isfinite(value) for value in (*joints, gripper)):
        raise ValueError(f"invalid initial pose in {path}")
    for index, (angle, (low, high)) in enumerate(zip(joints, joint_limits, strict=True)):
        if not low <= angle <= high:
            raise ValueError(
                f"initial pose joint {index}={angle} deg is outside calibration "
                f"[{low},{high}]"
            )
    if not 0.0 <= gripper <= gripper_stroke_mm:
        raise ValueError(
            f"initial pose gripper={gripper} mm is outside [0,{gripper_stroke_mm}]"
        )
    return PhysicalPose(joints, gripper)


def raw_feedback_pose(raw: dict[str, Any]) -> PhysicalPose:
    joints = tuple(float(raw[f"joint_{index}.pos"]) / 1000.0 for index in range(6))
    gripper = float(raw["joint_6.pos"]) / 1000.0
    if not all(math.isfinite(value) for value in (*joints, gripper)):
        raise RuntimeError(f"non-finite PIPER feedback: {raw}")
    return PhysicalPose(joints, gripper)


def apply_j2_feedback_offset(
    pose: PhysicalPose,
    j2_feedback_offset_deg: float,
) -> PhysicalPose:
    """Correct only the reported J2 measurement; never alter a command target."""

    joints = list(pose.joints_deg)
    joints[J2_INDEX] += float(j2_feedback_offset_deg)
    return PhysicalPose(tuple(joints), pose.gripper_mm)


def apply_j2_command_offset(
    pose: PhysicalPose,
    j2_command_offset_deg: float,
) -> PhysicalPose:
    """Map a model/recorded J2 target to the firmware J2 convention."""

    joints = list(pose.joints_deg)
    joints[J2_INDEX] += float(j2_command_offset_deg)
    return PhysicalPose(tuple(joints), pose.gripper_mm)


def apply_j3_command_offset(
    pose: PhysicalPose,
    j3_command_offset_deg: float,
) -> PhysicalPose:
    """Map a model/recorded J3 target to the firmware J3 convention."""

    joints = list(pose.joints_deg)
    joints[J3_INDEX] += float(j3_command_offset_deg)
    return PhysicalPose(tuple(joints), pose.gripper_mm)


def apply_j5_command_offset(
    pose: PhysicalPose,
    j5_command_offset_deg: float,
) -> PhysicalPose:
    """Map a model/recorded J5 target to the firmware J5 convention."""

    joints = list(pose.joints_deg)
    joints[J5_INDEX] += float(j5_command_offset_deg)
    return PhysicalPose(tuple(joints), pose.gripper_mm)


def corrected_feedback_pose(
    sdk: PiperSDKInterface,
    j2_feedback_offset_deg: float,
) -> PhysicalPose:
    return apply_j2_feedback_offset(
        raw_feedback_pose(sdk.get_status()),
        j2_feedback_offset_deg,
    )


def physical_pose_to_normalized(
    pose: PhysicalPose,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> np.ndarray:
    values = [
        (angle - low) / (high - low) * 200.0 - 100.0
        for angle, (low, high) in zip(pose.joints_deg, joint_limits, strict=True)
    ]
    values.append(pose.gripper_mm / gripper_stroke_mm * 100.0)
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (7,) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid normalized home command: {result.tolist()}")
    return result


def normalized_feedback(sdk: PiperSDKInterface) -> np.ndarray:
    feedback = sdk.get_status_normalized()
    result = np.asarray(
        [float(feedback[f"joint_{index}.pos"]) for index in range(7)],
        dtype=np.float64,
    )
    if result.shape != (7,) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid normalized feedback: {result.tolist()}")
    return result


def bounded_send(
    sdk: PiperSDKInterface,
    requested: np.ndarray,
    *,
    max_arm_relative_target: float,
    max_gripper_relative_target: float,
) -> dict[str, list[float] | bool]:
    present = normalized_feedback(sdk)
    limits = np.asarray(
        [max_arm_relative_target] * 6 + [max_gripper_relative_target],
        dtype=np.float64,
    )
    delta = requested - present
    sent = present + np.clip(delta, -limits, limits)
    # A measured joint may begin just outside the calibrated deployment range
    # while still inside ``home_start_limit_margin_deg``.  Keep the exact
    # continuous recovery path in that narrow band; clipping to +/-100 here
    # would create an immediate jump to the calibration endpoint.
    sent[6] = np.clip(sent[6], 0.0, 100.0)
    sdk.set_joint_positions(sent.tolist())
    return {
        "requested": requested.tolist(),
        "present": present.tolist(),
        "sent": sent.tolist(),
        "clamped": bool(np.any(np.abs(delta) > limits)),
    }


def latch_current_pose(sdk: PiperSDKInterface) -> list[float]:
    current = normalized_feedback(sdk)
    sdk.set_joint_positions(current.tolist())
    return current.tolist()


def quintic_blend(fraction: float) -> float:
    t = max(0.0, min(1.0, float(fraction)))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def quintic_joint_duration(
    start: tuple[float, ...],
    goal: tuple[float, ...],
    velocity_deg_s: float,
    acceleration_deg_s2: float,
) -> float:
    delta = max(abs(a - b) for a, b in zip(start, goal, strict=True))
    return max(
        1.0 / CONTROL_HZ,
        1.875 * delta / velocity_deg_s,
        math.sqrt(5.774 * delta / acceleration_deg_s2),
    )


def validate_start_pose(
    measured: PhysicalPose,
    args: argparse.Namespace,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> tuple[PhysicalPose, dict[str, Any]]:
    recovery_joints: list[int] = []
    for index, (angle, (low, high)) in enumerate(
        zip(measured.joints_deg, joint_limits, strict=True)
    ):
        allowed_low = low - args.home_start_limit_margin_deg
        allowed_high = high + args.home_start_limit_margin_deg
        if not allowed_low <= angle <= allowed_high:
            raise RuntimeError(
                f"cannot home from joint {index}={angle} deg outside recovery "
                f"range [{allowed_low},{allowed_high}]"
            )
        if not low <= angle <= high:
            recovery_joints.append(index)
    if not (
        -args.home_gripper_tolerance_mm
        <= measured.gripper_mm
        <= gripper_stroke_mm + args.home_gripper_tolerance_mm
    ):
        raise RuntimeError(
            f"cannot home from gripper={measured.gripper_mm} mm outside endpoint margin"
        )
    clamped_gripper = min(gripper_stroke_mm, max(0.0, measured.gripper_mm))
    return PhysicalPose(measured.joints_deg, clamped_gripper), {
        "measured_joints_deg": list(measured.joints_deg),
        "measured_gripper_mm": measured.gripper_mm,
        "recovery_joints": recovery_joints,
        "gripper_endpoint_clamped": clamped_gripper != measured.gripper_mm,
    }


def verify_health(
    sdk: PiperSDKInterface,
    *,
    require_enabled: bool,
    check_controller: bool,
) -> list[dict[str, Any]]:
    status = sdk.piper.GetArmStatus().arm_status
    if check_controller and (status.err_code or status.arm_status or status.ctrl_mode != 1):
        raise RuntimeError(f"PIPER controller fault during homing: {status}")
    fault_names = (
        "voltage_too_low",
        "motor_overheating",
        "driver_overcurrent",
        "driver_overheating",
        "collision_protection_triggered",
        "driver_error",
        "stall_protection_triggered",
    )
    motors = sdk.get_motor_faults()
    for index, motor in enumerate(motors):
        active = [name for name in fault_names if motor[name]]
        if active:
            raise RuntimeError(f"PIPER motor {index + 1} fault during homing: {active}")
        if require_enabled and not motor["driver_enabled"]:
            raise RuntimeError(f"PIPER motor {index + 1} lost enable during homing")
    return motors


def require_live_motor_feedback(
    raw_feedback: dict[str, Any],
    motors: list[dict[str, Any]],
    *,
    context: str,
) -> None:
    """Fail closed when SDK telemetry is still its all-zero cache default."""

    if len(motors) != 6:
        raise RuntimeError(
            f"PIPER returned {len(motors)} motor records during {context}; expected 6"
        )
    voltages = np.asarray(
        [float(motor["voltage_v"]) for motor in motors], dtype=np.float64
    )
    if not np.isfinite(voltages).all():
        raise RuntimeError(
            f"non-finite PIPER motor voltage feedback during {context}: "
            f"{voltages.tolist()}"
        )
    if not np.any(voltages > 0.0):
        raw_state = [
            float(raw_feedback[f"joint_{index}.pos"]) for index in range(7)
        ]
        raise RuntimeError(
            "no live PIPER motor feedback during "
            f"{context}: motor_voltage_v={voltages.tolist()}, raw_state={raw_state}. "
            "The CAN interface may be UP while the arm is powered off, disconnected, "
            "or attached to a different CAN port. No motion command was sent."
        )


def wait_for_stable_health(
    sdk: PiperSDKInterface,
    *,
    hold_position: dict[str, float] | None = None,
) -> None:
    """Keep an adopted controller in CAN mode through the disconnect race.

    A previous deployment connection can report one final CAN_CTRL sample and
    only then fall back to STANDBY after its port is closed.  Merely debouncing
    five early samples therefore accepts a stale mode and Home loses the
    controller a fraction of a second later.  During the complete handoff
    window, repeatedly command the already measured hold target and recover a
    delayed STANDBY transition without issuing quick-stop or DisablePiper.
    """
    deadline = time.perf_counter() + 2.0
    healthy = 0
    last_error: Exception | None = None
    while time.perf_counter() <= deadline:
        try:
            verify_health(sdk, require_enabled=True, check_controller=False)
            status = sdk.piper.GetArmStatus().arm_status
            if status.err_code or status.arm_status:
                raise RuntimeError(f"PIPER controller fault during homing: {status}")
            if status.ctrl_mode == 0 and hold_position is not None:
                healthy = 0
                sdk.piper.EnablePiper()
                sdk.set_joint_positions(
                    [hold_position[f"joint_{index}.pos"] for index in range(7)]
                )
                sdk.piper.MotionCtrl_2(
                    0x01,
                    0x01,
                    int(sdk.speed_percent),
                    0x00,
                )
            elif status.ctrl_mode == 1:
                if hold_position is not None:
                    sdk.set_joint_positions(
                        [hold_position[f"joint_{index}.pos"] for index in range(7)]
                    )
                healthy += 1
            else:
                raise RuntimeError(
                    "PIPER controller entered an unsupported mode during "
                    f"Home handoff: {status}"
                )
        except RuntimeError as exc:
            healthy = 0
            last_error = exc
        time.sleep(1.0 / CONTROL_HZ)
    if healthy >= 5:
        return
    raise RuntimeError(
        "PIPER feedback did not remain healthy after motion enable; "
        f"last status: {last_error}"
    )


def adopt_already_enabled_motion_hold_current(
    sdk: PiperSDKInterface,
) -> dict[str, float]:
    """Adopt an already enabled arm without issuing a second quick-stop.

    The legacy ``enable_motion_hold_current()`` transition intentionally sends
    ``EmergencyStop(0x01)`` before enabling the drivers. That transition is
    only appropriate when all six drivers are currently disabled. On this
    PIPER-L, issuing it again while Home or deployment has already left the
    drivers enabled can make the drivers lose enable a few seconds later.

    This path mirrors the safe Home-to-deployment handoff: preload the measured
    target, change STANDBY to CAN/MOVE_J when necessary, mark this software
    connection motion-enabled, and latch the newly measured pose. It never
    sends EmergencyStop or DisablePiper. If an apparently-current CAN status
    later falls back to STANDBY, it may refresh EnablePiper without disabling
    the already-enabled drivers.
    """
    if sdk.motion_enabled:
        raise RuntimeError(
            "home connection is already marked motion-enabled; refusing an "
            "ambiguous second handoff"
        )

    status = sdk.piper.GetArmStatus().arm_status
    if status.err_code or status.arm_status:
        raise RuntimeError(
            "enabled-arm Home handoff requires NORMAL controller state with "
            f"error code 0; got {status}"
        )
    if status.ctrl_mode not in (0, 1):
        raise RuntimeError(
            "enabled-arm Home handoff only accepts STANDBY(0) or CAN control(1); "
            f"got ctrl_mode={status.ctrl_mode}"
        )

    motors = verify_health(sdk, require_enabled=False, check_controller=False)
    enabled = [bool(motor["driver_enabled"]) for motor in motors]
    if not all(enabled):
        raise RuntimeError(
            "enabled-arm Home handoff requires all six drivers enabled; "
            f"driver_enabled={enabled}"
        )

    raw_hold = sdk.get_status()
    held_position = {
        key: float(value) for key, value in sdk.get_status_normalized().items()
    }
    try:
        primer = getattr(sdk, "_prime_measured_raw_target", None)
        if primer is None:
            joints = [int(raw_hold[f"joint_{index}.pos"]) for index in range(6)]
            sdk.piper.JointCtrl(*joints)
            sdk.piper.GripperCtrl(
                int(raw_hold["joint_6.pos"]),
                1000,
                0x01,
                0,
            )
        else:
            primer(raw_hold)

        if status.ctrl_mode == 0:
            # A disconnected process can leave all six drivers enabled while
            # the controller itself falls back to STANDBY. MotionCtrl_2 alone
            # is accepted only intermittently in that state. Try it once
            # without touching enable; if feedback remains STANDBY, refresh
            # EnablePiper and retry. This never emits a quick-stop.
            sdk.piper.MotionCtrl_2(
                0x01,
                0x01,
                int(sdk.speed_percent),
                0x00,
            )
            time.sleep(0.1)
            transition_status = sdk.piper.GetArmStatus().arm_status
            if transition_status.ctrl_mode == 0:
                deadline = time.time() + 2.0
                while time.time() <= deadline:
                    sdk.piper.EnablePiper()
                    if primer is None:
                        sdk.piper.JointCtrl(
                            *[
                                int(raw_hold[f"joint_{index}.pos"])
                                for index in range(6)
                            ]
                        )
                        sdk.piper.GripperCtrl(
                            int(raw_hold["joint_6.pos"]),
                            1000,
                            0x01,
                            0,
                        )
                    else:
                        primer(raw_hold)
                    sdk.piper.MotionCtrl_2(
                        0x01,
                        0x01,
                        int(sdk.speed_percent),
                        0x00,
                    )
                    time.sleep(0.1)
                    transition_status = sdk.piper.GetArmStatus().arm_status
                    if transition_status.ctrl_mode == 1:
                        break
                    if transition_status.ctrl_mode != 0:
                        raise RuntimeError(
                            "Home handoff left STANDBY but entered an unsafe "
                            f"control mode: {transition_status}"
                        )
                else:
                    raise RuntimeError(
                        "Home handoff could not promote STANDBY to CAN control "
                        "within 2 seconds without quick-stop"
                    )
            elif transition_status.ctrl_mode != 1:
                raise RuntimeError(
                    "Home handoff entered an unsafe control mode after "
                    f"MotionCtrl_2: {transition_status}"
                )

        sdk.motion_enabled = True
        latch_current_pose(sdk)
        verifier = getattr(sdk, "_verify_controller_accepts_motion", None)
        if verifier is not None:
            verifier()
        wait_for_stable_health(sdk, hold_position=held_position)
    except BaseException:
        # Keep the software flag truthful. Do not send a stop/disable command:
        # hardware may still be enabled and supporting its own weight.
        sdk.motion_enabled = False
        raise
    return held_position


def enable_disabled_motion_hold_current(
    sdk: PiperSDKInterface,
) -> dict[str, float]:
    """Enable a fully disabled arm while holding feedback, without quick-stop.

    This is the recovery companion to
    :func:`adopt_already_enabled_motion_hold_current`. It preloads the measured
    raw target before and after ``EnablePiper`` so an old firmware target cannot
    replay. Unlike the legacy helper, it never sends ``EmergencyStop(0x01)``.
    If the controller is already blocked, it may send only the documented
    resume command ``EmergencyStop(0x02)`` before enabling.
    """
    if sdk.motion_enabled:
        raise RuntimeError(
            "home connection is already marked motion-enabled; refusing an "
            "ambiguous enable"
        )

    status = sdk.piper.GetArmStatus().arm_status
    if status.err_code or status.arm_status not in (0, 1):
        raise RuntimeError(
            "disabled-arm Home enable requires a recoverable controller state; "
            f"got {status}"
        )
    motors = verify_health(sdk, require_enabled=False, check_controller=False)
    enabled = [bool(motor["driver_enabled"]) for motor in motors]
    if any(enabled):
        raise RuntimeError(
            "disabled-arm Home enable requires all six drivers disabled; "
            f"driver_enabled={enabled}"
        )

    raw_hold = sdk.get_status()
    held_position = {
        key: float(value) for key, value in sdk.get_status_normalized().items()
    }
    primer = getattr(sdk, "_prime_measured_raw_target", None)

    try:
        if primer is None:
            joints = [int(raw_hold[f"joint_{index}.pos"]) for index in range(6)]
            sdk.piper.JointCtrl(*joints)
            sdk.piper.GripperCtrl(
                int(raw_hold["joint_6.pos"]),
                1000,
                0x01,
                0,
            )
        else:
            primer(raw_hold)

        resume = getattr(sdk, "_resume_controller_if_needed", None)
        if resume is not None:
            resume()
        elif (
            status.arm_status == 1
            or status.ctrl_mode == 2
            or status.motion_status != 0
        ):
            sdk.piper.EmergencyStop(0x02)
            time.sleep(0.1)

        deadline = time.time() + 10.0
        attempts = 0
        while not sdk.piper.EnablePiper():
            attempts += 1
            if time.time() > deadline:
                voltages = [
                    float(motor["voltage_v"])
                    for motor in sdk.get_motor_faults()
                ]
                raise RuntimeError(
                    "EnablePiper did not succeed within 10 seconds "
                    f"after {attempts} retries; motor_voltage_v={voltages}"
                )
            time.sleep(0.01)

        if primer is None:
            joints = [int(raw_hold[f"joint_{index}.pos"]) for index in range(6)]
            sdk.piper.JointCtrl(*joints)
            sdk.piper.GripperCtrl(
                int(raw_hold["joint_6.pos"]),
                1000,
                0x01,
                0,
            )
        else:
            primer(raw_hold)
        sdk.piper.MotionCtrl_2(
            0x01,
            0x01,
            int(sdk.speed_percent),
            0x00,
        )
        sdk.motion_enabled = True
        latch_current_pose(sdk)
        verifier = getattr(sdk, "_verify_controller_accepts_motion", None)
        if verifier is not None:
            verifier()
        wait_for_stable_health(sdk, hold_position=held_position)
    except BaseException:
        # Never quick-stop or disable automatically on this failure path. The
        # user may be physically supporting an arm whose drivers did enable.
        sdk.motion_enabled = False
        raise
    return held_position


def run_home(
    sdk: PiperSDKInterface,
    goal: PhysicalPose,
    args: argparse.Namespace,
    log: JsonlLogger,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
) -> None:
    started = time.perf_counter()
    deadline = started + args.home_timeout_s
    start, start_report = validate_start_pose(
        corrected_feedback_pose(sdk, args.j2_feedback_offset_deg),
        args,
        joint_limits,
        gripper_stroke_mm,
    )
    duration = quintic_joint_duration(
        start.joints_deg,
        goal.joints_deg,
        args.home_velocity_deg_s,
        args.home_acceleration_deg_s2,
    )
    initial_joint_error = max(
        abs(a - b) for a, b in zip(start.joints_deg, goal.joints_deg, strict=True)
    )
    initial_gripper_error = abs(start.gripper_mm - goal.gripper_mm)
    log.write(
        {
            "event": "home_start",
            "start_joints_deg": list(start.joints_deg),
            "goal_joints_deg": list(goal.joints_deg),
            "start_gripper_mm": start.gripper_mm,
            "goal_gripper_mm": goal.gripper_mm,
            "planned_joint_duration_s": duration,
            "initial_max_joint_error_deg": initial_joint_error,
            "initial_gripper_error_mm": initial_gripper_error,
            "start_feedback_validation": start_report,
            "command_sent": False,
        }
    )
    wait_for_stable_health(sdk)

    if initial_joint_error > args.home_joint_tolerance_deg:
        stage_started = time.perf_counter()
        tick = 0
        while True:
            tick_started = time.perf_counter()
            if tick_started > deadline:
                raise RuntimeError("homing timed out during joint return")
            measured = corrected_feedback_pose(sdk, args.j2_feedback_offset_deg)
            verify_health(sdk, require_enabled=True, check_controller=False)
            elapsed = tick_started - stage_started
            blend = quintic_blend(elapsed / duration)
            target = PhysicalPose(
                tuple(
                    a + (b - a) * blend
                    for a, b in zip(start.joints_deg, goal.joints_deg, strict=True)
                ),
                start.gripper_mm,
            )
            report = bounded_send(
                sdk,
                physical_pose_to_normalized(target, joint_limits, gripper_stroke_mm),
                max_arm_relative_target=args.max_relative_target,
                max_gripper_relative_target=args.max_gripper_relative_target,
            )
            joint_error = max(
                abs(a - b)
                for a, b in zip(measured.joints_deg, goal.joints_deg, strict=True)
            )
            if tick % 30 == 0:
                log.write(
                    {
                        "event": "home_progress",
                        "stage": "joints",
                        "elapsed_s": tick_started - started,
                        "observed_joints_deg": list(measured.joints_deg),
                        "max_joint_error_deg": joint_error,
                        "command": report,
                        "command_sent": True,
                    }
                )
            if elapsed >= duration and joint_error <= args.home_joint_tolerance_deg:
                break
            tick += 1
            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - tick_started)))

    measured = corrected_feedback_pose(sdk, args.j2_feedback_offset_deg)
    grip_start = measured.gripper_mm
    gripper_duration = max(
        1.0 / CONTROL_HZ,
        1.875 * abs(grip_start - goal.gripper_mm) / args.home_gripper_velocity_mm_s,
    )
    if abs(grip_start - goal.gripper_mm) > args.home_gripper_tolerance_mm:
        stage_started = time.perf_counter()
        tick = 0
        while True:
            tick_started = time.perf_counter()
            if tick_started > deadline:
                raise RuntimeError("homing timed out during gripper restore")
            measured = corrected_feedback_pose(sdk, args.j2_feedback_offset_deg)
            verify_health(sdk, require_enabled=True, check_controller=False)
            elapsed = tick_started - stage_started
            blend = quintic_blend(elapsed / gripper_duration)
            target = PhysicalPose(
                goal.joints_deg,
                grip_start + (goal.gripper_mm - grip_start) * blend,
            )
            report = bounded_send(
                sdk,
                physical_pose_to_normalized(target, joint_limits, gripper_stroke_mm),
                max_arm_relative_target=args.max_relative_target,
                max_gripper_relative_target=args.max_gripper_relative_target,
            )
            joint_error = max(
                abs(a - b)
                for a, b in zip(measured.joints_deg, goal.joints_deg, strict=True)
            )
            gripper_error = abs(measured.gripper_mm - goal.gripper_mm)
            if tick % 30 == 0:
                log.write(
                    {
                        "event": "home_progress",
                        "stage": "gripper",
                        "elapsed_s": tick_started - started,
                        "observed_gripper_mm": measured.gripper_mm,
                        "gripper_error_mm": gripper_error,
                        "max_joint_error_deg": joint_error,
                        "command": report,
                        "command_sent": True,
                    }
                )
            if (
                elapsed >= gripper_duration
                and joint_error <= args.home_joint_tolerance_deg
                and gripper_error <= args.home_gripper_tolerance_mm
            ):
                break
            tick += 1
            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - tick_started)))

    hold_started = time.perf_counter()
    goal_normalized = physical_pose_to_normalized(goal, joint_limits, gripper_stroke_mm)
    while time.perf_counter() - hold_started < args.home_hold_s:
        tick_started = time.perf_counter()
        verify_health(sdk, require_enabled=True, check_controller=False)
        bounded_send(
            sdk,
            goal_normalized,
            max_arm_relative_target=args.max_relative_target,
            max_gripper_relative_target=args.max_gripper_relative_target,
        )
        time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - tick_started)))

    final_pose = corrected_feedback_pose(sdk, args.j2_feedback_offset_deg)
    final_joint_error = max(
        abs(a - b) for a, b in zip(final_pose.joints_deg, goal.joints_deg, strict=True)
    )
    final_gripper_error = abs(final_pose.gripper_mm - goal.gripper_mm)
    wait_for_stable_health(sdk)
    if (
        final_joint_error > args.home_joint_tolerance_deg
        or final_gripper_error > args.home_gripper_tolerance_mm
    ):
        raise RuntimeError(
            "home hold drifted outside tolerance: "
            f"joint={final_joint_error:.3f} deg, "
            f"gripper={final_gripper_error:.3f} mm"
        )
    log.write(
        {
            "event": "home_complete",
            "elapsed_s": time.perf_counter() - started,
            "final_joints_deg": list(final_pose.joints_deg),
            "final_gripper_mm": final_pose.gripper_mm,
            "max_joint_error_deg": final_joint_error,
            "gripper_error_mm": final_gripper_error,
            "command_sent": True,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "execute"), default="check")
    parser.add_argument("--can-port", required=True)
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--initial-pose-json", type=Path, default=DEFAULT_INITIAL_POSE)
    parser.add_argument(
        "--joint-zero-offset-json",
        type=Path,
        default=DEFAULT_JOINT_ZERO_OFFSET,
        help=(
            "audited full-trajectory hardware-minus-model offset; an active "
            "mapping requires passing multi-pose validation"
        ),
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--max-relative-target", type=float, default=0.5)
    parser.add_argument("--max-gripper-relative-target", type=float, default=100.0)
    parser.add_argument(
        "--gripper-effort",
        type=int,
        default=5000,
        help="PIPER gripper torque in 0.001 N*m SDK units (0..5000)",
    )
    parser.add_argument("--home-timeout-s", type=float, default=60.0)
    parser.add_argument("--home-velocity-deg-s", type=float, default=5.0)
    parser.add_argument("--home-acceleration-deg-s2", type=float, default=10.0)
    parser.add_argument("--home-gripper-velocity-mm-s", type=float, default=10.0)
    parser.add_argument("--home-joint-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--home-gripper-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--home-start-limit-margin-deg", type=float, default=10.0)
    parser.add_argument("--home-hold-s", type=float, default=0.5)
    parser.add_argument(
        "--j2-feedback-offset-deg",
        type=float,
        default=0.0,
        help=(
            "value added only to firmware J2 feedback used by Home checks and "
            "tracking; Home command targets are unchanged (use -6 when the "
            "reported J2 is 6 deg too high)"
        ),
    )
    parser.add_argument(
        "--j2-command-offset-deg",
        type=float,
        default=0.0,
        help=(
            "hardware-minus-model J2 offset added to the saved Home target "
            "before it is sent to firmware; raw firmware feedback remains "
            "unchanged for tracking (use 6 when 42.9 model deg is 48.9 firmware deg)"
        ),
    )
    parser.add_argument(
        "--j3-command-offset-deg",
        type=float,
        default=0.0,
        help=(
            "hardware-minus-model J3 offset added to the saved Home target "
            "before it is sent to firmware; raw firmware feedback remains "
            "unchanged for tracking"
        ),
    )
    parser.add_argument(
        "--j5-command-offset-deg",
        type=float,
        default=0.0,
        help=(
            "hardware-minus-model J5 offset added to the saved Home target "
            "before it is sent to firmware; raw firmware feedback remains "
            "unchanged for tracking"
        ),
    )
    parser.add_argument("--allow-motion", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "execute" and not args.allow_motion:
        parser.error("execute mode requires --allow-motion")
    if args.mode != "execute" and args.allow_motion:
        parser.error("--allow-motion is only valid with --mode execute")
    if not args.calibration_json.is_file():
        parser.error(f"calibration does not exist: {args.calibration_json}")
    if not args.initial_pose_json.is_file():
        parser.error(f"initial pose does not exist: {args.initial_pose_json}")
    if not args.joint_zero_offset_json.is_file():
        parser.error(
            "joint zero offset does not exist: "
            f"{args.joint_zero_offset_json}"
        )
    if not 1 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [1,100]")
    if not 0 <= args.gripper_effort <= 5000:
        parser.error("--gripper-effort must be in [0,5000]")
    for option in (
        "max_relative_target",
        "max_gripper_relative_target",
        "home_timeout_s",
        "home_velocity_deg_s",
        "home_acceleration_deg_s2",
        "home_gripper_velocity_mm_s",
        "home_joint_tolerance_deg",
        "home_gripper_tolerance_mm",
        "home_start_limit_margin_deg",
        "home_hold_s",
    ):
        value = getattr(args, option)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{option.replace('_', '-')} must be finite and positive")
    if args.max_gripper_relative_target > 100:
        parser.error("--max-gripper-relative-target must not exceed 100")
    if (
        not math.isfinite(args.j2_feedback_offset_deg)
        or abs(args.j2_feedback_offset_deg) > 15.0
    ):
        parser.error("--j2-feedback-offset-deg must be finite and within [-15,15]")
    if (
        not math.isfinite(args.j2_command_offset_deg)
        or abs(args.j2_command_offset_deg) > 15.0
    ):
        parser.error("--j2-command-offset-deg must be finite and within [-15,15]")
    if (
        not math.isfinite(args.j3_command_offset_deg)
        or abs(args.j3_command_offset_deg) > 15.0
    ):
        parser.error("--j3-command-offset-deg must be finite and within [-15,15]")
    if (
        not math.isfinite(args.j5_command_offset_deg)
        or abs(args.j5_command_offset_deg) > 15.0
    ):
        parser.error("--j5-command-offset-deg must be finite and within [-15,15]")
    if args.j2_feedback_offset_deg != 0.0 and args.j2_command_offset_deg != 0.0:
        parser.error(
            "do not combine feedback-side and command-side J2 offsets; "
            "for the current arm use --j2-feedback-offset-deg 0 and "
            "--j2-command-offset-deg 6"
        )
    return args


def main() -> None:
    args = parse_args()
    joint_limits, gripper_stroke_mm = load_calibration(args.calibration_json)
    hardware_goal = load_initial_pose(
        args.initial_pose_json, joint_limits, gripper_stroke_mm
    )
    zero_offset, zero_status, zero_validation = load_joint_zero_offset(
        args.joint_zero_offset_json
    )
    model_goal = PhysicalPose(
        tuple(
            float(value)
            for value in np.asarray(hardware_goal.joints_deg, dtype=np.float64)
            + zero_offset
        ),
        hardware_goal.gripper_mm,
    )
    goal = apply_j2_command_offset(model_goal, args.j2_command_offset_deg)
    goal = apply_j3_command_offset(goal, args.j3_command_offset_deg)
    goal = apply_j5_command_offset(goal, args.j5_command_offset_deg)
    for index, (value, (low, high)) in enumerate(
        zip(goal.joints_deg, joint_limits, strict=True)
    ):
        if not low <= value <= high:
            raise RuntimeError(
                f"zero-offset Home joint {index + 1}={value} is outside "
                f"hardware limits [{low},{high}]"
            )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = JsonlLogger(args.log_dir / f"home_{args.mode}_{timestamp}.jsonl")
    sdk: PiperSDKInterface | None = None
    motion_enabled = False
    try:
        sdk = PiperSDKInterface(
            port=args.can_port,
            gripper_stroke_mm=gripper_stroke_mm,
            gripper_effort=args.gripper_effort,
            speed_percent=args.speed_percent,
            joint_limits_deg=joint_limits,
            enable_motion=False,
        )
        raw_feedback = sdk.get_status()
        current_firmware = raw_feedback_pose(raw_feedback)
        current = apply_j2_feedback_offset(
            current_firmware,
            args.j2_feedback_offset_deg,
        )
        start, start_report = validate_start_pose(
            current,
            args,
            joint_limits,
            gripper_stroke_mm,
        )
        motors = verify_health(
            sdk,
            require_enabled=False,
            check_controller=False,
        )
        planned_duration = quintic_joint_duration(
            start.joints_deg,
            goal.joints_deg,
            args.home_velocity_deg_s,
            args.home_acceleration_deg_s2,
        )
        preflight = {
            "event": "home_preflight",
            "mode": args.mode,
            "can_port": args.can_port,
            "current_firmware_joints_deg": list(current_firmware.joints_deg),
            "current_joints_deg": list(current.joints_deg),
            "current_gripper_mm": current.gripper_mm,
            "j2_feedback_offset_deg": args.j2_feedback_offset_deg,
            "j2_command_offset_deg": args.j2_command_offset_deg,
            "j3_command_offset_deg": args.j3_command_offset_deg,
            "j5_command_offset_deg": args.j5_command_offset_deg,
            "model_goal_joints_deg": list(model_goal.joints_deg),
            "firmware_goal_joints_deg": list(goal.joints_deg),
            "goal_joints_deg": list(goal.joints_deg),
            "goal_gripper_mm": goal.gripper_mm,
            "gripper_effort_sdk_units": args.gripper_effort,
            "gripper_effort_nm": args.gripper_effort * 0.001,
            "saved_hardware_goal_joints_deg": list(hardware_goal.joints_deg),
            "joint_zero_offset_json": str(args.joint_zero_offset_json),
            "joint_zero_offset_status": zero_status,
            "hardware_minus_model_zero_offset_deg": zero_offset.tolist(),
            "joint_zero_offset_multi_pose_validation": zero_validation,
            "planned_joint_duration_s": planned_duration,
            "start_feedback_validation": start_report,
            "motor_voltage_v": [float(motor["voltage_v"]) for motor in motors],
            "motor_enabled": [bool(motor["driver_enabled"]) for motor in motors],
            "controller_status": str(sdk.piper.GetArmStatus().arm_status),
            "checkpoint_loaded": False,
            "policy_constructed": False,
            "camera_opened": False,
            "command_sent": False,
        }
        log.write(preflight)
        print(json.dumps(preflight, indent=2))
        require_live_motor_feedback(
            raw_feedback,
            motors,
            context="Home preflight",
        )
        if args.mode == "check":
            print(f"CHECK PASSED: observation-only; no motion command was sent. log={log.path}")
            return

        enabled = [bool(motor["driver_enabled"]) for motor in motors]
        if all(enabled):
            held_position = adopt_already_enabled_motion_hold_current(sdk)
            transition = "adopt_already_enabled_without_quick_stop"
        elif any(enabled):
            raise RuntimeError(
                "refusing Home from a partially enabled arm; "
                f"driver_enabled={enabled}"
            )
        else:
            held_position = enable_disabled_motion_hold_current(sdk)
            transition = "enable_from_all_disabled_without_quick_stop"
        motion_enabled = True
        log.write(
            {
                "event": "motion_enable",
                "transition": transition,
                "held_position": held_position,
                "command_sent": True,
            }
        )
        run_home(sdk, goal, args, log, joint_limits, gripper_stroke_mm)
        held = latch_current_pose(sdk)
        log.write(
            {
                "event": "current_pose_latched",
                "reason": "normal_completion",
                "normalized_state": held,
                "command_sent": True,
            }
        )
        print(f"HOME COMPLETE: target reached and latched. log={log.path}")
    except BaseException as error:
        log.write(
            {
                "event": "home_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "command_sent": False,
            }
        )
        if sdk is not None and motion_enabled:
            try:
                held = latch_current_pose(sdk)
                log.write(
                    {
                        "event": "current_pose_latched",
                        "reason": "exception_abort",
                        "normalized_state": held,
                        "command_sent": True,
                    }
                )
            except BaseException as hold_error:
                log.write(
                    {
                        "event": "current_pose_latch_failed",
                        "reason": "exception_abort",
                        "error": repr(hold_error),
                        "command_sent": False,
                    }
                )
        raise
    finally:
        if sdk is not None:
            sdk.disconnect(home_first=False, disable_torque=False)
        log.close()


if __name__ == "__main__":
    main()
