#!/usr/bin/env python3
"""Model-free tests for the Home-to-deployment motor handoff."""

from __future__ import annotations

import ast
import types
import unittest
from pathlib import Path
from typing import Any

import numpy as np


_SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robot/deploy_assembly_new1_hpt_joint.py"
)
_SOURCE = _SOURCE_PATH.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)
_FUNCTION_NAMES = {
    "observation_to_driver_state",
    "bind_runtime_calibration",
    "verify_homing_health",
    "require_live_robot_feedback",
    "require_deployment_pose_inside_limits",
    "wait_for_stable_homing_health",
    "latch_current_raw_position",
    "adopt_already_enabled_motion_hold_current",
}
_FUNCTIONS = [
    node
    for node in _TREE.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name in _FUNCTION_NAMES
]
assert {node.name for node in _FUNCTIONS} == _FUNCTION_NAMES


class _FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        self.now += 0.05
        return self.now

    def time(self) -> float:
        self.now += 0.05
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


_namespace: dict[str, Any] = {
    "Any": Any,
    "Piper": Any,
    "math": __import__("math"),
    "np": np,
    "time": _FakeTime(),
    "precise_sleep": lambda _seconds: None,
}
exec(
    compile(
        ast.Module(body=_FUNCTIONS, type_ignores=[]),
        str(_SOURCE_PATH),
        "exec",
    ),
    _namespace,
)
adopt = _namespace["adopt_already_enabled_motion_hold_current"]
bind_runtime_calibration = _namespace["bind_runtime_calibration"]
require_live_robot_feedback = _namespace["require_live_robot_feedback"]
require_deployment_pose_inside_limits = _namespace[
    "require_deployment_pose_inside_limits"
]


_HOME_SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "robot/home_assembly_new1.py"
)
_HOME_SOURCE = _HOME_SOURCE_PATH.read_text(encoding="utf-8")
_HOME_TREE = ast.parse(_HOME_SOURCE)
_HOME_FUNCTION_NAMES = {
    "normalized_feedback",
    "latch_current_pose",
    "verify_health",
    "require_live_motor_feedback",
    "wait_for_stable_health",
    "adopt_already_enabled_motion_hold_current",
    "enable_disabled_motion_hold_current",
}
_HOME_FUNCTIONS = [
    node
    for node in _HOME_TREE.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name in _HOME_FUNCTION_NAMES
]
assert {node.name for node in _HOME_FUNCTIONS} == _HOME_FUNCTION_NAMES
_home_namespace: dict[str, Any] = {
    "Any": Any,
    "PiperSDKInterface": Any,
    "np": np,
    "time": _FakeTime(),
    "CONTROL_HZ": 30.0,
}
exec(
    compile(
        ast.Module(body=_HOME_FUNCTIONS, type_ignores=[]),
        str(_HOME_SOURCE_PATH),
        "exec",
    ),
    _home_namespace,
)
adopt_home = _home_namespace["adopt_already_enabled_motion_hold_current"]
enable_home = _home_namespace["enable_disabled_motion_hold_current"]
require_live_motor_feedback = _home_namespace["require_live_motor_feedback"]


def _healthy_motor(enabled: bool) -> dict[str, Any]:
    return {
        "voltage_too_low": False,
        "motor_overheating": False,
        "driver_overcurrent": False,
        "driver_overheating": False,
        "collision_protection_triggered": False,
        "driver_error": False,
        "stall_protection_triggered": False,
        "driver_enabled": enabled,
        "voltage_v": 23.0,
    }


class FeedbackAvailabilityTests(unittest.TestCase):
    def test_deployment_rejects_all_zero_motor_voltage_cache(self):
        robot = types.SimpleNamespace(
            sdk=types.SimpleNamespace(
                piper=_FakeController(ctrl_mode=0),
                get_motor_faults=lambda: [
                    {**_healthy_motor(False), "voltage_v": 0.0} for _ in range(6)
                ],
            )
        )
        raw = {f"joint_{index}.pos": 0.0 for index in range(7)}
        with self.assertRaisesRegex(RuntimeError, "no live PIPER motor feedback"):
            require_live_robot_feedback(robot, raw, context="test preflight")

    def test_home_accepts_nonzero_voltage_telemetry(self):
        raw = {f"joint_{index}.pos": 0.0 for index in range(7)}
        require_live_motor_feedback(
            raw,
            [_healthy_motor(False) for _ in range(6)],
            context="test preflight",
        )

    def test_live_dry_run_requires_home_for_recovery_only_joint_pose(self):
        with self.assertRaisesRegex(RuntimeError, "requires model-free Home"):
            require_deployment_pose_inside_limits(
                {"recovery_joints": [2]},
                execute=False,
                home_at_start=False,
            )

    def test_execute_may_recover_only_when_home_at_start_is_enabled(self):
        require_deployment_pose_inside_limits(
            {"recovery_joints": [2]},
            execute=True,
            home_at_start=True,
        )
        with self.assertRaisesRegex(RuntimeError, "requires model-free Home"):
            require_deployment_pose_inside_limits(
                {"recovery_joints": [2]},
                execute=True,
                home_at_start=False,
            )


class _FakeController:
    def __init__(self, ctrl_mode: int = 0) -> None:
        self.dangerous_calls: list[str] = []
        self.motion_control_calls: list[tuple[int, int, int, int]] = []
        self.ignore_next_motion_control = False
        self.status_reads = 0
        self.fallback_at_status_read: int | None = None
        self.arm_status = types.SimpleNamespace(
            err_code=0,
            arm_status=0,
            ctrl_mode=ctrl_mode,
            motion_status=0,
        )
        self.sdk = None

    def GetArmStatus(self):
        self.status_reads += 1
        if self.status_reads == self.fallback_at_status_read:
            self.arm_status.ctrl_mode = 0
        return types.SimpleNamespace(arm_status=self.arm_status)

    def EmergencyStop(self, *_args):
        self.dangerous_calls.append("EmergencyStop")

    def EnablePiper(self, *_args):
        self.dangerous_calls.append("EnablePiper")
        if self.sdk is not None:
            for motor in self.sdk.motors:
                motor["driver_enabled"] = True
        return True

    def DisablePiper(self, *_args):
        self.dangerous_calls.append("DisablePiper")
        return True

    def MotionCtrl_2(self, *args):
        self.motion_control_calls.append(tuple(args))
        if self.ignore_next_motion_control:
            self.ignore_next_motion_control = False
            return
        self.arm_status.ctrl_mode = 1


class _FakeSdk:
    def __init__(self, enabled: bool = True, ctrl_mode: int = 0) -> None:
        self.motion_enabled = False
        self.piper = _FakeController(ctrl_mode=ctrl_mode)
        self.piper.sdk = self
        self.speed_percent = 5
        self.min_pos = [-1510.0, 9.0, -1701.0, -1242.0, -850.0, -1684.0]
        self.max_pos = [1510.0, 1860.0, -10.0, 1249.0, 877.0, 1684.0]
        self.gripper_stroke_mm = 102.8
        self.motors = [_healthy_motor(enabled) for _ in range(6)]
        self.position_commands: list[list[float]] = []
        self.raw_primes: list[dict[str, float]] = []

    def get_motor_faults(self):
        return self.motors

    def get_status_normalized(self):
        return {f"joint_{index}.pos": float(index) for index in range(7)}

    def get_status(self):
        return {
            **{f"joint_{index}.pos": 0.0 for index in range(6)},
            "joint_6.pos": 50_000.0,
        }

    def set_joint_positions(self, positions):
        if not self.motion_enabled:
            raise RuntimeError("test SDK is observation-only")
        self.position_commands.append(list(positions))

    def _prime_measured_raw_target(self, raw):
        self.raw_primes.append(dict(raw))

    def _verify_controller_accepts_motion(self):
        return None

    def _resume_controller_if_needed(self):
        return False


class MotionHandoffTest(unittest.TestCase):
    def test_runtime_calibration_replaces_stale_cache_before_motion(self):
        sdk = _FakeSdk(enabled=True)
        sdk.min_pos = [-1546.0, -21.0, -1734.0, -1354.0, -888.0, -1706.0, 0.0]
        sdk.max_pos = [1500.0, 1911.0, 20.0, 1198.0, 899.0, 1723.0, 100.0]
        raw = {
            "joint_0.pos": 6411.0,
            "joint_1.pos": 42823.0,
            "joint_2.pos": -27453.0,
            "joint_3.pos": -11153.0,
            "joint_4.pos": 41765.0,
            "joint_5.pos": 7867.0,
            "joint_6.pos": 99900.0,
        }
        limits = [
            (-151.0, 151.0),
            (0.9, 186.0),
            (-170.1, -1.0),
            (-124.2, 124.9),
            (-85.0, 87.7),
            (-168.4, 168.4),
        ]

        report = bind_runtime_calibration(
            types.SimpleNamespace(sdk=sdk), raw, limits, 102.8
        )

        self.assertTrue(report["cache_override_detected"])
        self.assertEqual(sdk.min_pos, [-1510.0, 9.0, -1701.0, -1242.0, -850.0, -1684.0, 0.0])
        self.assertEqual(sdk.max_pos, [1510.0, 1860.0, -10.0, 1249.0, 877.0, 1684.0, 100.0])
        self.assertLessEqual(report["max_policy_sdk_normalized_error"], 1e-9)

    def test_adopts_enabled_hardware_without_stop_or_enable_calls(self):
        robot = types.SimpleNamespace(sdk=_FakeSdk(enabled=True, ctrl_mode=0))
        held = adopt(robot)

        self.assertTrue(robot.sdk.motion_enabled)
        self.assertEqual(len(robot.sdk.raw_primes), 1)
        self.assertGreaterEqual(len(robot.sdk.position_commands), 2)
        self.assertEqual(robot.sdk.piper.dangerous_calls, [])
        self.assertEqual(
            robot.sdk.piper.motion_control_calls,
            [(0x01, 0x01, 5, 0x00)],
        )
        self.assertEqual(held["joint_0.pos"], 0.0)
        self.assertEqual(held["joint_6.pos"], 6.0)

    def test_already_in_can_control_does_not_resend_motion_mode(self):
        robot = types.SimpleNamespace(sdk=_FakeSdk(enabled=True, ctrl_mode=1))
        adopt(robot)

        self.assertTrue(robot.sdk.motion_enabled)
        self.assertEqual(robot.sdk.piper.motion_control_calls, [])
        self.assertEqual(robot.sdk.piper.dangerous_calls, [])

    def test_refuses_when_home_has_not_left_all_motors_enabled(self):
        robot = types.SimpleNamespace(sdk=_FakeSdk(enabled=False))
        with self.assertRaisesRegex(RuntimeError, "requires all six motors"):
            adopt(robot)

        self.assertFalse(robot.sdk.motion_enabled)
        self.assertEqual(robot.sdk.position_commands, [])
        self.assertEqual(robot.sdk.piper.dangerous_calls, [])

    def test_home_adopts_enabled_hardware_without_second_quick_stop(self):
        sdk = _FakeSdk(enabled=True, ctrl_mode=0)
        held = adopt_home(sdk)

        self.assertTrue(sdk.motion_enabled)
        self.assertEqual(len(sdk.raw_primes), 1)
        self.assertGreaterEqual(len(sdk.position_commands), 2)
        self.assertEqual(sdk.piper.dangerous_calls, [])
        self.assertEqual(
            sdk.piper.motion_control_calls,
            [(0x01, 0x01, 5, 0x00)],
        )
        self.assertEqual(held["joint_0.pos"], 0.0)
        self.assertEqual(held["joint_6.pos"], 6.0)

    def test_home_enabled_handoff_refuses_partial_enable(self):
        sdk = _FakeSdk(enabled=True, ctrl_mode=0)
        sdk.motors[3]["driver_enabled"] = False

        with self.assertRaisesRegex(RuntimeError, "all six drivers enabled"):
            adopt_home(sdk)

        self.assertFalse(sdk.motion_enabled)
        self.assertEqual(sdk.position_commands, [])
        self.assertEqual(sdk.piper.dangerous_calls, [])

    def test_home_enables_fully_disabled_hardware_without_quick_stop(self):
        sdk = _FakeSdk(enabled=False, ctrl_mode=0)
        held = enable_home(sdk)

        self.assertTrue(sdk.motion_enabled)
        self.assertEqual(len(sdk.raw_primes), 2)
        self.assertGreaterEqual(len(sdk.position_commands), 2)
        self.assertEqual(sdk.piper.dangerous_calls, ["EnablePiper"])
        self.assertNotIn("EmergencyStop", sdk.piper.dangerous_calls)
        self.assertNotIn("DisablePiper", sdk.piper.dangerous_calls)
        self.assertEqual(
            sdk.piper.motion_control_calls,
            [(0x01, 0x01, 5, 0x00)],
        )
        self.assertEqual(held["joint_0.pos"], 0.0)
        self.assertEqual(held["joint_6.pos"], 6.0)

    def test_home_refreshes_enable_when_standby_ignores_first_mode_command(self):
        sdk = _FakeSdk(enabled=True, ctrl_mode=0)
        sdk.piper.ignore_next_motion_control = True

        adopt_home(sdk)

        self.assertTrue(sdk.motion_enabled)
        self.assertEqual(sdk.piper.dangerous_calls, ["EnablePiper"])
        self.assertEqual(len(sdk.piper.motion_control_calls), 2)
        self.assertNotIn("EmergencyStop", sdk.piper.dangerous_calls)
        self.assertNotIn("DisablePiper", sdk.piper.dangerous_calls)

    def test_deployment_refreshes_enable_when_standby_ignores_first_mode_command(self):
        robot = types.SimpleNamespace(sdk=_FakeSdk(enabled=True, ctrl_mode=0))
        robot.sdk.piper.ignore_next_motion_control = True

        adopt(robot)

        self.assertTrue(robot.sdk.motion_enabled)
        self.assertEqual(robot.sdk.piper.dangerous_calls, ["EnablePiper"])
        self.assertEqual(len(robot.sdk.piper.motion_control_calls), 2)
        self.assertNotIn("EmergencyStop", robot.sdk.piper.dangerous_calls)
        self.assertNotIn("DisablePiper", robot.sdk.piper.dangerous_calls)

    def test_home_recovers_delayed_standby_after_stale_can_sample(self):
        sdk = _FakeSdk(enabled=True, ctrl_mode=1)
        sdk.piper.fallback_at_status_read = 4

        adopt_home(sdk)

        self.assertTrue(sdk.motion_enabled)
        self.assertIn("EnablePiper", sdk.piper.dangerous_calls)
        self.assertGreaterEqual(len(sdk.piper.motion_control_calls), 1)
        self.assertNotIn("EmergencyStop", sdk.piper.dangerous_calls)
        self.assertNotIn("DisablePiper", sdk.piper.dangerous_calls)

    def test_deployment_recovers_delayed_standby_after_stale_can_sample(self):
        robot = types.SimpleNamespace(sdk=_FakeSdk(enabled=True, ctrl_mode=1))
        robot.sdk.piper.fallback_at_status_read = 8

        adopt(robot)

        self.assertTrue(robot.sdk.motion_enabled)
        self.assertIn("EnablePiper", robot.sdk.piper.dangerous_calls)
        self.assertGreaterEqual(len(robot.sdk.piper.motion_control_calls), 1)
        self.assertNotIn("EmergencyStop", robot.sdk.piper.dangerous_calls)
        self.assertNotIn("DisablePiper", robot.sdk.piper.dangerous_calls)


if __name__ == "__main__":
    unittest.main()
