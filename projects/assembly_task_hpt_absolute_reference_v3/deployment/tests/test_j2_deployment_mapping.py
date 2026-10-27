#!/usr/bin/env python3
from __future__ import annotations

import ast
import math
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = BUNDLE_ROOT / "robot/deploy_assembly_new1_hpt_joint.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
NAMES = {
    "InitialPose",
    "hardware_observation_for_model",
    "apply_j2_command_offset_to_hardware_action",
    "apply_j2_command_offset_to_initial_pose",
    "apply_j3_command_offset_to_hardware_action",
    "apply_j3_command_offset_to_initial_pose",
    "apply_j5_command_offset_to_hardware_action",
    "apply_j5_command_offset_to_initial_pose",
}
NODES = [
    node
    for node in TREE.body
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in NAMES
]
NS: dict[str, Any] = {
    "Any": Any,
    "math": math,
    "dataclass": dataclass,
    "J2_INDEX": 1,
    "J3_INDEX": 2,
    "J5_INDEX": 4,
}
exec(compile(ast.Module(body=NODES, type_ignores=[]), str(SOURCE), "exec"), NS)


class J2DeploymentMappingTest(unittest.TestCase):
    def test_proprio_uses_inverse_mapping_without_mutating_raw_feedback(self):
        observation = {
            **{f"joint_{index}.pos": float(index * 1000) for index in range(7)},
            "joint_1.pos": 48_900.0,
            "435if": object(),
        }
        converted = NS["hardware_observation_for_model"](observation, 6.0)
        self.assertEqual(observation["joint_1.pos"], 48_900.0)
        self.assertEqual(converted["joint_1.pos"], 42_900.0)
        self.assertIs(converted["435if"], observation["435if"])

    def test_action_uses_forward_mapping(self):
        limits = [(-151.0, 151.0), (3.0, 177.0)] + [(-180.0, 180.0)] * 4
        model_j2_deg = 42.9
        model_driver = (model_j2_deg - 3.0) / (177.0 - 3.0) * 200.0 - 100.0
        action = {f"joint_{index}.pos": 0.0 for index in range(7)}
        action["joint_1.pos"] = model_driver
        converted, report = NS["apply_j2_command_offset_to_hardware_action"](
            action, 6.0, limits
        )
        firmware_j2_deg = (
            3.0 + (converted["joint_1.pos"] + 100.0) / 200.0 * (177.0 - 3.0)
        )
        self.assertAlmostEqual(firmware_j2_deg, 48.9, places=9)
        self.assertAlmostEqual(report["model_j2_target_deg"], 42.9, places=9)
        self.assertAlmostEqual(report["firmware_j2_target_deg"], 48.9, places=9)
        self.assertEqual(action["joint_1.pos"], model_driver)

    def test_home_uses_forward_mapping(self):
        pose = NS["InitialPose"]((0.0, 42.9, 0.0, 0.0, 0.0, 0.0), 99.7)
        limits = [(-151.0, 151.0), (3.0, 177.0)] + [(-180.0, 180.0)] * 4
        converted = NS["apply_j2_command_offset_to_initial_pose"](
            pose, 6.0, limits
        )
        self.assertAlmostEqual(converted.joints_deg[1], 48.9, places=9)
        self.assertAlmostEqual(pose.joints_deg[1], 42.9, places=9)

    def test_shifted_action_outside_hardware_limits_is_rejected(self):
        limits = [(-151.0, 151.0), (3.0, 177.0)] + [(-180.0, 180.0)] * 4
        action = {f"joint_{index}.pos": 0.0 for index in range(7)}
        action["joint_1.pos"] = 100.0
        with self.assertRaisesRegex(RuntimeError, "exceed hardware calibration"):
            NS["apply_j2_command_offset_to_hardware_action"](action, 6.0, limits)

    def test_j3_uses_inverse_proprio_and_forward_action_mapping(self):
        observation = {
            **{f"joint_{index}.pos": float(index * 1000) for index in range(7)},
            "joint_2.pos": -25_500.0,
        }
        converted_observation = NS["hardware_observation_for_model"](
            observation, 0.0, 1.5, 0.0
        )
        self.assertEqual(observation["joint_2.pos"], -25_500.0)
        self.assertEqual(converted_observation["joint_2.pos"], -27_000.0)

        limits = [(-180.0, 180.0), (-180.0, 180.0), (-170.1, -1.0)] + [
            (-180.0, 180.0)
        ] * 3
        model_j3_deg = -27.0
        model_driver = (
            (model_j3_deg + 170.1) / (-1.0 + 170.1) * 200.0 - 100.0
        )
        action = {f"joint_{index}.pos": 0.0 for index in range(7)}
        action["joint_2.pos"] = model_driver
        converted_action, report = NS[
            "apply_j3_command_offset_to_hardware_action"
        ](action, 1.5, limits)
        firmware_j3_deg = (
            -170.1
            + (converted_action["joint_2.pos"] + 100.0) / 200.0
            * (-1.0 + 170.1)
        )
        self.assertAlmostEqual(firmware_j3_deg, -25.5, places=9)
        self.assertAlmostEqual(report["model_j3_target_deg"], -27.0, places=9)
        self.assertAlmostEqual(report["firmware_j3_target_deg"], -25.5, places=9)

    def test_j3_home_uses_forward_mapping(self):
        pose = NS["InitialPose"]((0.0, 0.0, -27.0, 0.0, 0.0, 0.0), 99.7)
        limits = [(-180.0, 180.0), (-180.0, 180.0), (-170.1, -1.0)] + [
            (-180.0, 180.0)
        ] * 3
        converted = NS["apply_j3_command_offset_to_initial_pose"](
            pose, 1.5, limits
        )
        self.assertAlmostEqual(converted.joints_deg[2], -25.5, places=9)
        self.assertAlmostEqual(pose.joints_deg[2], -27.0, places=9)

    def test_j5_uses_inverse_proprio_and_forward_action_mapping(self):
        observation = {
            **{f"joint_{index}.pos": float(index * 1000) for index in range(7)},
            "joint_4.pos": 43_800.0,
        }
        converted_observation = NS["hardware_observation_for_model"](
            observation, 0.0, 0.0, 2.0
        )
        self.assertEqual(observation["joint_4.pos"], 43_800.0)
        self.assertEqual(converted_observation["joint_4.pos"], 41_800.0)

        limits = [(-180.0, 180.0)] * 4 + [(-85.0, 87.7), (-180.0, 180.0)]
        model_j5_deg = 41.8
        model_driver = (model_j5_deg + 85.0) / (87.7 + 85.0) * 200.0 - 100.0
        action = {f"joint_{index}.pos": 0.0 for index in range(7)}
        action["joint_4.pos"] = model_driver
        converted_action, report = NS[
            "apply_j5_command_offset_to_hardware_action"
        ](action, 2.0, limits)
        firmware_j5_deg = (
            -85.0
            + (converted_action["joint_4.pos"] + 100.0) / 200.0
            * (87.7 + 85.0)
        )
        self.assertAlmostEqual(firmware_j5_deg, 43.8, places=9)
        self.assertAlmostEqual(report["model_j5_target_deg"], 41.8, places=9)
        self.assertAlmostEqual(report["firmware_j5_target_deg"], 43.8, places=9)
        self.assertEqual(action["joint_4.pos"], model_driver)

    def test_j5_home_uses_forward_mapping(self):
        pose = NS["InitialPose"]((0.0, 0.0, 0.0, 0.0, 41.8, 0.0), 99.7)
        limits = [(-180.0, 180.0)] * 4 + [(-85.0, 87.7), (-180.0, 180.0)]
        converted = NS["apply_j5_command_offset_to_initial_pose"](
            pose, 2.0, limits
        )
        self.assertAlmostEqual(converted.joints_deg[4], 43.8, places=9)
        self.assertAlmostEqual(pose.joints_deg[4], 41.8, places=9)


if __name__ == "__main__":
    unittest.main()
