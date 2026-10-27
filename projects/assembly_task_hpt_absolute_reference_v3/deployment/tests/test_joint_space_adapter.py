from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_ROOT / "robot"))

from joint_space_adapter import JointSpaceAdapter


class JointSpaceAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = JointSpaceAdapter.from_files(
            mapping_path=BUNDLE_ROOT / "assets/reset_control.json",
            model_calibration_path=(
                BUNDLE_ROOT / "assets/piper_follower_calibration.json"
            ),
            hardware_calibration_path=(
                BUNDLE_ROOT / "assets/piper_original_leader_calibration.json"
            ),
        )

    def test_follower_physical_home_is_preserved(self):
        mapped = self.adapter.model_degrees_to_hardware_degrees(
            self.adapter.model_home_deg
        )
        self.assertTrue(np.allclose(mapped, self.adapter.model_home_deg, atol=1e-9))

    def test_joint_mapping_round_trip(self):
        follower = np.asarray([-20.0, 80.0, -90.0, 15.0, 20.0, -40.0])
        leader = self.adapter.model_degrees_to_hardware_degrees(follower)
        recovered = self.adapter.hardware_degrees_to_model_degrees(leader)
        self.assertTrue(np.allclose(recovered, follower, atol=1e-9))

    def test_hardware_observation_is_converted_to_training_follower_space(self):
        observation = {
            **{
                f"joint_{index}.pos": value * 1000.0
                for index, value in enumerate(self.adapter.model_home_deg)
            },
            "joint_6.pos": self.adapter.hardware_home_gripper_mm * 1000.0,
            "435if": np.zeros((2, 2, 3), dtype=np.uint8),
        }
        converted = self.adapter.hardware_observation_to_model_observation(
            observation
        )
        converted_joints = np.asarray(
            [converted[f"joint_{index}.pos"] / 1000.0 for index in range(6)]
        )
        self.assertTrue(
            np.allclose(converted_joints, self.adapter.model_home_deg, atol=1e-9)
        )
        self.assertAlmostEqual(
            converted["joint_6.pos"] / 1000.0,
            self.adapter.model_home_gripper_mm,
            places=9,
        )
        self.assertIs(converted["435if"], observation["435if"])

    def test_driver_conversion_preserves_physical_joint_angles(self):
        action = {f"joint_{index}.pos": -75.0 + index * 25.0 for index in range(6)}
        action["joint_6.pos"] = 100.0
        converted = self.adapter.model_driver_action_to_hardware_action(action)
        model_driver = np.asarray([action[f"joint_{index}.pos"] for index in range(6)])
        actual = np.asarray([converted[f"joint_{index}.pos"] for index in range(6)])
        model_deg = np.asarray(
            [
                low + (value + 100.0) / 200.0 * (high - low)
                for value, (low, high) in zip(
                    model_driver, self.adapter.model_limits_deg, strict=True
                )
            ]
        )
        hardware_deg = np.asarray(
            [
                low + (value + 100.0) / 200.0 * (high - low)
                for value, (low, high) in zip(
                    actual, self.adapter.hardware_limits_deg, strict=True
                )
            ]
        )
        self.assertTrue(np.allclose(hardware_deg, model_deg, atol=1e-9))
        self.assertFalse(np.allclose(actual, model_driver, atol=1e-3))
        self.assertEqual(converted["joint_6.pos"], 100.0)

    def test_target_outside_original_leader_physical_limits_is_rejected(self):
        target = np.asarray(self.adapter.model_home_deg)
        target[1] = self.adapter.hardware_limits_deg[1][1] + 0.1
        with self.assertRaisesRegex(RuntimeError, "physical limits"):
            self.adapter.model_degrees_to_hardware_degrees(target)

    def test_approved_zero_offset_is_applied_bidirectionally_to_all_states(self):
        offset = np.asarray([0.2, -0.3, 0.4, -1.0, 0.8, 1.5])
        with tempfile.TemporaryDirectory() as directory:
            offset_path = Path(directory) / "offset.json"
            offset_path.write_text(
                json.dumps(
                    {
                        "schema": "assembly_new1_joint_zero_offset_v1",
                        "status": "approved_active",
                        "hardware_minus_model_joint_offset_deg": offset.tolist(),
                        "multi_pose_validation": {
                            "required": True,
                            "passed": True,
                            "pose_count": 3,
                        },
                    }
                ),
                encoding="utf-8",
            )
            adapter = JointSpaceAdapter.from_files(
                mapping_path=BUNDLE_ROOT / "assets/reset_control.json",
                model_calibration_path=(
                    BUNDLE_ROOT / "assets/piper_follower_calibration.json"
                ),
                hardware_calibration_path=(
                    BUNDLE_ROOT / "assets/piper_original_leader_calibration.json"
                ),
                joint_zero_offset_path=offset_path,
            )
        model = np.asarray(adapter.model_home_deg)
        hardware = adapter.model_degrees_to_hardware_degrees(model)
        expected = model + offset
        self.assertTrue(np.allclose(hardware, expected, atol=1e-9))
        self.assertTrue(
            np.allclose(
                adapter.hardware_degrees_to_model_degrees(hardware),
                model,
                atol=1e-9,
            )
        )

    def test_home_only_candidate_cannot_be_activated(self):
        with tempfile.TemporaryDirectory() as directory:
            offset_path = Path(directory) / "candidate.json"
            offset_path.write_text(
                json.dumps(
                    {
                        "schema": "assembly_new1_joint_zero_offset_v1",
                        "status": "candidate_not_activated",
                        "hardware_minus_model_joint_offset_deg": [0.0] * 6,
                        "multi_pose_validation": {
                            "required": True,
                            "passed": False,
                            "pose_count": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not deployable"):
                JointSpaceAdapter.from_files(
                    mapping_path=BUNDLE_ROOT / "assets/reset_control.json",
                    model_calibration_path=(
                        BUNDLE_ROOT / "assets/piper_follower_calibration.json"
                    ),
                    hardware_calibration_path=(
                        BUNDLE_ROOT / "assets/piper_original_leader_calibration.json"
                    ),
                    joint_zero_offset_path=offset_path,
                )

    def test_experimental_offset_requires_explicit_opt_in(self):
        offset = np.asarray([0.0, 0.0, 0.0, 7.67, -0.68, -5.14])
        with tempfile.TemporaryDirectory() as directory:
            offset_path = Path(directory) / "experimental.json"
            offset_path.write_text(
                json.dumps(
                    {
                        "schema": "assembly_new1_joint_zero_offset_v1",
                        "status": "experimental_unvalidated",
                        "hardware_minus_model_joint_offset_deg": offset.tolist(),
                        "multi_pose_validation": {
                            "required": True,
                            "passed": False,
                            "pose_count": 3,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "explicit deployment opt-in"):
                JointSpaceAdapter.from_files(
                    mapping_path=BUNDLE_ROOT / "assets/reset_control.json",
                    model_calibration_path=(
                        BUNDLE_ROOT / "assets/piper_follower_calibration.json"
                    ),
                    hardware_calibration_path=(
                        BUNDLE_ROOT / "assets/piper_original_leader_calibration.json"
                    ),
                    joint_zero_offset_path=offset_path,
                )
            adapter = JointSpaceAdapter.from_files(
                mapping_path=BUNDLE_ROOT / "assets/reset_control.json",
                model_calibration_path=(
                    BUNDLE_ROOT / "assets/piper_follower_calibration.json"
                ),
                hardware_calibration_path=(
                    BUNDLE_ROOT / "assets/piper_original_leader_calibration.json"
                ),
                joint_zero_offset_path=offset_path,
                allow_experimental_joint_mapping=True,
            )
        model = np.asarray(adapter.model_home_deg)
        self.assertEqual(adapter.joint_zero_offset_status, "experimental_unvalidated")
        self.assertTrue(
            np.allclose(
                adapter.model_degrees_to_hardware_degrees(model),
                model + offset,
                atol=1e-9,
            )
        )


if __name__ == "__main__":
    unittest.main()
