from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "episode_replay", BUNDLE_ROOT / "robot/replay_assembly_new1_episode.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)


class EpisodeReplayTest(unittest.TestCase):
    def test_j2_command_offset_adds_six_and_inverse_maps_tracking_feedback(self):
        replay_action = np.asarray([1.0, 42.0, -30.0, 4.0, 5.0, 6.0])
        commanded = REPLAY.replay_actions_to_firmware_commands(replay_action, 6.0)
        firmware_feedback = commanded.copy()
        feedback = REPLAY.firmware_feedback_to_replay_joints(
            firmware_feedback,
            0.0,
            6.0,
        )
        self.assertTrue(
            np.allclose(commanded, [1.0, 48.0, -30.0, 4.0, 5.0, 6.0])
        )
        self.assertTrue(np.array_equal(feedback, replay_action))
        self.assertTrue(np.array_equal(firmware_feedback, commanded))
        self.assertEqual(replay_action[1], 42.0)

    def test_j2_feedback_offset_subtracts_six_degrees(self):
        firmware = np.asarray([1.0, 48.0, -30.0, 4.0, 5.0, 6.0])
        replay_frame = REPLAY.firmware_feedback_to_replay_joints(firmware, -6.0)
        self.assertTrue(
            np.allclose(replay_frame, [1.0, 42.0, -30.0, 4.0, 5.0, 6.0])
        )

    def test_j2_feedback_offset_does_not_touch_other_axes(self):
        joints = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        corrected = REPLAY.firmware_feedback_to_replay_joints(joints, -6.0)
        self.assertTrue(np.array_equal(corrected[[0, 2, 3, 4, 5]], joints[[0, 2, 3, 4, 5]]))
        self.assertEqual(corrected[1], -4.0)

    def test_j5_command_offset_adds_two_and_inverse_maps_tracking_feedback(self):
        replay_action = np.asarray([1.0, 42.0, -30.0, 4.0, 41.8, 6.0])
        commanded = REPLAY.replay_actions_to_firmware_commands(
            replay_action, 0.0, 0.0, 2.0
        )
        feedback = REPLAY.firmware_feedback_to_replay_joints(
            commanded, 0.0, 0.0, 0.0, 2.0
        )
        self.assertTrue(
            np.allclose(commanded, [1.0, 42.0, -30.0, 4.0, 43.8, 6.0])
        )
        self.assertTrue(np.allclose(feedback, replay_action))
        self.assertEqual(replay_action[4], 41.8)

    def test_j3_command_offset_adds_one_point_five_and_inverse_maps_feedback(self):
        replay_action = np.asarray([1.0, 42.0, -27.0, 4.0, 5.0, 6.0])
        commanded = REPLAY.replay_actions_to_firmware_commands(
            replay_action, 0.0, 1.5, 0.0
        )
        feedback = REPLAY.firmware_feedback_to_replay_joints(
            commanded, 0.0, 0.0, 1.5, 0.0
        )
        self.assertTrue(
            np.allclose(commanded, [1.0, 42.0, -25.5, 4.0, 5.0, 6.0])
        )
        self.assertTrue(np.allclose(feedback, replay_action))

    def test_closed_hold_separates_contact_aperture_from_clamp_target(self):
        self.assertEqual(
            REPLAY.commanded_gripper_percent(
                48.8, hold_gripper_closed=True
            ),
            0.0,
        )
        self.assertEqual(
            REPLAY.commanded_gripper_percent(
                48.8, hold_gripper_closed=False
            ),
            48.8,
        )

    def test_closed_hold_is_preserved_when_replay_latches(self):
        class FakeHome:
            @staticmethod
            def normalized_feedback(_sdk):
                return np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 49.0])

        class FakeSdk:
            def __init__(self):
                self.command = None

            def set_joint_positions(self, command):
                self.command = command

        sdk = FakeSdk()
        held = REPLAY.latch_replay_pose(
            FakeHome, sdk, hold_gripper_closed=True
        )
        self.assertEqual(held, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0])
        self.assertEqual(sdk.command, held)

    def test_binary_action_emits_only_fully_open_or_closed(self):
        self.assertEqual(
            REPLAY.commanded_gripper_percent(
                49.9,
                hold_gripper_closed=False,
                command_mode="binary-action",
                binary_threshold_percent=50.0,
            ),
            0.0,
        )
        self.assertEqual(
            REPLAY.commanded_gripper_percent(
                50.0,
                hold_gripper_closed=False,
                command_mode="binary-action",
                binary_threshold_percent=50.0,
            ),
            100.0,
        )

    def test_binary_latch_does_not_fall_back_to_contact_aperture(self):
        class FakeHome:
            @staticmethod
            def normalized_feedback(_sdk):
                return np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 49.0])

        class FakeSdk:
            def __init__(self):
                self.command = None

            def set_joint_positions(self, command):
                self.command = command

        sdk = FakeSdk()
        held = REPLAY.latch_replay_pose(
            FakeHome,
            sdk,
            hold_gripper_closed=False,
            commanded_gripper_target_percent=100.0,
        )
        self.assertEqual(held[-1], 100.0)
        self.assertEqual(sdk.command, held)

    def test_resampler_enforces_joint_and_gripper_step_bounds(self):
        joints = np.asarray(
            [[0.0] * 6, [1.0, -0.5, 0.2, 0.0, 0.0, 0.0]], dtype=np.float64
        )
        gripper = np.asarray([100.0, 60.0], dtype=np.float64)
        frames = np.asarray([0, 1], dtype=np.int64)
        q, g, source = REPLAY.resample_bounded(
            joints,
            gripper,
            frames,
            time_scale=2.0,
            max_joint_step_deg=0.25,
            max_gripper_step_percent=5.0,
        )
        self.assertLessEqual(float(np.max(np.abs(np.diff(q, axis=0)))), 0.25 + 1e-12)
        self.assertLessEqual(float(np.max(np.abs(np.diff(g)))), 5.0 + 1e-12)
        self.assertEqual(len(q), 9)
        self.assertAlmostEqual(float(source[-1]), 1.0)

    def test_resampler_preserves_descending_source_timeline(self):
        joints = np.asarray(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0] * 6],
            dtype=np.float64,
        )
        gripper = np.asarray([0.0, 0.0], dtype=np.float64)
        frames = np.asarray([350, 250], dtype=np.int64)
        q, _, source = REPLAY.resample_bounded(
            joints,
            gripper,
            frames,
            time_scale=1.0,
            max_joint_step_deg=0.25,
            max_gripper_step_percent=5.0,
        )
        self.assertLessEqual(float(np.max(np.abs(np.diff(q, axis=0)))), 0.25 + 1e-12)
        self.assertTrue(np.all(np.diff(source) < 0.0))
        self.assertAlmostEqual(float(source[0]), 350.0)
        self.assertAlmostEqual(float(source[-1]), 250.0)

    def test_loader_rejects_calibration_hash_drift(self):
        calibration = BUNDLE_ROOT / "assets/piper_original_leader_calibration.json"
        metadata = {
            "schema": "assembly_new1_physical_joint_episode_replay_v2",
            "leader_calibration_sha256": "0" * 64,
            "episode_index": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            trajectory = Path(directory) / "replay.npz"
            np.savez_compressed(
                trajectory,
                metadata_json=np.asarray(json.dumps(metadata)),
                frame_index=np.asarray([0]),
                action_percent=np.asarray([[0.0] * 6 + [100.0]]),
                model_follower_joint_deg=np.asarray([[0.0] * 6]),
                recorded_follower_joint_deg=np.asarray([[0.0] * 6]),
                hardware_joint_deg=np.asarray([[0.0] * 6]),
                recorded_follower_gripper_mm=np.asarray([100.0]),
                gripper_percent=np.asarray([100.0]),
            )
            with self.assertRaisesRegex(RuntimeError, "calibration differs"):
                REPLAY.load_trajectory(trajectory, calibration)


if __name__ == "__main__":
    unittest.main()
