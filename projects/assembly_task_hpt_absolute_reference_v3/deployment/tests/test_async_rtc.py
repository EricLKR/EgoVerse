#!/usr/bin/env python
from __future__ import annotations

import ast
import json
import math
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Unit-test the scheduler without importing camera/CAN drivers.  The base
# gripper state machine is lifted verbatim from the audited Assembly New1
# synchronous deployment source, so its behavior is still tested rather than
# replaced by a hand-written mock.
_BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BUNDLE_ROOT / "robot"))
_LEGACY_PATH = _BUNDLE_ROOT / "robot/deploy_assembly_new1_hpt_joint.py"
_tree = ast.parse(_LEGACY_PATH.read_text(encoding="utf-8"))
_gripper_node = next(
    node
    for node in _tree.body
    if isinstance(node, ast.ClassDef)
    and node.name == "IndependentGripperStateMachine"
)
_legacy_stub = types.ModuleType("deploy_assembly_new1_hpt_joint")
_legacy_stub.__dict__.update({"np": np, "Any": Any})
exec(
    compile(
        ast.Module(body=[_gripper_node], type_ignores=[]),
        str(_LEGACY_PATH),
        "exec",
    ),
    _legacy_stub.__dict__,
)


def _stub_live_observation_to_batch(observation, device, adapter=None):
    return {
        "observation.state": torch.tensor(
            [observation[f"joint_{index}.pos"] for index in range(7)],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
    }


_legacy_stub.live_observation_to_batch = _stub_live_observation_to_batch
_legacy_stub.hardware_observation_for_model = lambda observation, j2_offset, j3_offset=0.0, j5_offset=0.0: {
    **observation,
    "joint_1.pos": float(observation["joint_1.pos"]) - float(j2_offset) * 1000.0,
    "joint_2.pos": float(observation["joint_2.pos"]) - float(j3_offset) * 1000.0,
    "joint_4.pos": float(observation["joint_4.pos"]) - float(j5_offset) * 1000.0,
}
sys.modules["deploy_assembly_new1_hpt_joint"] = _legacy_stub

import deploy_assembly_new1_hpt_joint_async_rtc as rtc
import deploy_assembly_new1_hpt_joint_async_rtc_boundary_bridge as bridge_rtc


class FakePolicy:
    def __init__(self) -> None:
        self._last_planned_velocity_driver = np.zeros(6, dtype=np.float64)

    @staticmethod
    def _joint_rad_to_driver(joints: np.ndarray) -> np.ndarray:
        return np.asarray(joints, dtype=np.float32) * 10.0

    @staticmethod
    def _filter_driver_action(raw, current, velocity):
        raw = np.asarray(raw, dtype=np.float32).copy()
        next_velocity = raw[:6].astype(np.float64) - np.asarray(current)
        return raw, next_velocity


class FakeBoundaryPolicy(FakePolicy):
    def __init__(self) -> None:
        super().__init__()
        # This range makes the inverse conversion exactly driver = rad * 10.
        self._joint_low_rad = np.full(6, -10.0, dtype=np.float64)
        self._joint_high_rad = np.full(6, 10.0, dtype=np.float64)
        self.config = types.SimpleNamespace(
            max_joint_command_step=1.0,
            max_joint_command_accel=0.25,
        )


def report(value: float, steps: int = 30) -> dict:
    physical = np.zeros((steps, 7), dtype=np.float32)
    physical[:, :6] = value
    physical[:, 6] = 1.0
    return {"selected_physical_actions": physical.tolist()}


class DecodedJointRtcTest(unittest.TestCase):
    @staticmethod
    def write_first_grasp_profile(directory: str) -> Path:
        path = Path(directory) / "first_grasp_pose_profile.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "assembly_first_grasp_pose_profile_v1",
                    "samples_deg": [
                        [0.0, 110.0, -35.0, -20.0, -30.0, 5.0],
                        [1.0, 109.0, -34.0, -19.0, -29.0, 6.0],
                    ],
                    "percentiles_deg": {
                        "p01": [-1.0, 107.0, -38.0, -23.0, -33.0, 2.0],
                        "p99": [2.0, 112.0, -32.0, -17.0, -27.0, 8.0],
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_three_camera_snapshot_is_worker_owned(self):
        observation = {
            **{f"joint_{index}.pos": float(index) for index in range(7)},
            "435if": np.full((480, 640, 3), 1, dtype=np.uint8),
            "405": np.full((480, 640, 3), 2, dtype=np.uint8),
            "side": np.full((480, 640, 3), 3, dtype=np.uint8),
        }
        snapshot = rtc.copy_live_observation(observation)
        observation["side"].fill(99)
        self.assertEqual(int(snapshot["side"][0, 0, 0]), 3)
        self.assertIn("_captured_ns", snapshot)

    def test_background_predictor_is_single_flight_and_returns_plan(self):
        class WorkerPolicy:
            def __init__(self):
                self.last_prediction_report = None

            def predict_action_chunk(self, batch):
                self.last_prediction_report = report(0.25)
                return torch.full((30, 7), 2.5, dtype=torch.float32)

        observation = {
            **{f"joint_{index}.pos": float(index) for index in range(7)},
            "435if": np.zeros((480, 640, 3), dtype=np.uint8),
            "405": np.zeros((480, 640, 3), dtype=np.uint8),
            "side": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        snapshot = rtc.copy_live_observation(observation)
        worker = rtc.BackgroundPredictor(WorkerPolicy(), torch.device("cpu"))
        try:
            request = rtc.InferenceRequest(
                request_id=7,
                request_tick=4,
                request_frame=12,
                observation=snapshot,
                submitted_ns=time.monotonic_ns(),
                snapshot_ms=0.2,
            )
            self.assertTrue(worker.submit(request))
            self.assertFalse(worker.submit(request))
            result = None
            deadline = time.monotonic() + 2.0
            while result is None and time.monotonic() < deadline:
                result = worker.poll()
                time.sleep(0.005)
            self.assertIsNotNone(result)
            self.assertIsNone(result.error)
            self.assertEqual(result.request_tick, 4)
            self.assertEqual(result.actions.shape, (30, 7))
            self.assertEqual(
                np.asarray(result.prediction_report["selected_physical_actions"]).shape,
                (30, 7),
            )
        finally:
            worker.close()

    def test_clip_report_ignores_float_noise_but_detects_real_clamp(self):
        proposed = {f"joint_{index}.pos": 10.0 for index in range(7)}
        float_noise = dict(proposed)
        float_noise["joint_4.pos"] += 1.8e-6
        report = rtc.command_clip_report(
            proposed,
            float_noise,
            arm_tolerance=1e-4,
        )
        self.assertFalse(report["arm_clipped"])
        self.assertAlmostEqual(report["arm_max_abs_delta"], 1.8e-6)

        real_clamp = dict(proposed)
        real_clamp["joint_4.pos"] += 0.01
        report = rtc.command_clip_report(
            proposed,
            real_clamp,
            arm_tolerance=1e-4,
        )
        self.assertTrue(report["arm_clipped"])
        self.assertAlmostEqual(report["arm_max_abs_delta"], 0.01)

    def test_delayed_result_skips_expired_prefix_and_blends_arm_only(self):
        controller = rtc.DecodedJointRtcController(FakePolicy(), 8, 8)
        first = controller.install(
            plan_id=0,
            start_tick=0,
            actions=np.zeros((30, 7), dtype=np.float32),
            prediction_report=report(0.0),
            result_age_ms=0.0,
            max_result_age_ms=500.0,
        )
        self.assertTrue(first["accepted"])
        for _ in range(13):
            controller.peek(np.zeros(7))
            controller.advance()
        second = controller.install(
            plan_id=1,
            start_tick=10,
            actions=np.ones((30, 7), dtype=np.float32),
            prediction_report=report(1.0),
            result_age_ms=100.0,
            max_result_age_ms=500.0,
        )
        self.assertTrue(second["accepted"])
        self.assertEqual(second["skipped_prefix_actions"], 3)
        self.assertEqual(second["blended_actions"], 8)
        self.assertAlmostEqual(second["switch_joint_rms_before_rad"], 1.0)
        self.assertLess(second["switch_joint_rms_after_rad"], 1.0)
        active = controller.active_plan.physical_actions
        self.assertEqual(float(active[3, 6]), 1.0)

    def test_stale_and_short_results_are_rejected(self):
        controller = rtc.DecodedJointRtcController(FakePolicy(), 8, 8)
        controller.install(
            plan_id=0,
            start_tick=0,
            actions=np.zeros((30, 7), dtype=np.float32),
            prediction_report=report(0.0),
            result_age_ms=0.0,
            max_result_age_ms=500.0,
        )
        stale = controller.install(
            plan_id=1,
            start_tick=0,
            actions=np.zeros((30, 7), dtype=np.float32),
            prediction_report=report(0.0),
            result_age_ms=501.0,
            max_result_age_ms=500.0,
        )
        self.assertFalse(stale["accepted"])
        self.assertEqual(stale["reason"], "result_exceeded_wall_time_age_limit")
        controller.action_tick = 23
        short = controller.install(
            plan_id=2,
            start_tick=0,
            actions=np.zeros((30, 7), dtype=np.float32),
            prediction_report=report(0.0),
            result_age_ms=100.0,
            max_result_age_ms=500.0,
        )
        self.assertFalse(short["accepted"])
        self.assertEqual(
            short["reason"], "result_has_too_few_time_aligned_suffix_actions"
        )

    def test_exhausted_plan_holds_last_action_without_advancing_time(self):
        controller = rtc.DecodedJointRtcController(FakePolicy(), 8, 8)
        physical = np.zeros((30, 7), dtype=np.float32)
        physical[-1, :6] = 0.25
        physical[-1, 6] = 1.0
        installed = controller.install(
            plan_id=0,
            start_tick=0,
            actions=np.zeros((30, 7), dtype=np.float32),
            prediction_report={"selected_physical_actions": physical.tolist()},
            result_age_ms=0.0,
            max_result_age_ms=500.0,
        )
        self.assertTrue(installed["accepted"])
        controller.action_tick = 30

        action, status = controller.peek(np.zeros(7))

        self.assertTrue(status["plan_exhausted_hold"])
        self.assertEqual(status["requested_plan_offset"], 30)
        self.assertEqual(status["active_plan_offset"], 29)
        self.assertTrue(np.allclose(action[:6], 2.5))
        self.assertEqual(float(action[6]), 100.0)
        controller.cancel_pending()
        self.assertEqual(controller.action_tick, 30)

    def test_finish_active_buffers_result_and_executes_final_action(self):
        controller = rtc.DecodedJointRtcController(
            FakePolicy(),
            8,
            8,
            finish_active_plan_before_install=True,
        )
        first_physical = np.zeros((30, 7), dtype=np.float32)
        first_physical[:, :6] = np.arange(30, dtype=np.float32)[:, None] / 100.0
        controller.install(
            plan_id=0,
            start_tick=0,
            actions=np.zeros((30, 7), dtype=np.float32),
            prediction_report={
                "selected_physical_actions": first_physical.tolist()
            },
            result_age_ms=0.0,
            max_result_age_ms=500.0,
        )
        controller.action_tick = 24
        deferred = controller.install(
            plan_id=1,
            start_tick=22,
            actions=np.ones((30, 7), dtype=np.float32),
            prediction_report=report(1.0),
            result_age_ms=80.0,
            max_result_age_ms=500.0,
        )
        self.assertTrue(deferred["accepted"])
        self.assertTrue(deferred["deferred_install"])
        self.assertEqual(deferred["scheduled_activation_tick"], 30)
        self.assertEqual(controller.active_plan.plan_id, 0)
        self.assertEqual(controller.buffered_plan.plan_id, 1)
        self.assertTrue(controller.has_buffered_plan())

        executed_offsets = []
        while controller.action_tick < 30:
            _, status = controller.peek(np.zeros(7))
            executed_offsets.append(status["active_plan_offset"])
            controller.advance()
        self.assertEqual(executed_offsets[-1], 29)

        self.assertEqual(controller.remaining_actions(), 22)
        self.assertFalse(controller.has_buffered_plan())
        self.assertEqual(controller.active_plan.plan_id, 1)
        action, status = controller.peek(np.zeros(7))
        self.assertEqual(status["active_plan_offset"], 8)
        self.assertTrue(status["last_boundary_activation"]["activated"])
        self.assertEqual(
            status["last_boundary_activation"]["rtc_overlap_blend_actions"], 0
        )
        self.assertTrue(np.allclose(action[:6], 10.0))
        controller.cancel_pending()

    def test_finish_active_keeps_latest_usable_buffered_result(self):
        controller = rtc.DecodedJointRtcController(
            FakePolicy(),
            8,
            8,
            finish_active_plan_before_install=True,
        )
        controller.install(
            plan_id=0,
            start_tick=0,
            actions=np.zeros((30, 7), dtype=np.float32),
            prediction_report=report(0.0),
            result_age_ms=0.0,
            max_result_age_ms=500.0,
        )
        controller.action_tick = 20
        first = controller.install(
            plan_id=1,
            start_tick=18,
            actions=np.ones((30, 7), dtype=np.float32),
            prediction_report=report(1.0),
            result_age_ms=50.0,
            max_result_age_ms=500.0,
        )
        second = controller.install(
            plan_id=2,
            start_tick=20,
            actions=np.full((30, 7), 2.0, dtype=np.float32),
            prediction_report=report(2.0),
            result_age_ms=50.0,
            max_result_age_ms=500.0,
        )
        self.assertIsNone(first["replaced_buffered_plan_id"])
        self.assertEqual(second["replaced_buffered_plan_id"], 1)
        self.assertEqual(controller.buffered_plan.plan_id, 2)

    def test_finish_active_boundary_bridge_is_applied_and_accel_continuous(self):
        controller = bridge_rtc.BoundaryBridgedDecodedJointRtcController(
            FakeBoundaryPolicy(),
            2,
            8,
            finish_active_plan_before_install=True,
        )
        controller.install(
            plan_id=0,
            start_tick=0,
            actions=np.zeros((30, 7), dtype=np.float32),
            prediction_report=report(0.0),
            result_age_ms=0.0,
            max_result_age_ms=500.0,
        )
        current = np.zeros(7, dtype=np.float64)
        sent = []
        for _ in range(28):
            action, _ = controller.peek(current)
            sent.append(action[:6].astype(np.float64))
            current[:6] = action[:6]
            controller.advance()
        deferred = controller.install(
            plan_id=1,
            start_tick=28,
            actions=np.ones((30, 7), dtype=np.float32),
            prediction_report=report(1.0),
            result_age_ms=60.0,
            max_result_age_ms=500.0,
        )
        self.assertTrue(deferred["deferred_install"])
        for _ in range(2):
            action, _ = controller.peek(current)
            sent.append(action[:6].astype(np.float64))
            current[:6] = action[:6]
            controller.advance()

        action, status = controller.peek(current)
        sent.append(action[:6].astype(np.float64))
        event = status["last_boundary_activation"]
        self.assertTrue(event["activated"])
        self.assertEqual(event["rtc_overlap_blend_actions"], 8)
        self.assertEqual(
            event["boundary_bridge"],
            "decoded_joint_quintic_constant_velocity",
        )
        self.assertLess(
            event["switch_joint_rms_after_deg"],
            event["switch_joint_rms_before_deg"],
        )
        self.assertEqual(
            status["command_filter_anchor"], "previous_sent_command"
        )
        # The bridge changes arm joints only; gripper remains binary/open.
        self.assertEqual(float(action[6]), 100.0)
        self.assertLess(float(action[0]), 1.0)

        sent_array = np.asarray(sent)
        velocity = np.diff(sent_array, axis=0)
        acceleration = np.diff(velocity, axis=0)
        self.assertLessEqual(
            float(np.max(np.abs(acceleration))), 0.25 + 1e-6
        )
        controller.cancel_pending()

    def test_contact_force_replan_is_consumed_once(self):
        controller = rtc.DecodedJointRtcController(FakePolicy(), 8, 8)
        first = controller.request_replan("gripper_object_contact_latched")
        duplicate = controller.request_replan("duplicate")
        self.assertTrue(first["requested"])
        self.assertFalse(duplicate["requested"])
        self.assertEqual(
            controller.pending_replan_reason(),
            "gripper_object_contact_latched",
        )
        self.assertEqual(
            controller.consume_replan_request(),
            "gripper_object_contact_latched",
        )
        self.assertIsNone(controller.consume_replan_request())

    def test_cosine_transition_is_monotonic(self):
        weights = [rtc.DecodedJointRtcController._take_new_weight(i, 8) for i in range(8)]
        self.assertTrue(all(0.0 < value < 1.0 for value in weights))
        self.assertTrue(all(a < b for a, b in zip(weights, weights[1:])))
        self.assertAlmostEqual(weights[0], 1.0 - weights[-1], places=7)

    def test_first_close_reconciles_inflight_bias_and_active_plan(self):
        class FakeBiasTransform:
            enabled = True
            bias_mm = -8.0

            @staticmethod
            def settings_report():
                return {"enabled": True, "bias_mm": -8.0}

            @staticmethod
            def apply_to_prediction_report(prediction_report):
                output = dict(prediction_report)
                physical = np.asarray(
                    output["selected_physical_actions"], dtype=np.float32
                ).copy()
                physical[:, :6] += 1.0
                output["selected_physical_actions"] = physical.tolist()
                output["cartesian_z_bias"] = {
                    "enabled": True,
                    "applied_z_delta_mm": -8.0,
                }
                return output

        gate = rtc.PhaseGatedCartesianZBiasTransform(
            FakeBiasTransform(), disable_after_first_close=True
        )
        transformed = gate.apply_to_prediction_report(report(0.0))
        self.assertTrue(
            np.allclose(
                np.asarray(transformed["selected_physical_actions"])[:, :6],
                1.0,
            )
        )
        controller = rtc.DecodedJointRtcController(FakePolicy(), 8, 8)
        installed = controller.install(
            plan_id=0,
            start_tick=0,
            actions=np.zeros((30, 7), dtype=np.float32),
            prediction_report=transformed,
            result_age_ms=0.0,
            max_result_age_ms=500.0,
        )
        self.assertTrue(installed["accepted"])
        controller.action_tick = 5
        disabled = gate.disable_for_first_close(action_tick=5)
        self.assertTrue(disabled["changed"])
        transition = controller.transition_active_plan_to_unbiased(
            effective_tick=6, blend_actions=4
        )
        self.assertTrue(transition["changed"])
        active = controller.active_plan.physical_actions
        self.assertTrue(np.allclose(active[5, :6], 1.0))
        self.assertTrue(np.allclose(active[10:, :6], 0.0))

        reconciled, reconcile = gate.reconcile_prediction_report(transformed)
        self.assertTrue(reconcile["reconciled_to_unbiased"])
        self.assertTrue(
            np.allclose(
                np.asarray(reconciled["selected_physical_actions"])[:, :6],
                0.0,
            )
        )

    def test_plan_diagnostic_preserves_raw_aligned_and_blended_actions(self):
        controller = rtc.DecodedJointRtcController(FakePolicy(), 8, 4)
        controller.install(
            plan_id=0,
            start_tick=0,
            actions=np.zeros((30, 7), dtype=np.float32),
            prediction_report=report(0.0),
            result_age_ms=0.0,
            max_result_age_ms=500.0,
        )
        for _ in range(12):
            controller.peek(np.zeros(7))
            controller.advance()
        diagnostic_report = report(1.0)
        diagnostic_report["full_control_gripper_raw"] = [0.75] * 30
        installed = controller.install(
            plan_id=9,
            start_tick=10,
            actions=np.ones((30, 7), dtype=np.float32),
            prediction_report=diagnostic_report,
            result_age_ms=75.0,
            max_result_age_ms=500.0,
            include_diagnostic=True,
        )
        diagnostic = installed.pop("_plan_diagnostic")
        self.assertTrue(installed["accepted"])
        self.assertEqual(installed["skipped_prefix_actions"], 2)
        self.assertEqual(
            np.asarray(diagnostic["decoded_physical_actions_before_rtc"]).shape,
            (30, 7),
        )
        self.assertEqual(
            np.asarray(diagnostic["effective_suffix_from_install_tick"]).shape,
            (28, 7),
        )
        before = np.asarray(diagnostic["new_aligned_overlap_before_blend"])
        after = np.asarray(diagnostic["new_aligned_overlap_after_blend"])
        self.assertTrue(np.allclose(before[:, :6], 1.0))
        self.assertTrue(np.all(after[:, :6] < before[:, :6]))
        self.assertEqual(
            diagnostic["prediction_report"]["full_control_gripper_raw"],
            [0.75] * 30,
        )

    def test_gripper_horizon_uses_capture_tick(self):
        gripper = rtc.AbsoluteTimeGripper(
            mode="direct",
            close_confirm_frames=1,
            open_confirm_frames=1,
            min_close_hold_frames=0,
            min_open_hold_frames=0,
            open_reached_confirm_frames=1,
            reclose_consensus_horizons=2,
            close_feedback_threshold=10,
            open_feedback_threshold=90,
            terminal_open_after_release=False,
        )
        commands = np.full(30, 100.0)
        commands[4:8] = 0.0
        gripper.prediction_tick = 6
        update = gripper.observe_prediction_horizon_at(7, 2, commands)
        self.assertEqual(update["expired_prefix_actions"], 4)
        state = gripper.update_from_prediction_horizons(
            100.0,
            50.0,
            active_plan_id=7,
            active_plan_offset=4,
        )
        self.assertEqual(state["model_command_pct"], 0.0)
        self.assertEqual(state["selected_prediction_offset"], 4)
        self.assertEqual(
            state["prediction_source"], "active_rtc_plan_full_horizon"
        )

    def test_gripper_contact_latches_preloaded_aperture(self):
        gripper = rtc.AbsoluteTimeGripper(
            mode="latched",
            close_confirm_frames=1,
            open_confirm_frames=5,
            min_close_hold_frames=60,
            min_open_hold_frames=0,
            open_reached_confirm_frames=1,
            reclose_consensus_horizons=2,
            contact_detection=True,
            contact_max_feedback=70,
            contact_stall_frames=2,
            contact_motion_tolerance=0.1,
            contact_min_close_frames=2,
            contact_min_drop=10,
            contact_preload_percent=2,
            close_feedback_threshold=10,
            open_feedback_threshold=90,
            terminal_open_after_release=False,
        )

        self.assertEqual(gripper.update(0.0, 100.0)["transition"], "hold->close")
        self.assertFalse(gripper.update(0.0, 60.0)["object_contact_latched"])
        self.assertFalse(gripper.update(0.0, 45.0)["object_contact_latched"])
        self.assertFalse(gripper.update(0.0, 45.05)["object_contact_latched"])
        state = gripper.update(0.0, 45.02)

        self.assertEqual(state["contact_transition"], "object_contact_latched")
        self.assertTrue(state["object_contact_latched"])
        self.assertFalse(state["close_reached"])
        self.assertTrue(state["target_reached"])
        self.assertAlmostEqual(state["contact_latched_target_pct"], 43.02)
        self.assertAlmostEqual(state["command_pct"], 43.02)

        held = gripper.update(0.0, 44.8)
        self.assertTrue(held["object_contact_latched"])
        self.assertAlmostEqual(held["command_pct"], 43.02)

    def test_gripper_contact_does_not_replace_empty_full_close(self):
        gripper = rtc.AbsoluteTimeGripper(
            mode="latched",
            close_confirm_frames=1,
            open_confirm_frames=1,
            min_close_hold_frames=0,
            min_open_hold_frames=0,
            open_reached_confirm_frames=1,
            reclose_consensus_horizons=2,
            contact_detection=True,
            contact_max_feedback=70,
            contact_stall_frames=2,
            contact_motion_tolerance=0.1,
            contact_min_close_frames=1,
            contact_min_drop=10,
            contact_preload_percent=2,
            close_feedback_threshold=10,
            open_feedback_threshold=90,
            terminal_open_after_release=False,
        )

        gripper.update(0.0, 100.0)
        gripper.update(0.0, 60.0)
        gripper.update(0.0, 20.0)
        state = gripper.update(0.0, 8.0)

        self.assertFalse(state["object_contact_latched"])
        self.assertTrue(state["close_reached"])
        self.assertTrue(state["target_reached"])
        self.assertEqual(state["command_pct"], 0.0)

    def test_gripper_contact_latch_clears_on_release(self):
        gripper = rtc.AbsoluteTimeGripper(
            mode="latched",
            close_confirm_frames=1,
            open_confirm_frames=1,
            min_close_hold_frames=0,
            min_open_hold_frames=0,
            open_reached_confirm_frames=1,
            reclose_consensus_horizons=2,
            contact_detection=True,
            contact_max_feedback=70,
            contact_stall_frames=1,
            contact_motion_tolerance=0.1,
            contact_min_close_frames=1,
            contact_min_drop=10,
            contact_preload_percent=2,
            close_feedback_threshold=10,
            open_feedback_threshold=90,
            terminal_open_after_release=False,
        )

        gripper.update(0.0, 100.0)
        gripper.update(0.0, 45.0)
        contact = gripper.update(0.0, 45.0)
        self.assertTrue(contact["object_contact_latched"])

        released = gripper.update(100.0, 45.0)
        self.assertEqual(released["transition"], "close->open")
        self.assertFalse(released["object_contact_latched"])
        self.assertIsNone(released["contact_latched_target_pct"])
        self.assertEqual(released["command_pct"], 100.0)

    def test_first_close_follows_active_rtc_plan_not_stale_horizon(self):
        gripper = rtc.AbsoluteTimeGripper(
            mode="latched",
            close_confirm_frames=1,
            open_confirm_frames=1,
            min_close_hold_frames=0,
            min_open_hold_frames=0,
            open_reached_confirm_frames=1,
            reclose_consensus_horizons=2,
            close_feedback_threshold=10,
            open_feedback_threshold=90,
            terminal_open_after_release=False,
        )
        gripper.prediction_tick = 127
        stale_close = np.full(30, 100.0)
        stale_close[27:] = 0.0
        active_open = np.full(30, 100.0)
        gripper.observe_prediction_horizon_at(10, 100, stale_close)
        gripper.observe_prediction_horizon_at(12, 120, active_open)
        state = gripper.update_from_prediction_horizons(
            100.0,
            100.0,
            active_plan_id=12,
            active_plan_offset=7,
        )
        self.assertEqual(state["state"], "open")
        self.assertEqual(state["model_request"], "open")
        self.assertEqual(state["selected_prediction_chunk_id"], 12)
        self.assertEqual(state["selected_prediction_offset"], 7)
        self.assertTrue(state["active_plan_horizon_match"])
        self.assertEqual(
            state["prediction_source"], "active_rtc_plan_full_horizon"
        )

    def test_any_open_in_remaining_active_horizon_respects_close_hold(self):
        gripper = rtc.AbsoluteTimeGripper(
            mode="latched",
            close_confirm_frames=1,
            open_confirm_frames=1,
            min_close_hold_frames=60,
            min_open_hold_frames=0,
            open_reached_confirm_frames=1,
            reclose_consensus_horizons=2,
            release_on_any_open_in_active_horizon=True,
            close_feedback_threshold=10,
            open_feedback_threshold=90,
            terminal_open_after_release=False,
        )

        initial_close = np.zeros(30)
        gripper.observe_prediction_horizon_at(1, 0, initial_close)
        closed = gripper.update_from_prediction_horizons(
            0.0,
            100.0,
            active_plan_id=1,
            active_plan_offset=0,
        )
        self.assertEqual(closed["transition"], "hold->close")
        gripper.advance_prediction_tick()

        release_forecast = np.zeros(30)
        release_forecast[28:] = 100.0
        gripper.observe_prediction_horizon_at(2, 1, release_forecast)
        held = gripper.update_from_prediction_horizons(
            0.0,
            45.0,
            active_plan_id=2,
            active_plan_offset=0,
        )
        self.assertIsNone(held["transition"])
        self.assertEqual(held["state"], "close")
        self.assertTrue(held["release_on_any_open_triggered"])

        gripper.state_age_frames = gripper.min_close_hold_frames
        released = gripper.update_from_prediction_horizons(
            0.0,
            45.0,
            active_plan_id=2,
            active_plan_offset=0,
        )
        self.assertEqual(released["transition"], "close->open")
        self.assertEqual(released["model_command_pct"], 100.0)
        self.assertTrue(released["release_on_any_open_triggered"])
        self.assertEqual(released["remaining_horizon_open_offsets"], [28, 29])
        self.assertEqual(
            released["prediction_source"],
            "active_rtc_plan_any_open_in_remaining_horizon",
        )

    def test_any_open_release_does_not_block_initial_close(self):
        gripper = rtc.AbsoluteTimeGripper(
            mode="latched",
            close_confirm_frames=1,
            open_confirm_frames=1,
            min_close_hold_frames=0,
            min_open_hold_frames=0,
            open_reached_confirm_frames=1,
            reclose_consensus_horizons=2,
            release_on_any_open_in_active_horizon=True,
            close_feedback_threshold=10,
            open_feedback_threshold=90,
            terminal_open_after_release=False,
        )
        mixed_first_grasp = np.zeros(30)
        mixed_first_grasp[28:] = 100.0
        gripper.observe_prediction_horizon_at(1, 0, mixed_first_grasp)
        state = gripper.update_from_prediction_horizons(
            0.0,
            100.0,
            active_plan_id=1,
            active_plan_offset=0,
        )
        self.assertEqual(state["transition"], "hold->close")
        self.assertFalse(state["release_on_any_open_triggered"])

    def test_release_invalidates_old_forecasts_and_guards_reclose(self):
        gripper = rtc.AbsoluteTimeGripper(
            mode="latched",
            close_confirm_frames=1,
            open_confirm_frames=1,
            min_close_hold_frames=0,
            min_open_hold_frames=2,
            open_reached_confirm_frames=1,
            reclose_consensus_horizons=2,
            close_feedback_threshold=10,
            open_feedback_threshold=90,
            terminal_open_after_release=False,
        )

        first = np.full(30, 100.0)
        first[0] = 0.0
        gripper.observe_prediction_horizon_at(0, 0, first)
        state = gripper.update_from_prediction_horizons(
            100.0,
            100.0,
            active_plan_id=0,
            active_plan_offset=0,
        )
        self.assertEqual(state["transition"], "hold->close")
        gripper.advance_prediction_tick()

        stale_close = np.zeros(30)
        gripper.observe_prediction_horizon_at(1, 1, stale_close)
        state = gripper.update_from_prediction_horizons(100.0, 100.0)
        self.assertEqual(state["transition"], "close->open")
        self.assertEqual(state["release_epoch_tick"], 2)
        self.assertEqual(state["purged_horizons_on_release"], 2)
        self.assertFalse(state["reclose_armed"])
        gripper.advance_prediction_tick()

        late = gripper.observe_prediction_horizon_at(2, 1, stale_close)
        self.assertFalse(late["accepted"])
        self.assertEqual(late["reason"], "captured_before_release_epoch")

        fresh_close = np.zeros(30)
        gripper.observe_prediction_horizon_at(3, 2, fresh_close)
        state = gripper.update_from_prediction_horizons(0.0, 50.0)
        self.assertEqual(state["state"], "open")
        self.assertFalse(state["reclose_armed"])
        self.assertEqual(state["reclose_consensus_vote_count"], 1)
        self.assertEqual(state["open_reached_streak"], 0)
        self.assertEqual(state["open_hold_age_frames"], 0)
        gripper.advance_prediction_tick()

        gripper.observe_prediction_horizon_at(4, 3, fresh_close)
        state = gripper.update_from_prediction_horizons(0.0, 100.0)
        self.assertEqual(state["state"], "open")
        self.assertFalse(state["reclose_armed"])
        self.assertEqual(state["open_hold_age_frames"], 1)
        gripper.advance_prediction_tick()

        state = gripper.update_from_prediction_horizons(0.0, 100.0)
        self.assertEqual(state["state"], "open")
        self.assertTrue(state["reclose_armed"])
        gripper.advance_prediction_tick()

        state = gripper.update_from_prediction_horizons(0.0, 100.0)
        self.assertEqual(state["transition"], "open->close")
        self.assertEqual(state["reclose_consensus_vote_count"], 2)

    def test_latest_post_release_open_vote_vetoes_stale_close(self):
        gripper = rtc.AbsoluteTimeGripper(
            mode="latched",
            close_confirm_frames=1,
            open_confirm_frames=1,
            min_close_hold_frames=0,
            min_open_hold_frames=0,
            open_reached_confirm_frames=1,
            reclose_consensus_horizons=2,
            close_feedback_threshold=10,
            open_feedback_threshold=90,
            terminal_open_after_release=False,
        )
        gripper.release_epoch_tick = 10
        gripper.prediction_tick = 13
        gripper.state = "open"
        gripper.reclose_armed = True
        # Deliberately deliver results out of order.  Freshness must be based
        # on capture tick, not append/worker-completion order.
        gripper.observe_prediction_horizon_at(12, 12, np.full(30, 100.0))
        gripper.observe_prediction_horizon_at(10, 10, np.zeros(30))
        gripper.observe_prediction_horizon_at(11, 11, np.zeros(30))
        state = gripper.update_from_prediction_horizons(0.0, 100.0)
        self.assertEqual(state["state"], "open")
        self.assertFalse(state["reclose_close_consensus"])
        self.assertEqual(state["selected_prediction_chunk_id"], 12)

    def test_initial_close_uses_latest_horizon_consensus(self):
        with tempfile.TemporaryDirectory() as temporary:
            gripper = rtc.AbsoluteTimeGripper(
                mode="latched",
                close_confirm_frames=1,
                open_confirm_frames=1,
                min_close_hold_frames=0,
                min_open_hold_frames=0,
                open_reached_confirm_frames=1,
                reclose_consensus_horizons=2,
                initial_close_guard=True,
                initial_close_consensus_horizons=2,
                first_close_pose_profile=self.write_first_grasp_profile(temporary),
                close_feedback_threshold=10,
                open_feedback_threshold=90,
                terminal_open_after_release=False,
            )
            gripper.prediction_tick = 3
            gripper.observe_prediction_horizon_at(1, 1, np.zeros(30))
            gripper.observe_prediction_horizon_at(2, 2, np.full(30, 100.0))
            gripper.observe_prediction_horizon_at(3, 3, np.full(30, 100.0))
            state = gripper.update_from_prediction_horizons(
                0.0,
                100.0,
                current_model_joint_deg=[0.0, 110.0, -35.0, -20.0, -30.0, 5.0],
            )
            self.assertEqual(state["state"], "open")
            self.assertFalse(state["initial_close_consensus"])
            self.assertEqual(state["selected_prediction_chunk_id"], 3)

    def test_initial_close_pose_gate_rejects_above_object_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            gripper = rtc.AbsoluteTimeGripper(
                mode="latched",
                close_confirm_frames=1,
                open_confirm_frames=1,
                min_close_hold_frames=0,
                min_open_hold_frames=0,
                open_reached_confirm_frames=1,
                reclose_consensus_horizons=2,
                initial_close_guard=True,
                initial_close_consensus_horizons=2,
                first_close_pose_profile=self.write_first_grasp_profile(temporary),
                first_close_max_nearest_rms_deg=3.0,
                first_close_envelope_margin_deg=1.0,
                close_feedback_threshold=10,
                open_feedback_threshold=90,
                terminal_open_after_release=False,
            )
            gripper.prediction_tick = 2
            gripper.observe_prediction_horizon_at(1, 1, np.zeros(30))
            gripper.observe_prediction_horizon_at(2, 2, np.zeros(30))
            state = gripper.update_from_prediction_horizons(
                0.0,
                100.0,
                current_model_joint_deg=[3.1, 100.1, -39.6, -19.7, -15.8, 7.5],
            )
            self.assertEqual(state["state"], "open")
            self.assertTrue(state["initial_close_consensus"])
            self.assertFalse(state["first_close_pose_gate_accepted"])
            self.assertGreater(state["first_close_nearest_sample_rms_deg"], 3.0)

    def test_initial_close_pose_gate_accepts_training_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            gripper = rtc.AbsoluteTimeGripper(
                mode="latched",
                close_confirm_frames=1,
                open_confirm_frames=1,
                min_close_hold_frames=0,
                min_open_hold_frames=0,
                open_reached_confirm_frames=1,
                reclose_consensus_horizons=2,
                initial_close_guard=True,
                initial_close_consensus_horizons=2,
                first_close_pose_profile=self.write_first_grasp_profile(temporary),
                first_close_max_nearest_rms_deg=3.0,
                first_close_envelope_margin_deg=1.0,
                close_feedback_threshold=10,
                open_feedback_threshold=90,
                terminal_open_after_release=False,
            )
            gripper.prediction_tick = 2
            gripper.observe_prediction_horizon_at(1, 1, np.zeros(30))
            gripper.observe_prediction_horizon_at(2, 2, np.zeros(30))
            state = gripper.update_from_prediction_horizons(
                0.0,
                100.0,
                current_model_joint_deg=[0.0, 110.0, -35.0, -20.0, -30.0, 5.0],
            )
            self.assertEqual(state["transition"], "hold->close")
            self.assertTrue(state["first_close_pose_gate_accepted"])

    def test_initial_close_deadline_prevents_replan_procrastination(self):
        with tempfile.TemporaryDirectory() as temporary:
            gripper = rtc.AbsoluteTimeGripper(
                mode="latched",
                close_confirm_frames=5,
                open_confirm_frames=1,
                min_close_hold_frames=60,
                min_open_hold_frames=60,
                open_reached_confirm_frames=3,
                reclose_consensus_horizons=2,
                initial_close_guard=True,
                initial_close_consensus_horizons=2,
                initial_close_deadline_latch=True,
                initial_close_deadline_max_wait_frames=10,
                first_close_pose_profile=self.write_first_grasp_profile(temporary),
                close_feedback_threshold=10,
                open_feedback_threshold=90,
                terminal_open_after_release=False,
            )
            chunk_26 = np.full(30, 100.0)
            chunk_26[28:] = 0.0
            gripper.prediction_tick = 262
            first = gripper.observe_prediction_horizon_at(26, 260, chunk_26)
            self.assertIsNone(first["initial_close_deadline_tick"])

            chunk_27 = np.full(30, 100.0)
            chunk_27[29] = 0.0
            gripper.prediction_tick = 272
            second = gripper.observe_prediction_horizon_at(27, 270, chunk_27)
            self.assertEqual(second["initial_close_deadline_tick"], 288)
            self.assertEqual(
                second["initial_close_deadline_confirming_chunks"], [26, 27]
            )

            training_pose = [0.0, 110.0, -35.0, -20.0, -30.0, 5.0]
            for tick in range(272, 293):
                gripper.prediction_tick = tick
                state = gripper.update_from_prediction_horizons(
                    100.0,
                    100.0,
                    current_model_joint_deg=training_pose,
                    active_plan_id=27,
                    active_plan_offset=min(tick - 270, 29),
                )
                if tick < 288:
                    self.assertEqual(state["model_request"], "open")
            self.assertEqual(state["transition"], "open->close")
            self.assertEqual(state["initial_close_deadline_status"], "consumed")

    def test_initial_close_deadline_expires_if_pose_gate_never_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            gripper = rtc.AbsoluteTimeGripper(
                mode="latched",
                close_confirm_frames=1,
                open_confirm_frames=1,
                min_close_hold_frames=0,
                min_open_hold_frames=0,
                open_reached_confirm_frames=1,
                reclose_consensus_horizons=2,
                initial_close_guard=True,
                initial_close_consensus_horizons=2,
                initial_close_deadline_latch=True,
                initial_close_deadline_max_wait_frames=2,
                first_close_pose_profile=self.write_first_grasp_profile(temporary),
                close_feedback_threshold=10,
                open_feedback_threshold=90,
                terminal_open_after_release=False,
            )
            first = np.full(30, 100.0)
            first[5:] = 0.0
            second = np.full(30, 100.0)
            second[4:] = 0.0
            gripper.observe_prediction_horizon_at(1, 0, first)
            gripper.prediction_tick = 1
            update = gripper.observe_prediction_horizon_at(2, 1, second)
            self.assertEqual(update["initial_close_deadline_tick"], 5)

            rejected_pose = [20.0, 80.0, -70.0, 10.0, 5.0, 40.0]
            for tick in range(1, 9):
                gripper.prediction_tick = tick
                state = gripper.update_from_prediction_horizons(
                    100.0,
                    100.0,
                    current_model_joint_deg=rejected_pose,
                    active_plan_id=2,
                    active_plan_offset=tick - 1,
                )
            self.assertEqual(state["state"], "open")
            self.assertEqual(
                state["initial_close_deadline_status"], "expired_pose_timeout"
            )
            self.assertIsNone(state["initial_close_deadline_tick"])

    def test_open_forecast_resets_unlocked_close_deadline_consensus(self):
        with tempfile.TemporaryDirectory() as temporary:
            gripper = rtc.AbsoluteTimeGripper(
                mode="latched",
                close_confirm_frames=1,
                open_confirm_frames=1,
                min_close_hold_frames=0,
                min_open_hold_frames=0,
                open_reached_confirm_frames=1,
                reclose_consensus_horizons=2,
                initial_close_guard=True,
                initial_close_consensus_horizons=2,
                initial_close_deadline_latch=True,
                initial_close_deadline_max_wait_frames=5,
                first_close_pose_profile=self.write_first_grasp_profile(temporary),
                close_feedback_threshold=10,
                open_feedback_threshold=90,
                terminal_open_after_release=False,
            )
            close_tail = np.full(30, 100.0)
            close_tail[-2:] = 0.0
            gripper.observe_prediction_horizon_at(1, 0, close_tail)
            gripper.prediction_tick = 1
            gripper.observe_prediction_horizon_at(2, 1, np.full(30, 100.0))
            gripper.prediction_tick = 2
            update = gripper.observe_prediction_horizon_at(3, 2, close_tail)
            self.assertEqual(update["initial_close_deadline_candidate_streak"], 1)
            self.assertIsNone(update["initial_close_deadline_tick"])

    def test_async_diagnostic_image_writer_saves_without_control_thread_io(self):
        observation = {
            "435if": np.full((480, 640, 3), 32, dtype=np.uint8),
            "405": np.full((480, 640, 3), 64, dtype=np.uint8),
            "side": np.full((480, 640, 3), 96, dtype=np.uint8),
        }
        with tempfile.TemporaryDirectory() as temporary:
            writer = rtc.AsyncDiagnosticImageWriter(
                output_dir=Path(temporary),
                view_builder=rtc.raw_diagnostic_image_views,
                queue_size=1,
            )
            submit = writer.submit(
                rtc.DiagnosticImageRequest(
                    sample_id=0,
                    frame=30,
                    action_tick=29,
                    request_id=3,
                    observation=observation,
                )
            )
            self.assertTrue(submit["scheduled"])
            results = writer.close()
            self.assertEqual(len(results), 1)
            self.assertIsNone(results[0]["error"])
            self.assertEqual(
                set(results[0]["paths"]),
                {
                    "front_raw",
                    "wrist_raw",
                    "side_raw",
                    "front_model",
                    "wrist_model",
                    "side_model",
                },
            )
            for path in results[0]["paths"].values():
                self.assertTrue(Path(path).is_file())
            self.assertEqual(writer.status()["written"], 1)
            self.assertEqual(writer.status()["errors"], 0)

    def test_async_plan_diagnostic_writer_saves_json_and_model_inputs(self):
        observation = {
            "435if": np.full((480, 640, 3), 16, dtype=np.uint8),
            "405": np.full((480, 640, 3), 96, dtype=np.uint8),
            "side": np.full((480, 640, 3), 144, dtype=np.uint8),
        }
        payload = {
            "diagnostic_only_no_control_effect": True,
            "decoded_physical_actions_before_rtc": np.zeros((30, 7)),
            "prediction_report": {"full_control_gripper_raw": [0.5] * 30},
        }
        with tempfile.TemporaryDirectory() as temporary:
            writer = rtc.AsyncPlanDiagnosticWriter(
                output_dir=Path(temporary),
                view_builder=rtc.raw_diagnostic_image_views,
                queue_size=1,
            )
            submit = writer.submit(
                rtc.PlanDiagnosticRequest(
                    plan_id=9,
                    frame=92,
                    request_tick=90,
                    install_tick=92,
                    observation=observation,
                    payload=payload,
                )
            )
            self.assertTrue(submit["scheduled"])
            results = writer.close()
            self.assertEqual(len(results), 1)
            self.assertIsNone(results[0]["error"])
            artifact = json.loads(Path(results[0]["json_path"]).read_text())
            self.assertEqual(artifact["plan_id"], 9)
            self.assertEqual(artifact["request_tick"], 90)
            self.assertEqual(artifact["install_tick"], 92)
            self.assertEqual(
                np.asarray(artifact["decoded_physical_actions_before_rtc"]).shape,
                (30, 7),
            )
            self.assertEqual(
                set(artifact["image_paths"]),
                {
                    "front_raw",
                    "wrist_raw",
                    "side_raw",
                    "front_model",
                    "wrist_model",
                    "side_model",
                },
            )
            self.assertEqual(writer.status()["written"], 1)
            self.assertEqual(writer.status()["errors"], 0)

    def test_second_pick_phase_diagnostic_is_log_only(self):
        profile = {
            "schema": "assembly_second_pick_pose_profile_v1",
            "second_close_joint_deg": {
                "p05": [-2, -2, -2, -2, -2, -2],
                "median": [0, 0, 0, 0, 0, 0],
                "p95": [2, 2, 2, 2, 2, 2],
            },
            "release_to_second_close_s": {
                "p05": 6.0,
                "median": 8.0,
                "p95": 10.0,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            profile_path = Path(temporary) / "profile.json"
            profile_path.write_text(json.dumps(profile))
            diagnostic = rtc.SecondPickPhaseDiagnostic(profile_path)
            release = diagnostic.update(
                frame=100,
                action_tick=90,
                elapsed_s=5.0,
                joints_deg=[1, 1, 1, 1, 1, 1],
                gripper_transition="close->open",
            )
            self.assertEqual(release["event"], "first_release_detected")
            self.assertEqual(release["phase"], "second_pick_active")
            self.assertTrue(release["all_joints_within_second_close_training_5_95"])
            self.assertTrue(release["diagnostic_only_no_control_effect"])

            late = diagnostic.update(
                frame=350,
                action_tick=340,
                elapsed_s=13.5,
                joints_deg=[0, 0, 0, 0, 0, 3],
                gripper_transition=None,
            )
            self.assertTrue(late["past_training_median_time"])
            self.assertFalse(late["all_joints_within_second_close_training_5_95"])
            self.assertEqual(late["max_abs_error_to_second_close_median_deg"], 3.0)

            closed = diagnostic.update(
                frame=360,
                action_tick=350,
                elapsed_s=14.0,
                joints_deg=[0, 0, 0, 0, 0, 0],
                gripper_transition="open->close",
            )
            self.assertEqual(closed["event"], "second_close_detected")
            self.assertEqual(closed["phase"], "second_pick_complete")


if __name__ == "__main__":
    unittest.main()
