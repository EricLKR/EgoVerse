#!/usr/bin/env python
"""CPU-only structural tests for the additive V3/V4 RTC deployment."""

from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig

from deploy_egoverse_pi05_piper_v3_v4_rtc_joint import (
    MODEL_ACTION_DIM,
    MODEL_HORIZON,
    MaskedRTCProcessor,
    RTCCommandQueue,
    RTCGripperStateMachine,
    RTCInferenceWorker,
    _decode_physical_joints,
)


def test_masked_rtc_guidance() -> None:
    config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=10.0,
        execution_horizon=10,
    )
    processor = MaskedRTCProcessor(config, "pose-joint")
    x = torch.linspace(-0.5, 0.5, MODEL_HORIZON * MODEL_ACTION_DIM).reshape(
        1, MODEL_HORIZON, MODEL_ACTION_DIM
    )
    previous = torch.zeros_like(x)

    def denoiser(value: torch.Tensor) -> torch.Tensor:
        return 0.25 * value.square() + 0.1

    base = denoiser(x)
    guided = processor.denoise_step(
        x,
        previous,
        inference_delay=3,
        time_value=torch.tensor(0.8),
        original_denoise_step_partial=denoiser,
    )
    assert guided.shape == x.shape
    assert torch.isfinite(guided).all()
    assert not torch.allclose(guided[..., :15], base[..., :15])
    assert torch.allclose(guided[..., 15:], base[..., 15:], atol=1e-7, rtol=0.0)


def test_decoded_joint_rtc_guidance() -> None:
    config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        max_guidance_weight=10.0,
        execution_horizon=10,
    )
    processor = MaskedRTCProcessor(config, "pose-joint", "decoded-joint")
    x = torch.zeros(1, MODEL_HORIZON, MODEL_ACTION_DIM)
    x[..., 9:15] = 0.25
    previous = torch.zeros_like(x)
    current = torch.zeros(1, 7)
    decoder = SimpleNamespace(
        contract_version=4,
        joint_residual_scale_rad_vector=torch.ones(6),
        joint_error_scale_rad=torch.tensor(0.1),
    )

    def denoiser(value: torch.Tensor) -> torch.Tensor:
        return 0.2 * value

    base = denoiser(x)
    guided = processor.denoise_step(
        x,
        previous,
        inference_delay=3,
        time_value=torch.tensor(0.8),
        original_denoise_step_partial=denoiser,
        prev_chunk_decoded_joints=torch.zeros(MODEL_HORIZON, 6),
        current_raw=current,
        joint_decoder=decoder,
    )
    assert guided.shape == x.shape
    assert torch.isfinite(guided).all()
    assert not torch.allclose(guided[..., 9:15], base[..., 9:15])
    assert torch.allclose(guided[..., :9], base[..., :9], atol=1e-7, rtol=0.0)
    assert torch.allclose(guided[..., 15:], base[..., 15:], atol=1e-7, rtol=0.0)

    decoded = _decode_physical_joints(x, current, decoder)
    assert torch.allclose(decoded, torch.tanh(x[..., 9:15]))


def test_queue_uses_successful_action_ticks() -> None:
    original = torch.arange(
        MODEL_HORIZON * MODEL_ACTION_DIM, dtype=torch.float32
    ).reshape(MODEL_HORIZON, MODEL_ACTION_DIM)
    processed = torch.arange(MODEL_HORIZON * 7, dtype=torch.float32).reshape(
        MODEL_HORIZON, 7
    )
    physical = processed / 100.0
    command_queue = RTCCommandQueue()
    assert command_queue.load_initial(original, physical, processed) == 1
    for _ in range(3):
        command_queue.advance()
    snapshot = command_queue.snapshot_for_replan()
    for _ in range(4):
        command_queue.advance()

    new_original = original + 1000.0
    new_physical = physical + 1.0
    new_processed = processed + 1000.0
    report = command_queue.merge(
        snapshot, new_original, new_physical, new_processed
    )
    assert report["actual_delay_ticks"] == 4
    assert report["remaining_new_actions"] == MODEL_HORIZON - 4
    assert report["switch_physical_joint_rms_rad"] is not None
    assert report["switch_physical_joint_rms_deg"] is not None
    queued, status = command_queue.peek()
    assert status["tick"] == 7
    assert torch.equal(queued, new_processed[4])


def test_gripper_absolute_timeline() -> None:
    controller = RTCGripperStateMachine(
        mode="direct",
        close_confirm_frames=1,
        open_confirm_frames=1,
        min_close_hold_frames=0,
        close_feedback_threshold=10.0,
        open_feedback_threshold=90.0,
        terminal_open_after_release=False,
    )
    commands = np.full(MODEL_HORIZON, 100.0, dtype=np.float64)
    commands[5:8] = 0.0
    report = controller.observe_prediction_horizon_at(1, commands, 0)
    assert report["first_close_offset"] == 5
    for _ in range(5):
        state = controller.update_from_prediction_horizons(100.0, 100.0)
        assert state["model_request"] == "open"
        controller.advance_prediction_tick()
    state = controller.update_from_prediction_horizons(100.0, 100.0)
    assert state["model_request"] == "close"


def test_measured_action_delay_replaces_wall_clock_estimate() -> None:
    args = SimpleNamespace(
        rtc_latency_window=4,
        rtc_control_hz=30.0,
        rtc_initial_delay_ticks=6,
        rtc_max_delay_ticks=25,
    )
    worker = RTCInferenceWorker(
        args=args,
        policy=None,
        command_queue=RTCCommandQueue(),
        device=torch.device("cpu"),
        initial_inference_latency_s=0.5,
    )
    assert worker.estimated_delay_ticks() == 15
    worker._observed_delay_ticks.extend((8, 9))
    assert worker.estimated_delay_ticks() == 9


def main() -> None:
    test_masked_rtc_guidance()
    test_decoded_joint_rtc_guidance()
    test_queue_uses_successful_action_ticks()
    test_gripper_absolute_timeline()
    test_measured_action_delay_replaces_wall_clock_estimate()
    print("PI05_PIPER_V3_V4_RTC_UNIT_TESTS_OK")


if __name__ == "__main__":
    main()
