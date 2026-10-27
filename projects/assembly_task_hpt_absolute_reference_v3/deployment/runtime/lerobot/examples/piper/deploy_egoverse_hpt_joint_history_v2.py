#!/usr/bin/env python
"""Four-frame causal-history deployment for EgoVerse joint History V2.

This is a thin, isolated extension of ``deploy_egoverse_hpt_joint.py``.  The
audited robot, command filtering, gripper state machine, queue and logging code
are reused unchanged.  A proxy records camera/proprio observations and creates
the checkpoint's ``[t-300,t-200,t-100,t] ms`` input before every inference.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import torch

import deploy_egoverse_hpt_joint as legacy
from lerobot.policies.egoverse_hpt.configuration_egoverse_hpt_joint_history_v2 import (
    EgoVerseHPTJointHistoryV2Config,
)
from lerobot.policies.egoverse_hpt.modeling_egoverse_hpt_joint_history_v2 import (
    EgoVerseHPTJointHistoryV2Policy,
)


class CausalObservationHistoryV2:
    def __init__(self, history: int, step_ms: float, max_skew_ms: float):
        self.history = int(history)
        self.step_ns = int(float(step_ms) * 1e6)
        self.max_skew_ns = int(float(max_skew_ms) * 1e6)
        self.samples: deque[tuple[int, dict]] = deque(maxlen=max(32, history * 8))
        self.last_selection_report: dict | None = None
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self.samples.clear()
            self.last_selection_report = None

    def append(self, observation: dict, timestamp_ns: int | None = None) -> None:
        timestamp_ns = time.monotonic_ns() if timestamp_ns is None else int(timestamp_ns)
        copied = dict(observation)
        copied["435if"] = np.asarray(observation["435if"]).copy()
        copied["405"] = np.asarray(observation["405"]).copy()
        copied["_history_v2_timestamp_ns"] = timestamp_ns
        with self._lock:
            if self.samples and timestamp_ns <= self.samples[-1][0]:
                raise RuntimeError("History V2 timestamps must be strictly increasing")
            self.samples.append((timestamp_ns, copied))

    def select(self, anchor_ns: int | None = None) -> tuple[list[dict], dict]:
        with self._lock:
            available = list(self.samples)
        if anchor_ns is not None:
            available = [sample for sample in available if sample[0] <= int(anchor_ns)]
        if not available:
            raise RuntimeError("History V2 buffer has no sample at the requested anchor")
        newest_ns = available[-1][0]
        selected: list[tuple[int, dict]] = []
        targets = [
            newest_ns - self.step_ns * offset
            for offset in range(self.history - 1, -1, -1)
        ]
        for target_ns in targets:
            candidates = [sample for sample in available if sample[0] <= target_ns]
            if not candidates:
                raise RuntimeError(
                    "History V2 is not primed far enough into the past; no sample "
                    f"at or before target age {(newest_ns - target_ns) / 1e6:.1f} ms"
                )
            selected.append(candidates[-1])
        timestamps = [sample[0] for sample in selected]
        if any(a > b for a, b in zip(timestamps, timestamps[1:], strict=False)):
            raise RuntimeError("History V2 selected non-monotonic observations")
        skew_ns = [target - actual for target, actual in zip(targets, timestamps, strict=True)]
        if any(skew < 0 or skew > self.max_skew_ns for skew in skew_ns):
            raise RuntimeError(
                "History V2 timestamp skew exceeds limit: "
                f"skew_ms={[value / 1e6 for value in skew_ns]}, "
                f"limit_ms={self.max_skew_ns / 1e6}"
            )
        report = {
            "selected_ages_ms": [(newest_ns - value) / 1e6 for value in timestamps],
            "target_ages_ms": [(newest_ns - value) / 1e6 for value in targets],
            "selection_skew_ms": [value / 1e6 for value in skew_ns],
            "buffer_samples": len(available),
            "anchor_timestamp_ns": newest_ns,
            "causal": True,
        }
        self.last_selection_report = report
        return [sample[1] for sample in selected], report


_ACTIVE_HISTORY: CausalObservationHistoryV2 | None = None
_ORIGINAL_MAKE_ROBOT = legacy.make_robot
_ORIGINAL_LIVE_OBSERVATION_TO_BATCH = legacy.live_observation_to_batch
HISTORY_V2_CHECKPOINT_PROFILES = {
    "piper-history-v2": {
        "checkpoint": "/mnt/wd10/kerun/checkpoints/piper_history_v2_best.ckpt",
        "sha256": "8cfe8e29f34389cd929be0e36ebf3e43101a29035308fa411550824b2c9abffd",
        "description": "four-frame PIPER-only History V2, validation-selected checkpoint",
    },
    "mixed-history-v2": {
        "checkpoint": "/mnt/wd10/kerun/checkpoints/mixed_history_v2_best.ckpt",
        "sha256": "692f045d32229d2223960d52a15e3ff2bd1900555233554e231b33d3a20898b6",
        "description": "four-frame PIPER+human Mixed History V2, validation-selected checkpoint",
    },
}


class HistoryPrimingRobot:
    """Delegating robot proxy with an inference-independent 30 Hz sampler."""

    def __init__(self, robot, history_buffer: CausalObservationHistoryV2):
        self._robot = robot
        self._history = history_buffer
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._sequence = 0
        self._last_returned_sequence = -1

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def connect(self, *args, **kwargs):
        self._history.clear()
        self._stop.clear()
        self._thread = None
        self._worker_error = None
        self._sequence = 0
        self._last_returned_sequence = -1
        return self._robot.connect(*args, **kwargs)

    def _start_sampler(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="history-v2-camera-sampler",
            daemon=True,
        )
        self._thread.start()

    def _sample_loop(self) -> None:
        period_s = min(1.0 / 30.0, self._history.step_ns / 1e9)
        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                observation = self._robot.get_observation()
                timestamp_ns = time.monotonic_ns()
                self._history.append(observation, timestamp_ns)
                with self._condition:
                    self._sequence += 1
                    self._condition.notify_all()
                remaining = period_s - (time.perf_counter() - started)
                if remaining > 0:
                    self._stop.wait(remaining)
        except BaseException as exc:
            with self._condition:
                self._worker_error = exc
                self._condition.notify_all()

    def get_observation(self):
        self._start_sampler()
        deadline = time.monotonic() + 10.0
        while True:
            with self._condition:
                if self._worker_error is not None:
                    raise RuntimeError("History V2 sampler failed") from self._worker_error
                new_sample = self._sequence > self._last_returned_sequence
            if new_sample:
                try:
                    samples, _ = self._history.select()
                except RuntimeError:
                    samples = []
                if len(samples) == self._history.history:
                    latest = dict(samples[-1])
                    with self._condition:
                        self._last_returned_sequence = self._sequence
                    return latest
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("History V2 sampler did not prime within 10 seconds")
            with self._condition:
                self._condition.wait(timeout=min(remaining, 0.1))

    def disconnect(self, *args, **kwargs):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
            if thread.is_alive():
                raise RuntimeError("History V2 sampler did not stop before disconnect")
        return self._robot.disconnect(*args, **kwargs)


def make_history_robot(args, joint_limits, gripper_stroke_mm):
    if _ACTIVE_HISTORY is None:
        raise RuntimeError("History V2 buffer was not initialized")
    return HistoryPrimingRobot(
        _ORIGINAL_MAKE_ROBOT(args, joint_limits, gripper_stroke_mm),
        _ACTIVE_HISTORY,
    )


def make_history_batch_from_samples(samples: list[dict], device: torch.device):
    # ``main`` replaces ``legacy.live_observation_to_batch`` so the unchanged
    # legacy run loop requests a history batch.  Building that batch must still
    # start from the original single-frame converter; calling through the
    # monkey-patched module attribute here would recurse indefinitely.
    current = _ORIGINAL_LIVE_OBSERVATION_TO_BATCH(samples[-1], device)
    raw_states = np.stack(
        [
            np.asarray([sample[f"joint_{index}.pos"] for index in range(7)], dtype=np.float32)
            for sample in samples
        ],
        axis=0,
    )
    current["observation.history_v2.state"] = torch.from_numpy(raw_states).unsqueeze(0).to(device)
    current["observation.history_v2.images.435if"] = torch.from_numpy(
        np.stack([np.asarray(sample["435if"]) for sample in samples], axis=0)
    ).unsqueeze(0).to(device)
    current["observation.history_v2.images.405"] = torch.from_numpy(
        np.stack([np.asarray(sample["405"]) for sample in samples], axis=0)
    ).unsqueeze(0).to(device)
    return current


def live_observation_to_history_batch(observation, device):
    if _ACTIVE_HISTORY is None:
        raise RuntimeError("History V2 buffer was not initialized")
    anchor_ns = observation.get("_history_v2_timestamp_ns")
    if anchor_ns is None:
        raise RuntimeError("History V2 live observation lacks its causal anchor timestamp")
    samples, report = _ACTIVE_HISTORY.select(int(anchor_ns))
    batch = make_history_batch_from_samples(samples, device)
    batch["observation.history_v2.selection_ages_ms"] = torch.tensor(
        report["selected_ages_ms"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    return batch


def make_policy(args, joint_limits, gripper_stroke_mm):
    config = EgoVerseHPTJointHistoryV2Config(
        ckpt_path=str(args.checkpoint),
        joint_limits_deg=joint_limits,
        gripper_stroke_mm=gripper_stroke_mm,
        device=args.device,
        action_horizon_frames=30,
        query_frequency=args.query_frequency,
        num_inference_steps=args.inference_steps,
        inference_seed=args.seed,
        require_residual_joint_head=not args.allow_legacy_joint_head,
        expected_image_height=480,
        expected_image_width=640,
        resize_mismatched_images=False,
        gripper_threshold=args.gripper_threshold,
        joint_limit_margin_rad=np.radians(args.joint_limit_margin_deg),
        max_joint_command_step=args.max_joint_command_step,
        max_joint_command_accel=args.max_joint_command_accel,
        temporal_ensemble=args.temporal_ensemble,
        temporal_replan_interval=args.temporal_replan_interval,
        temporal_ensemble_decay=args.temporal_ensemble_decay,
        chunk_smoothing=args.chunk_smoothing,
        chunk_smoothing_window=args.chunk_smoothing_window,
        chunk_smoothing_sigma=args.chunk_smoothing_sigma,
        observation_history=args.observation_history,
        n_obs_steps=args.observation_history,
        observation_history_step_ms=args.observation_history_step_ms,
        observation_history_max_skew_ms=args.observation_history_max_skew_ms,
    )
    return EgoVerseHPTJointHistoryV2Policy(config)


def parse_args():
    history_parser = argparse.ArgumentParser(add_help=False)
    history_parser.add_argument("--observation-history", type=int, default=4)
    history_parser.add_argument("--observation-history-step-ms", type=float, default=100.0)
    history_parser.add_argument("--observation-history-max-skew-ms", type=float, default=50.0)
    history_args, remaining = history_parser.parse_known_args()
    old_profiles = legacy.CHECKPOINT_PROFILES
    old_argv = sys.argv
    try:
        # Prevent selecting a one-frame checkpoint through the four-frame
        # entrypoint.  The model contract would reject it later, but keeping
        # the CLI choices disjoint makes profile switching unambiguous.
        legacy.CHECKPOINT_PROFILES = HISTORY_V2_CHECKPOINT_PROFILES
        sys.argv = [old_argv[0], *remaining]
        args = legacy.parse_args()
    finally:
        legacy.CHECKPOINT_PROFILES = old_profiles
        sys.argv = old_argv
    for key, value in vars(history_args).items():
        setattr(args, key, value)
    if args.observation_history != 4:
        raise ValueError("The initial History V2 checkpoints are fixed to four frames")
    if args.observation_history_step_ms <= 0 or args.observation_history_max_skew_ms <= 0:
        raise ValueError("history timing values must be positive")
    return args


def run_offline(args, policy, log, device):
    if args.offline_npz is None:
        raise ValueError("offline mode requires --offline-npz")
    keys = ("observation.state", "observation.images.435if", "observation.images.405")
    with np.load(args.offline_npz, allow_pickle=False) as archive:
        missing = [key for key in keys if key not in archive]
        if missing:
            raise KeyError(f"offline NPZ is missing {missing}; required={keys}")
        arrays = {key: archive[key] for key in keys}
    frame_count = min(args.max_steps, *(len(arrays[key]) for key in keys))
    stride = max(1, int(round(args.observation_history_step_ms * 30.0 / 1000.0)))
    for frame_index in range(frame_count):
        indices = [max(0, frame_index - stride * offset) for offset in range(3, -1, -1)]
        samples = []
        for index in indices:
            samples.append(
                {
                    **{f"joint_{joint}.pos": arrays[keys[0]][index][joint] for joint in range(7)},
                    "435if": arrays[keys[1]][index],
                    "405": arrays[keys[2]][index],
                }
            )
        action = policy.select_action(make_history_batch_from_samples(samples, device))
        log.write(
            {
                "mode": "offline",
                "frame": frame_index,
                "history_indices": indices,
                "action": action.tolist(),
                "queue_status": policy.queue_status(),
                "new_chunk_report": policy.last_prediction_report,
                "command_sent": False,
            }
        )


def main():
    global _ACTIVE_HISTORY
    args = parse_args()
    if args.list_checkpoint_profiles:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    actual_sha256 = legacy.sha256_file(args.checkpoint)
    if actual_sha256 != args.checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: expected {args.checkpoint_sha256}, got {actual_sha256}"
        )
    joint_limits, gripper_stroke_mm = legacy.load_calibration(args.calibration_json)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    policy = make_policy(args, joint_limits, gripper_stroke_mm)
    if args.mode == "checkpoint":
        legacy.logger.info("History V2 checkpoint loaded: %s", policy.checkpoint_report)
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = legacy.JsonlLogger(args.log_dir / f"history_v2_{args.mode}_{timestamp}.jsonl")
    try:
        log.write(
            {
                "event": "start",
                "mode": args.mode,
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": actual_sha256,
                "checkpoint_contract": policy.checkpoint_report,
                "observation_history": args.observation_history,
                "observation_history_step_ms": args.observation_history_step_ms,
                "observation_history_max_skew_ms": args.observation_history_max_skew_ms,
                "command_sent": False,
            }
        )
        if args.mode == "offline":
            run_offline(args, policy, log, device)
        else:
            _ACTIVE_HISTORY = CausalObservationHistoryV2(
                args.observation_history,
                args.observation_history_step_ms,
                args.observation_history_max_skew_ms,
            )
            legacy.make_robot = make_history_robot
            legacy.live_observation_to_batch = live_observation_to_history_batch
            legacy.run_live(
                args,
                policy,
                log,
                device,
                joint_limits,
                gripper_stroke_mm,
                timestamp,
            )
    finally:
        log.close()
        legacy.logger.info("wrote History V2 audit log to %s", log.path)


if __name__ == "__main__":
    main()
