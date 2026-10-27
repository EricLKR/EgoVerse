#!/usr/bin/env python
"""Asynchronous inference plus decoded-joint RTC for Assembly New1 HPT.

This is an additive deployment entrypoint.  It imports the audited synchronous
``deploy_assembly_new1_hpt_joint.py`` for checkpoint loading, calibration, cameras,
homing, command limits and the gripper state machine, but replaces only the
action-chunk scheduler:

* the first chunk is inferred synchronously and continuity-checked;
* later observations are copied into an owned snapshot and inferred in one
  background CUDA worker;
* every result is anchored to the successful action tick at capture time;
* actions that expired while inference was running are discarded;
* old/new overlap is blended with a cosine ramp in decoded physical 6-D joint
  radians (never in normalized driver space or gripper space);
* policy time advances only after a command is accepted by the driver's arm
  relative-target limiter;
* the existing binary, feedback-aware gripper state machine remains separate.

``checkpoint`` and ``offline`` never construct a robot. ``live-dry-run`` is
observation-only. ``execute`` still requires ``--allow-motion``.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import deploy_assembly_new1_hpt_joint as legacy
from cartesian_z_bias import CartesianZBiasTransform


logger = logging.getLogger("deploy_assembly_new1_hpt_joint_async_rtc")
CONTROL_PERIOD_S = 1.0 / 30.0
DEFAULT_PHASE_DIAGNOSTIC_PROFILE = Path(__file__).with_name(
    "assembly_second_pick_pose_profile.json"
)
DEFAULT_FIRST_GRASP_POSE_PROFILE = (
    Path(__file__).resolve().parents[1] / "assets/first_grasp_pose_profile.json"
)
DEFAULT_DELAYED_RELEASE_POSE_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "assets/gripper_delayed_release_pose_profile.json"
)


class PhaseGatedCartesianZBiasTransform:
    """Apply Cartesian-Z bias only before the first confirmed gripper close.

    The asynchronous worker may finish a prediction that was captured before
    the close transition. Every transformed report therefore retains the
    original decoded joint chunk, and the control thread reconciles returned
    reports against the current phase before installing them.
    """

    def __init__(
        self,
        transform: CartesianZBiasTransform,
        *,
        disable_after_first_close: bool,
    ) -> None:
        self.transform = transform
        self.disable_after_first_close = bool(disable_after_first_close)
        self._lock = threading.Lock()
        self._active = bool(transform.enabled)
        self._generation = 0
        self._disabled_action_tick: int | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._active)

    def settings_report(self) -> dict[str, Any]:
        with self._lock:
            active = bool(self._active)
            generation = int(self._generation)
            disabled_action_tick = self._disabled_action_tick
        report = self.transform.settings_report()
        report.update(
            {
                "phase_gated": bool(self.disable_after_first_close),
                "phase_gate_active": active,
                "phase_gate_generation": generation,
                "disable_after_first_close": bool(
                    self.disable_after_first_close
                ),
                "disabled_action_tick": disabled_action_tick,
            }
        )
        return report

    def disable_for_first_close(self, *, action_tick: int) -> dict[str, Any]:
        with self._lock:
            was_active = bool(self._active)
            if self.disable_after_first_close and self._active:
                self._active = False
                self._generation += 1
                self._disabled_action_tick = int(action_tick)
            return {
                "requested": bool(self.disable_after_first_close),
                "changed": bool(was_active and not self._active),
                "was_active": was_active,
                "active": bool(self._active),
                "generation": int(self._generation),
                "disabled_action_tick": self._disabled_action_tick,
            }

    def apply_to_prediction_report(
        self, prediction_report: dict[str, Any] | None
    ) -> dict[str, Any]:
        if prediction_report is None:
            raise RuntimeError("prediction report is missing")
        report = copy.deepcopy(prediction_report)
        original = np.asarray(
            report.get("selected_physical_actions"), dtype=np.float32
        )
        if (
            original.ndim != 2
            or original.shape[1] != 7
            or not np.isfinite(original).all()
        ):
            raise RuntimeError(
                "phase-gated Cartesian Z bias requires a finite (T,7) "
                f"decoded chunk, got {original.shape}"
            )
        report["selected_physical_actions_before_cartesian_z_bias"] = (
            original.tolist()
        )
        with self._lock:
            active_when_applied = bool(self._active)
            generation_when_applied = int(self._generation)
        if active_when_applied:
            report = self.transform.apply_to_prediction_report(report)
        else:
            bias_report = self.transform.settings_report()
            bias_report.update(
                {
                    "enabled": False,
                    "phase_gate_bypassed": True,
                    "requested_z_delta_mm": float(self.transform.bias_mm),
                    "applied_z_delta_mm": 0.0,
                    "applied_bias_fraction": 0.0,
                }
            )
            report["cartesian_z_bias"] = bias_report
        report["cartesian_z_bias_phase_gate"] = {
            "active_when_applied": active_when_applied,
            "generation_when_applied": generation_when_applied,
        }
        return report

    def reconcile_prediction_report(
        self, prediction_report: dict[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if prediction_report is None:
            raise RuntimeError("prediction report is missing")
        report = copy.deepcopy(prediction_report)
        with self._lock:
            active = bool(self._active)
            generation = int(self._generation)
        gate = dict(report.get("cartesian_z_bias_phase_gate") or {})
        transformed_generation = gate.get("generation_when_applied")
        reconciled = False
        if not active:
            original = np.asarray(
                report.get("selected_physical_actions_before_cartesian_z_bias"),
                dtype=np.float32,
            )
            if (
                original.ndim != 2
                or original.shape[1] != 7
                or not np.isfinite(original).all()
            ):
                raise RuntimeError(
                    "cannot remove post-grasp Z bias because the original "
                    "decoded chunk is missing"
                )
            current = np.asarray(
                report.get("selected_physical_actions"), dtype=np.float32
            )
            reconciled = bool(not np.array_equal(current, original))
            report["selected_physical_actions"] = original.tolist()
            bias_report = dict(report.get("cartesian_z_bias") or {})
            bias_report.update(
                {
                    "phase_gate_reconciled_to_unbiased": reconciled,
                    "applied_z_delta_mm_after_reconcile": 0.0,
                }
            )
            report["cartesian_z_bias"] = bias_report
        reconcile_report = {
            "phase_gate_active": active,
            "current_generation": generation,
            "transformed_generation": transformed_generation,
            "reconciled_to_unbiased": reconciled,
        }
        report["cartesian_z_bias_phase_gate_reconcile"] = reconcile_report
        return report, reconcile_report


def command_clip_report(
    proposed_action: dict[str, float],
    sent_action: dict[str, float],
    *,
    arm_tolerance: float,
) -> dict[str, Any]:
    """Classify material driver clipping without treating float noise as a clamp.

    PIPER actions use normalized driver position units. The robot layer may
    round-trip float32 commands through Python floats, producing differences of
    a few 1e-6 even when ``max_relative_target`` did not alter the command.
    Those numerical differences must not stop the absolute action clock.
    """
    arm_deltas = [
        abs(
            float(proposed_action[f"joint_{index}.pos"])
            - float(sent_action[f"joint_{index}.pos"])
        )
        for index in range(6)
    ]
    gripper_delta = abs(
        float(proposed_action["joint_6.pos"])
        - float(sent_action["joint_6.pos"])
    )
    return {
        "arm_clipped": max(arm_deltas, default=0.0) > arm_tolerance,
        "gripper_clipped": gripper_delta > arm_tolerance,
        "arm_max_abs_delta": max(arm_deltas, default=0.0),
        "arm_abs_deltas": arm_deltas,
        "gripper_abs_delta": gripper_delta,
        "tolerance": arm_tolerance,
    }


def copy_live_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Return a worker-owned CPU snapshot of one observation."""
    snapshot = {
        f"joint_{index}.pos": float(observation[f"joint_{index}.pos"])
        for index in range(7)
    }
    front = np.ascontiguousarray(observation["435if"], dtype=np.uint8).copy()
    wrist = np.ascontiguousarray(observation["405"], dtype=np.uint8).copy()
    side = np.ascontiguousarray(observation["side"], dtype=np.uint8).copy()
    if any(
        image.shape != (480, 640, 3)
        for image in (front, wrist, side)
    ):
        raise RuntimeError(
            "unexpected live RGB shape while making an async snapshot: "
            f"front={front.shape}, wrist={wrist.shape}, side={side.shape}"
        )
    snapshot["435if"] = front
    snapshot["405"] = wrist
    snapshot["side"] = side
    snapshot["_captured_ns"] = time.monotonic_ns()
    return snapshot


def raw_diagnostic_image_views(
    observation: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Default diagnostic views when no deployment color transform is active."""
    return {
        "front_raw": np.asarray(observation["435if"]),
        "wrist_raw": np.asarray(observation["405"]),
        "side_raw": np.asarray(observation["side"]),
        "front_model": np.asarray(observation["435if"]),
        "wrist_model": np.asarray(observation["405"]),
        "side_model": np.asarray(observation["side"]),
    }


@dataclass(frozen=True)
class DiagnosticImageRequest:
    sample_id: int
    frame: int
    action_tick: int
    request_id: int
    observation: dict[str, Any]


class AsyncDiagnosticImageWriter:
    """Bounded, non-blocking image logger isolated from the control loop."""

    _STOP = object()

    def __init__(
        self,
        *,
        output_dir: Path,
        view_builder: Callable[[dict[str, Any]], dict[str, np.ndarray]],
        queue_size: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.view_builder = view_builder
        self._requests: queue.Queue[Any] = queue.Queue(maxsize=int(queue_size))
        self._results: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._errors = 0
        self._thread = threading.Thread(
            target=self._run,
            name="assembly-hpt-async-diagnostic-images",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        from PIL import Image

        while True:
            request = self._requests.get()
            try:
                if request is self._STOP:
                    return
                started = time.perf_counter()
                self.output_dir.mkdir(parents=True, exist_ok=True)
                views = self.view_builder(request.observation)
                paths: dict[str, str] = {}
                for name, value in views.items():
                    image = np.asarray(value)
                    if image.ndim == 3 and image.shape[0] in (3, 4):
                        image = np.moveaxis(image[:3], 0, -1)
                    if image.shape != (480, 640, 3):
                        raise RuntimeError(
                            f"invalid diagnostic image {name}: {image.shape}"
                        )
                    image = np.clip(image, 0, 255).astype(np.uint8)
                    path = self.output_dir / (
                        f"sample_{request.sample_id:05d}_frame_{request.frame:06d}_"
                        f"tick_{request.action_tick:06d}_{name}.jpg"
                    )
                    Image.fromarray(image).save(
                        path, format="JPEG", quality=92, subsampling=0
                    )
                    paths[name] = str(path)
                self._written += 1
                self._results.put(
                    {
                        "sample_id": request.sample_id,
                        "frame": request.frame,
                        "action_tick": request.action_tick,
                        "request_id": request.request_id,
                        "paths": paths,
                        "write_ms": (time.perf_counter() - started) * 1e3,
                        "error": None,
                    }
                )
            except BaseException as exc:
                self._errors += 1
                if request is not self._STOP:
                    self._results.put(
                        {
                            "sample_id": request.sample_id,
                            "frame": request.frame,
                            "action_tick": request.action_tick,
                            "request_id": request.request_id,
                            "paths": {},
                            "write_ms": None,
                            "error": repr(exc),
                        }
                    )
            finally:
                self._requests.task_done()

    def submit(self, request: DiagnosticImageRequest) -> dict[str, Any]:
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            self._dropped += 1
            return {
                "scheduled": False,
                "reason": "diagnostic_image_queue_full",
                "sample_id": request.sample_id,
                "frame": request.frame,
                "action_tick": request.action_tick,
                "request_id": request.request_id,
            }
        self._submitted += 1
        return {
            "scheduled": True,
            "reason": None,
            "sample_id": request.sample_id,
            "frame": request.frame,
            "action_tick": request.action_tick,
            "request_id": request.request_id,
        }

    def poll_all(self) -> list[dict[str, Any]]:
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def status(self) -> dict[str, Any]:
        return {
            "worker_alive": self._thread.is_alive(),
            "queue_depth": self._requests.qsize(),
            "submitted": self._submitted,
            "written": self._written,
            "dropped": self._dropped,
            "errors": self._errors,
        }

    def close(self) -> list[dict[str, Any]]:
        if self._thread.is_alive():
            self._requests.put(self._STOP, timeout=2.0)
            self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise RuntimeError("diagnostic image writer did not stop within 30 seconds")
        return self.poll_all()


@dataclass(frozen=True)
class FullRunImageRequest:
    frame: int
    action_tick: int
    captured_ns: int
    wall_time_utc: str
    observation: dict[str, Any]


class AsyncFullRunImageWriter:
    """Record every control-loop camera triplet without blocking control.

    The writer owns copied uint8 observations. A bounded queue prevents an
    unexpectedly slow disk from consuming unbounded RAM; a full queue records
    an explicit dropped frame instead of delaying a robot command.
    """

    _STOP = object()
    _CAMERAS = {
        "front": "435if",
        "wrist": "405",
        "side": "side",
    }

    def __init__(
        self,
        *,
        output_dir: Path,
        queue_size: int,
        jpeg_quality: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.jpeg_quality = int(jpeg_quality)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for camera_name in self._CAMERAS:
            (self.output_dir / camera_name).mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.output_dir / "frames.jsonl"
        self._requests: queue.Queue[Any] = queue.Queue(maxsize=int(queue_size))
        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._errors = 0
        self._first_error: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="assembly-hpt-full-run-images",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        from PIL import Image

        with self._metadata_path.open("a", encoding="utf-8") as metadata:
            while True:
                request = self._requests.get()
                try:
                    if request is self._STOP:
                        return
                    started = time.perf_counter()
                    paths: dict[str, str] = {}
                    for camera_name, observation_key in self._CAMERAS.items():
                        image = np.asarray(request.observation[observation_key])
                        if image.shape != (480, 640, 3):
                            raise RuntimeError(
                                f"invalid full-run {camera_name} image: {image.shape}"
                            )
                        path = (
                            self.output_dir
                            / camera_name
                            / f"frame_{request.frame:06d}.jpg"
                        )
                        Image.fromarray(
                            np.ascontiguousarray(image, dtype=np.uint8)
                        ).save(
                            path,
                            format="JPEG",
                            quality=self.jpeg_quality,
                            subsampling=2,
                        )
                        paths[camera_name] = str(path)
                    write_ms = (time.perf_counter() - started) * 1e3
                    metadata.write(
                        json.dumps(
                            {
                                "frame": int(request.frame),
                                "action_tick": int(request.action_tick),
                                "captured_monotonic_ns": int(request.captured_ns),
                                "wall_time_utc": request.wall_time_utc,
                                "paths": paths,
                                "write_ms": write_ms,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    metadata.flush()
                    self._written += 1
                except BaseException as exc:
                    self._errors += 1
                    if self._first_error is None:
                        self._first_error = repr(exc)
                finally:
                    self._requests.task_done()

    def submit(self, request: FullRunImageRequest) -> dict[str, Any]:
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            self._dropped += 1
            return {
                "scheduled": False,
                "reason": "full_run_image_queue_full",
                "frame": int(request.frame),
                "action_tick": int(request.action_tick),
            }
        self._submitted += 1
        return {
            "scheduled": True,
            "reason": None,
            "frame": int(request.frame),
            "action_tick": int(request.action_tick),
        }

    def status(self) -> dict[str, Any]:
        return {
            "worker_alive": self._thread.is_alive(),
            "queue_depth": self._requests.qsize(),
            "submitted": int(self._submitted),
            "written": int(self._written),
            "dropped": int(self._dropped),
            "errors": int(self._errors),
            "first_error": self._first_error,
            "output_dir": str(self.output_dir),
            "metadata_path": str(self._metadata_path),
        }

    def close(self) -> dict[str, Any]:
        if self._thread.is_alive():
            # Motion has already stopped here, so waiting for queue capacity is
            # preferable to silently discarding the final camera frames.
            self._requests.put(self._STOP)
            self._thread.join(timeout=120.0)
        if self._thread.is_alive():
            raise RuntimeError("full-run image writer did not stop within 120 seconds")
        final_status = self.status()
        (self.output_dir / "summary.json").write_text(
            json.dumps(final_status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return final_status


@dataclass(frozen=True)
class PlanDiagnosticRequest:
    plan_id: int
    frame: int
    request_tick: int
    install_tick: int
    observation: dict[str, Any]
    payload: dict[str, Any]


def _json_ready(value: Any) -> Any:
    """Convert diagnostic numpy values without changing the control objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


class AsyncPlanDiagnosticWriter:
    """Persist complete RTC plans and their exact model-input images off-loop."""

    _STOP = object()

    def __init__(
        self,
        *,
        output_dir: Path,
        view_builder: Callable[[dict[str, Any]], dict[str, np.ndarray]],
        queue_size: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.view_builder = view_builder
        self._requests: queue.Queue[Any] = queue.Queue(maxsize=int(queue_size))
        self._results: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._errors = 0
        self._thread = threading.Thread(
            target=self._run,
            name="assembly-hpt-async-plan-diagnostics",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        from PIL import Image

        while True:
            request = self._requests.get()
            try:
                if request is self._STOP:
                    return
                started = time.perf_counter()
                plan_dir = self.output_dir / f"plan_{request.plan_id:05d}"
                plan_dir.mkdir(parents=True, exist_ok=True)
                image_paths: dict[str, str] = {}
                for name, value in self.view_builder(request.observation).items():
                    image = np.asarray(value)
                    if image.ndim == 3 and image.shape[0] in (3, 4):
                        image = np.moveaxis(image[:3], 0, -1)
                    if image.shape != (480, 640, 3):
                        raise RuntimeError(
                            f"invalid plan diagnostic image {name}: {image.shape}"
                        )
                    path = plan_dir / f"{name}.jpg"
                    Image.fromarray(
                        np.clip(image, 0, 255).astype(np.uint8)
                    ).save(path, format="JPEG", quality=95, subsampling=0)
                    image_paths[name] = str(path)

                payload = _json_ready(request.payload)
                payload.update(
                    {
                        "artifact_schema": "assembly_hpt_rtc_plan_diagnostic_v1",
                        "plan_id": int(request.plan_id),
                        "frame": int(request.frame),
                        "request_tick": int(request.request_tick),
                        "install_tick": int(request.install_tick),
                        "image_paths": image_paths,
                    }
                )
                json_path = plan_dir / "plan.json"
                json_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self._written += 1
                self._results.put(
                    {
                        "plan_id": int(request.plan_id),
                        "frame": int(request.frame),
                        "request_tick": int(request.request_tick),
                        "install_tick": int(request.install_tick),
                        "json_path": str(json_path),
                        "image_paths": image_paths,
                        "write_ms": (time.perf_counter() - started) * 1e3,
                        "error": None,
                    }
                )
            except BaseException as exc:
                self._errors += 1
                if request is not self._STOP:
                    self._results.put(
                        {
                            "plan_id": int(request.plan_id),
                            "frame": int(request.frame),
                            "request_tick": int(request.request_tick),
                            "install_tick": int(request.install_tick),
                            "json_path": None,
                            "image_paths": {},
                            "write_ms": None,
                            "error": repr(exc),
                        }
                    )
            finally:
                self._requests.task_done()

    def submit(self, request: PlanDiagnosticRequest) -> dict[str, Any]:
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            self._dropped += 1
            return {
                "scheduled": False,
                "reason": "plan_diagnostic_queue_full",
                "plan_id": int(request.plan_id),
            }
        self._submitted += 1
        return {
            "scheduled": True,
            "reason": None,
            "plan_id": int(request.plan_id),
        }

    def poll_all(self) -> list[dict[str, Any]]:
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def status(self) -> dict[str, Any]:
        return {
            "worker_alive": self._thread.is_alive(),
            "queue_depth": self._requests.qsize(),
            "submitted": self._submitted,
            "written": self._written,
            "dropped": self._dropped,
            "errors": self._errors,
        }

    def close(self) -> list[dict[str, Any]]:
        if self._thread.is_alive():
            self._requests.put(self._STOP, timeout=2.0)
            self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise RuntimeError("plan diagnostic writer did not stop within 30 seconds")
        return self.poll_all()


class SecondPickPhaseDiagnostic:
    """Observation-only event-aligned diagnostic; never gates or changes motion."""

    def __init__(self, profile_path: Path) -> None:
        self.profile_path = Path(profile_path)
        profile = json.loads(self.profile_path.read_text())
        if profile.get("schema") != "assembly_second_pick_pose_profile_v1":
            raise ValueError(
                f"unsupported phase diagnostic profile: {profile.get('schema')!r}"
            )
        self.profile = profile
        self.release_count = 0
        self.active = False
        self.completed = False
        self.release_frame: int | None = None
        self.release_action_tick: int | None = None
        self.release_elapsed_s: float | None = None

    @staticmethod
    def _validate_joints(joints_deg: Any) -> np.ndarray:
        joints = np.asarray(joints_deg, dtype=np.float64)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError(f"invalid diagnostic joint vector: {joints_deg!r}")
        return joints

    def update(
        self,
        *,
        frame: int,
        action_tick: int,
        elapsed_s: float,
        joints_deg: Any,
        gripper_transition: str | None,
    ) -> dict[str, Any]:
        joints = self._validate_joints(joints_deg)
        event = None
        if gripper_transition == "close->open":
            self.release_count += 1
            if self.release_count == 1:
                self.active = True
                self.completed = False
                self.release_frame = int(frame)
                self.release_action_tick = int(action_tick)
                self.release_elapsed_s = float(elapsed_s)
                event = "first_release_detected"
        elif (
            gripper_transition == "open->close"
            and self.active
            and self.release_count == 1
        ):
            self.active = False
            self.completed = True
            event = "second_close_detected"

        reference = self.profile["second_close_joint_deg"]
        p05 = self._validate_joints(reference["p05"])
        median = self._validate_joints(reference["median"])
        p95 = self._validate_joints(reference["p95"])
        error = joints - median
        within = (joints >= p05) & (joints <= p95)
        seconds_after_release = (
            None
            if self.release_elapsed_s is None
            else max(0.0, float(elapsed_s) - self.release_elapsed_s)
        )
        timing = self.profile["release_to_second_close_s"]
        return {
            "enabled": True,
            "profile": str(self.profile_path),
            "phase": (
                "second_pick_active"
                if self.active
                else "second_pick_complete"
                if self.completed
                else "before_first_release"
            ),
            "event": event,
            "release_count": int(self.release_count),
            "release_frame": self.release_frame,
            "release_action_tick": self.release_action_tick,
            "seconds_after_release": seconds_after_release,
            "training_second_close_time_s": timing,
            "current_joint_deg": joints.tolist(),
            "second_close_median_joint_deg": median.tolist(),
            "error_to_second_close_median_deg": error.tolist(),
            "abs_error_to_second_close_median_deg": np.abs(error).tolist(),
            "rms_error_to_second_close_median_deg": float(
                np.sqrt(np.mean(np.square(error)))
            ),
            "max_abs_error_to_second_close_median_deg": float(np.max(np.abs(error))),
            "within_second_close_training_5_95": within.tolist(),
            "all_joints_within_second_close_training_5_95": bool(np.all(within)),
            "past_training_median_time": bool(
                seconds_after_release is not None
                and seconds_after_release >= float(timing["median"])
            ),
            "past_training_p95_time": bool(
                seconds_after_release is not None
                and seconds_after_release >= float(timing["p95"])
            ),
            "diagnostic_only_no_control_effect": True,
        }


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    request_tick: int
    request_frame: int
    observation: dict[str, Any]
    submitted_ns: int
    snapshot_ms: float
    gripper_policy_override_fraction: float | None = None


@dataclass(frozen=True)
class InferenceResult:
    request_id: int
    request_tick: int
    request_frame: int
    captured_ns: int
    finished_ns: int
    actions: np.ndarray | None
    prediction_report: dict[str, Any] | None
    observation: dict[str, Any] | None
    timing: dict[str, float]
    error: BaseException | None


def _merge_gripper_only_counterfactual(
    *,
    normal_actions: np.ndarray,
    normal_report: dict[str, Any],
    counterfactual_actions: np.ndarray,
    counterfactual_report: dict[str, Any],
    override_fraction: float,
    override_hardware_raw: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use counterfactual gripper predictions without changing the arm path.

    The HPT decoder conditions both heads on proprioception, so changing only
    the observed gripper aperture can also move the six predicted joints.  The
    normal inference therefore remains authoritative for all arm columns.  A
    second inference with the training-lower-bound aperture contributes only
    column 6 and the full 30-step gripper forecast.
    """

    merged_actions = np.asarray(normal_actions, dtype=np.float32).copy()
    alternate_actions = np.asarray(counterfactual_actions, dtype=np.float32)
    if (
        merged_actions.ndim != 2
        or merged_actions.shape[1] != 7
        or alternate_actions.shape != merged_actions.shape
    ):
        raise RuntimeError(
            "gripper-only counterfactual requires matching (T,7) action chunks"
        )
    merged_actions[:, 6] = alternate_actions[:, 6]
    merged_report = copy.deepcopy(normal_report)

    seven_column_keys = (
        "full_model_chunk_normalized",
        "full_model_chunk_unnormalized",
        "full_model_chunk_smoothed",
        "selected_physical_actions_unsmoothed",
        "selected_physical_actions",
        "selected_raw_driver_actions",
        "selected_driver_actions",
    )
    for key in seven_column_keys:
        if key not in merged_report or key not in counterfactual_report:
            continue
        normal = np.asarray(merged_report[key], dtype=np.float32)
        alternate = np.asarray(counterfactual_report[key], dtype=np.float32)
        if (
            normal.ndim != 2
            or normal.shape[1] != 7
            or alternate.shape != normal.shape
        ):
            raise RuntimeError(
                f"invalid {key} while merging gripper-only counterfactual: "
                f"normal={normal.shape}, alternate={alternate.shape}"
            )
        normal[:, 6] = alternate[:, 6]
        merged_report[key] = normal.tolist()

    for key in (
        "full_control_gripper_raw",
        "full_control_gripper_commands_pct",
    ):
        if key in counterfactual_report:
            merged_report[key] = copy.deepcopy(counterfactual_report[key])

    normal_physical = np.asarray(
        normal_report.get("selected_physical_actions"), dtype=np.float64
    )
    alternate_physical = np.asarray(
        counterfactual_report.get("selected_physical_actions"), dtype=np.float64
    )
    joint_rms_deg = None
    joint_max_abs_deg = None
    if (
        normal_physical.ndim == 2
        and normal_physical.shape[1] == 7
        and alternate_physical.shape == normal_physical.shape
    ):
        joint_delta_deg = np.rad2deg(
            alternate_physical[:, :6] - normal_physical[:, :6]
        )
        joint_rms_deg = float(np.sqrt(np.mean(np.square(joint_delta_deg))))
        joint_max_abs_deg = float(np.max(np.abs(joint_delta_deg)))

    merged_report["gripper_delayed_release_assist"] = {
        "active": True,
        "policy_override_fraction": float(override_fraction),
        "policy_override_hardware_raw": float(override_hardware_raw),
        "arm_source": "normal_real_gripper_feedback_inference",
        "gripper_source": "counterfactual_training_lower_bound_inference",
        "counterfactual_joint_rms_deg_not_executed": joint_rms_deg,
        "counterfactual_joint_max_abs_deg_not_executed": joint_max_abs_deg,
        "counterfactual_proprio_physical": copy.deepcopy(
            counterfactual_report.get("proprio_physical")
        ),
        "normal_proprio_physical": copy.deepcopy(
            normal_report.get("proprio_physical")
        ),
    }
    return merged_actions, merged_report


class BackgroundPredictor:
    """Single-flight inference worker; the 30 Hz loop never waits for CUDA."""

    _STOP = object()

    def __init__(
        self,
        policy,
        device: torch.device,
        adapter=None,
        j2_command_offset_deg: float = 0.0,
        j3_command_offset_deg: float = 0.0,
        j5_command_offset_deg: float = 0.0,
        prediction_transform: PhaseGatedCartesianZBiasTransform | None = None,
    ) -> None:
        self.policy = policy
        self.device = device
        self.adapter = adapter
        self.j2_command_offset_deg = float(j2_command_offset_deg)
        self.j3_command_offset_deg = float(j3_command_offset_deg)
        self.j5_command_offset_deg = float(j5_command_offset_deg)
        self.prediction_transform = prediction_transform
        self._requests: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._results: queue.SimpleQueue[InferenceResult] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._queued_request_id: int | None = None
        self._running_request_id: int | None = None
        self._result_pending = False
        self._fatal_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="assembly-hpt-async-inference",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        if self.device.type == "cuda":
            device_index = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            try:
                torch.cuda.set_device(device_index)
            except BaseException as exc:
                with self._lock:
                    self._fatal_error = exc
                return
        while True:
            request = self._requests.get()
            try:
                if request is self._STOP:
                    return
                with self._lock:
                    self._queued_request_id = None
                    self._running_request_id = request.request_id
                started_ns = time.monotonic_ns()
                batch_started = time.perf_counter()
                model_input_observation = legacy.hardware_observation_for_model(
                    request.observation,
                    self.j2_command_offset_deg,
                    self.j3_command_offset_deg,
                    self.j5_command_offset_deg,
                )
                if self.adapter is None:
                    batch = legacy.live_observation_to_batch(
                        model_input_observation, self.device
                    )
                else:
                    batch = legacy.live_observation_to_batch(
                        model_input_observation, self.device, self.adapter
                    )
                batch_ms = (time.perf_counter() - batch_started) * 1e3
                inference_started = time.perf_counter()
                with torch.inference_mode():
                    actions = (
                        self.policy.predict_action_chunk(batch)
                        .detach()
                        .cpu()
                        .numpy()
                        .copy()
                    )
                policy_ms = (time.perf_counter() - inference_started) * 1e3
                report = copy.deepcopy(self.policy.last_prediction_report)
                release_assist_ms = 0.0
                release_assist_enabled = (
                    request.gripper_policy_override_fraction is not None
                )
                if release_assist_enabled:
                    if self.adapter is None:
                        raise RuntimeError(
                            "delayed gripper release assist requires a joint-space adapter"
                        )
                    fraction = float(request.gripper_policy_override_fraction)
                    if not 0.0 <= fraction <= 1.0 or not math.isfinite(fraction):
                        raise RuntimeError(
                            "invalid gripper policy override fraction: "
                            f"{request.gripper_policy_override_fraction}"
                        )
                    counterfactual_observation = dict(request.observation)
                    counterfactual_hardware_raw = float(
                        fraction
                        * self.adapter.hardware_gripper_stroke_mm
                        * 1000.0
                    )
                    counterfactual_observation["joint_6.pos"] = (
                        counterfactual_hardware_raw
                    )
                    counterfactual_model_input = (
                        legacy.hardware_observation_for_model(
                            counterfactual_observation,
                            self.j2_command_offset_deg,
                            self.j3_command_offset_deg,
                            self.j5_command_offset_deg,
                        )
                    )
                    counterfactual_batch = legacy.live_observation_to_batch(
                        counterfactual_model_input,
                        self.device,
                        self.adapter,
                    )
                    release_assist_started = time.perf_counter()
                    with torch.inference_mode():
                        counterfactual_actions = (
                            self.policy.predict_action_chunk(counterfactual_batch)
                            .detach()
                            .cpu()
                            .numpy()
                            .copy()
                        )
                    counterfactual_report = copy.deepcopy(
                        self.policy.last_prediction_report
                    )
                    release_assist_ms = (
                        time.perf_counter() - release_assist_started
                    ) * 1e3
                    normal_report_for_policy = copy.deepcopy(report)
                    actions, report = _merge_gripper_only_counterfactual(
                        normal_actions=actions,
                        normal_report=report,
                        counterfactual_actions=counterfactual_actions,
                        counterfactual_report=counterfactual_report,
                        override_fraction=fraction,
                        override_hardware_raw=counterfactual_hardware_raw,
                    )
                    # Keep the policy object's diagnostic event aligned with
                    # the authoritative real-feedback arm inference.  The
                    # merged report above is owned by this worker result.
                    self.policy.last_prediction_report = copy.deepcopy(
                        normal_report_for_policy
                    )
                postprocess_started = time.perf_counter()
                if self.prediction_transform is not None:
                    report = self.prediction_transform.apply_to_prediction_report(
                        report
                    )
                postprocess_ms = (time.perf_counter() - postprocess_started) * 1e3
                result = InferenceResult(
                    request_id=request.request_id,
                    request_tick=request.request_tick,
                    request_frame=request.request_frame,
                    captured_ns=int(request.observation["_captured_ns"]),
                    finished_ns=time.monotonic_ns(),
                    actions=actions,
                    prediction_report=report,
                    observation=request.observation,
                    timing={
                        "queue_wait_ms": (started_ns - request.submitted_ns) / 1e6,
                        "snapshot_ms": float(request.snapshot_ms),
                        "batch_transfer_ms": batch_ms,
                        "policy_total_ms": policy_ms,
                        "gripper_release_assist_enabled": bool(
                            release_assist_enabled
                        ),
                        "gripper_release_assist_policy_ms": release_assist_ms,
                        "cartesian_z_bias_ms": postprocess_ms,
                        "model_inference_ms": float(
                            (report or {}).get("model_inference_ms", 0.0)
                        ),
                    },
                    error=None,
                )
            except BaseException as exc:
                result = InferenceResult(
                    request_id=request.request_id,
                    request_tick=request.request_tick,
                    request_frame=request.request_frame,
                    captured_ns=int(request.observation["_captured_ns"]),
                    finished_ns=time.monotonic_ns(),
                    actions=None,
                    prediction_report=None,
                    observation=None,
                    timing={},
                    error=exc,
                )
            finally:
                if request is not self._STOP:
                    with self._lock:
                        self._running_request_id = None
                        self._result_pending = True
                    self._results.put(result)
                self._requests.task_done()

    def can_submit(self) -> bool:
        with self._lock:
            if self._fatal_error is not None:
                raise RuntimeError("async inference worker failed") from self._fatal_error
            return (
                self._queued_request_id is None
                and self._running_request_id is None
                and not self._result_pending
            )

    def submit(self, request: InferenceRequest) -> bool:
        with self._lock:
            if self._fatal_error is not None:
                raise RuntimeError("async inference worker failed") from self._fatal_error
            if (
                self._queued_request_id is not None
                or self._running_request_id is not None
                or self._result_pending
            ):
                return False
            self._queued_request_id = request.request_id
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            with self._lock:
                self._queued_request_id = None
            return False
        return True

    def poll(self) -> InferenceResult | None:
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            self._result_pending = False
        return result

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queued_request_id": self._queued_request_id,
                "running_request_id": self._running_request_id,
                "result_pending": self._result_pending,
                "worker_alive": self._thread.is_alive(),
                "fatal_error": repr(self._fatal_error) if self._fatal_error else None,
            }

    def close(self) -> None:
        if not self._thread.is_alive():
            return
        self._requests.put(self._STOP, timeout=2.0)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("async inference worker did not stop within 10 seconds")


class DelayedLowerBoundReleaseAssist:
    """Gate a gripper-only counterfactual on time, contact and release pose.

    The gate counts successful policy action ticks, not wall-clock control
    loops.  This prevents a retimer stall from consuming the training-derived
    close-to-release delay.  It never alters measured feedback or joint
    commands; it only requests an auxiliary gripper inference after every
    release condition has been satisfied.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        profile_path: Path,
        policy_fraction: float,
        minimum_close_action_ticks: list[int],
        max_nearest_pose_rms_deg: float,
        max_joint_motion_rms_deg: float,
        stable_frames: int,
        healthy_frames: int,
    ) -> None:
        self.enabled = bool(enabled)
        self.profile_path = Path(profile_path)
        self.policy_fraction = float(policy_fraction)
        self.minimum_close_action_ticks = [
            int(value) for value in minimum_close_action_ticks
        ]
        self.max_nearest_pose_rms_deg = float(max_nearest_pose_rms_deg)
        self.max_joint_motion_rms_deg = float(max_joint_motion_rms_deg)
        self.required_stable_frames = int(stable_frames)
        self.required_healthy_frames = int(healthy_frames)
        self.release_samples_deg: list[np.ndarray] = []
        if self.enabled:
            profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
            if profile.get("schema") != "assembly_gripper_release_pose_profile_v1":
                raise RuntimeError(
                    f"unsupported delayed-release profile: {self.profile_path}"
                )
            phases = profile.get("phases")
            if not isinstance(phases, list) or len(phases) != 3:
                raise RuntimeError(
                    "delayed-release profile must contain exactly three phases"
                )
            for phase in phases:
                samples = np.asarray(phase.get("samples_deg"), dtype=np.float64)
                if (
                    samples.ndim != 2
                    or samples.shape[1] != 6
                    or samples.shape[0] < 2
                    or not np.isfinite(samples).all()
                ):
                    raise RuntimeError(
                        "invalid delayed-release joint samples in "
                        f"{self.profile_path}"
                    )
                self.release_samples_deg.append(samples)
        self.close_count = 0
        self.close_action_tick: int | None = None
        self.contact_latched = False
        self.active = False
        self.stable_streak = 0
        self.healthy_streak = 0
        self.previous_joints_deg: np.ndarray | None = None
        self.last_report: dict[str, Any] = self._base_report()

    def _base_report(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "active": bool(self.active),
            "close_count": int(self.close_count),
            "phase_index": (
                int(self.close_count) if 1 <= self.close_count <= 3 else None
            ),
            "close_action_tick": self.close_action_tick,
            "contact_latched": bool(self.contact_latched),
            "policy_override_fraction": (
                float(self.policy_fraction) if self.active else None
            ),
            "profile_path": str(self.profile_path),
        }

    def current_override_fraction(self) -> float | None:
        if self.enabled and self.active:
            return float(self.policy_fraction)
        return None

    def observe(
        self,
        *,
        action_tick: int,
        current_model_joint_deg: Any,
        gripper_state: dict[str, Any],
        arm_clipped_or_stalled: bool,
    ) -> dict[str, Any]:
        if not self.enabled:
            self.last_report = self._base_report()
            return copy.deepcopy(self.last_report)

        joints = np.asarray(current_model_joint_deg, dtype=np.float64).reshape(-1)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise RuntimeError(
                "delayed release assist requires six finite model-space joints"
            )
        transition = gripper_state.get("transition")
        if transition in {"hold->close", "open->close"}:
            self.close_count += 1
            self.close_action_tick = int(action_tick)
            self.contact_latched = False
            self.active = False
            self.stable_streak = 0
            self.healthy_streak = 0
            self.previous_joints_deg = joints.copy()
        elif transition == "close->open":
            self.active = False
            self.contact_latched = False

        self.contact_latched = bool(
            self.contact_latched
            or gripper_state.get("object_contact_latched", False)
        )
        if self.previous_joints_deg is None:
            joint_motion_rms_deg = None
            self.stable_streak = 0
        else:
            joint_motion_rms_deg = float(
                np.sqrt(np.mean(np.square(joints - self.previous_joints_deg)))
            )
            if joint_motion_rms_deg <= self.max_joint_motion_rms_deg:
                self.stable_streak += 1
            else:
                self.stable_streak = 0
        self.previous_joints_deg = joints.copy()

        if arm_clipped_or_stalled:
            self.healthy_streak = 0
        else:
            self.healthy_streak += 1

        phase_index = self.close_count if 1 <= self.close_count <= 3 else None
        elapsed_ticks = (
            None
            if self.close_action_tick is None
            else int(action_tick - self.close_action_tick)
        )
        minimum_ticks = (
            None
            if phase_index is None
            else int(self.minimum_close_action_ticks[phase_index - 1])
        )
        nearest_pose_rms_deg = None
        nearest_pose_sample_index = None
        pose_gate_passed = False
        if phase_index is not None:
            samples = self.release_samples_deg[phase_index - 1]
            sample_rms = np.sqrt(
                np.mean(np.square(samples - joints[None, :]), axis=1)
            )
            nearest_pose_sample_index = int(np.argmin(sample_rms))
            nearest_pose_rms_deg = float(sample_rms[nearest_pose_sample_index])
            pose_gate_passed = bool(
                nearest_pose_rms_deg <= self.max_nearest_pose_rms_deg
            )

        time_gate_passed = bool(
            elapsed_ticks is not None
            and minimum_ticks is not None
            and elapsed_ticks >= minimum_ticks
        )
        stable_gate_passed = self.stable_streak >= self.required_stable_frames
        healthy_gate_passed = self.healthy_streak >= self.required_healthy_frames
        state_gate_passed = gripper_state.get("state") == "close"
        newly_activated = bool(
            not self.active
            and phase_index is not None
            and state_gate_passed
            and self.contact_latched
            and time_gate_passed
            and pose_gate_passed
            and stable_gate_passed
            and healthy_gate_passed
        )
        if newly_activated:
            self.active = True

        self.last_report = {
            **self._base_report(),
            "newly_activated": newly_activated,
            "action_tick": int(action_tick),
            "elapsed_close_action_ticks": elapsed_ticks,
            "minimum_close_action_ticks": minimum_ticks,
            "time_gate_passed": time_gate_passed,
            "state_gate_passed": state_gate_passed,
            "nearest_release_pose_sample_index": nearest_pose_sample_index,
            "nearest_release_pose_rms_deg": nearest_pose_rms_deg,
            "max_nearest_release_pose_rms_deg": float(
                self.max_nearest_pose_rms_deg
            ),
            "pose_gate_passed": pose_gate_passed,
            "joint_motion_rms_deg": joint_motion_rms_deg,
            "max_joint_motion_rms_deg": float(self.max_joint_motion_rms_deg),
            "stable_streak": int(self.stable_streak),
            "required_stable_frames": int(self.required_stable_frames),
            "stable_gate_passed": bool(stable_gate_passed),
            "healthy_streak": int(self.healthy_streak),
            "required_healthy_frames": int(self.required_healthy_frames),
            "healthy_gate_passed": bool(healthy_gate_passed),
            "arm_clipped_or_stalled": bool(arm_clipped_or_stalled),
            "current_model_joint_deg": joints.tolist(),
        }
        return copy.deepcopy(self.last_report)


@dataclass
class ActivePlan:
    plan_id: int
    start_tick: int
    physical_actions: np.ndarray
    prediction_report: dict[str, Any]


@dataclass
class BufferedPlan:
    """A time-aligned result waiting for the active chunk boundary."""

    plan_id: int
    start_tick: int
    physical_actions: np.ndarray
    prediction_report: dict[str, Any]
    buffered_tick: int
    result_age_ms: float


def physical_chunk_from_report(
    prediction_report: dict[str, Any] | None, expected_steps: int
) -> np.ndarray:
    if prediction_report is None:
        raise RuntimeError("RTC requires a prediction report")
    physical = np.asarray(
        prediction_report.get("selected_physical_actions"), dtype=np.float32
    )
    expected = (int(expected_steps), 7)
    if physical.shape != expected or not np.isfinite(physical).all():
        raise RuntimeError(
            "RTC requires selected decoded physical actions: "
            f"expected={expected}, got={physical.shape}"
        )
    return physical.copy()


class DecodedJointRtcController:
    """Absolute-time action controller with decoded-joint overlap blending."""

    def __init__(
        self,
        policy,
        min_suffix_actions: int,
        blend_actions: int,
        *,
        finish_active_plan_before_install: bool = False,
    ) -> None:
        self.policy = policy
        self.min_suffix_actions = int(min_suffix_actions)
        self.blend_actions = int(blend_actions)
        self.finish_active_plan_before_install = bool(
            finish_active_plan_before_install
        )
        self.action_tick = 0
        self.active_plan: ActivePlan | None = None
        self.buffered_plan: BufferedPlan | None = None
        self._forced_replan_reason: str | None = None
        self._last_boundary_activation: dict[str, Any] | None = None
        self._last_velocity = np.asarray(
            policy._last_planned_velocity_driver, dtype=np.float64
        ).copy()
        self._pending_velocity: np.ndarray | None = None

    @staticmethod
    def _take_new_weight(index: int, count: int) -> float:
        phase = float(index + 1) / float(count + 1)
        return 0.5 - 0.5 * math.cos(math.pi * phase)

    def install(
        self,
        *,
        plan_id: int,
        start_tick: int,
        actions: np.ndarray,
        prediction_report: dict[str, Any] | None,
        result_age_ms: float,
        max_result_age_ms: float,
        include_diagnostic: bool = False,
    ) -> dict[str, Any]:
        raw = np.asarray(actions, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 7 or not np.isfinite(raw).all():
            raise RuntimeError(f"invalid RTC action chunk: {raw.shape}")
        physical = physical_chunk_from_report(prediction_report, len(raw))
        physical_before_blend = physical.copy()
        old_plan = self.active_plan
        prefix = self.action_tick - int(start_tick)
        remaining = len(physical) - prefix
        active_end_tick = (
            None
            if self.active_plan is None
            else int(self.active_plan.start_tick + len(self.active_plan.physical_actions))
        )
        defer_to_boundary = bool(
            self.finish_active_plan_before_install
            and self.active_plan is not None
            and active_end_tick is not None
            and self.action_tick < active_end_tick
        )
        activation_tick = (
            int(active_end_tick) if defer_to_boundary else int(self.action_tick)
        )
        activation_prefix = activation_tick - int(start_tick)
        activation_remaining = len(physical) - activation_prefix
        reason = None
        if activation_prefix < 0:
            reason = "result_start_tick_is_in_the_future"
        elif result_age_ms > max_result_age_ms:
            reason = "result_exceeded_wall_time_age_limit"
        elif activation_remaining < self.min_suffix_actions:
            reason = "result_has_too_few_time_aligned_suffix_actions"

        old_offset = None
        blend_count = 0
        before_rad = None
        after_rad = None
        blend_weights: list[float] = []
        if (
            reason is None
            and not defer_to_boundary
            and self.active_plan is not None
        ):
            old_offset = self.action_tick - self.active_plan.start_tick
            if (
                0 <= old_offset < len(self.active_plan.physical_actions)
                and 0 <= prefix < len(physical)
            ):
                old_reference = self.active_plan.physical_actions[old_offset, :6]
                before_rad = float(
                    np.sqrt(np.mean(np.square(physical[prefix, :6] - old_reference)))
                )
                blend_count = min(
                    self.blend_actions,
                    len(physical) - prefix,
                    len(self.active_plan.physical_actions) - old_offset,
                )
                for blend_index in range(blend_count):
                    alpha = self._take_new_weight(blend_index, blend_count)
                    blend_weights.append(alpha)
                    new_index = prefix + blend_index
                    old_index = old_offset + blend_index
                    physical[new_index, :6] = (
                        (1.0 - alpha)
                        * self.active_plan.physical_actions[old_index, :6]
                        + alpha * physical[new_index, :6]
                    )
                after_rad = float(
                    np.sqrt(np.mean(np.square(physical[prefix, :6] - old_reference)))
                )

        accepted = reason is None
        replaced_buffered_plan_id = None
        if accepted:
            if defer_to_boundary:
                if self.buffered_plan is not None:
                    replaced_buffered_plan_id = int(self.buffered_plan.plan_id)
                self.buffered_plan = BufferedPlan(
                    plan_id=int(plan_id),
                    start_tick=int(start_tick),
                    physical_actions=physical,
                    prediction_report=copy.deepcopy(prediction_report),
                    buffered_tick=int(self.action_tick),
                    result_age_ms=float(result_age_ms),
                )
            else:
                self.active_plan = ActivePlan(
                    plan_id=int(plan_id),
                    start_tick=int(start_tick),
                    physical_actions=physical,
                    prediction_report=copy.deepcopy(prediction_report),
                )
                self.buffered_plan = None
        report = {
            "accepted": accepted,
            "reason": reason,
            "plan_id": int(plan_id),
            "request_tick": int(start_tick),
            "install_tick": int(self.action_tick),
            "inference_delay_action_ticks": int(max(0, prefix)),
            "skipped_prefix_actions": int(max(0, prefix)),
            "remaining_suffix_actions": int(max(0, remaining)),
            "rtc_space": "decoded_physical_6d_joint_radians",
            "rtc_gripper_guidance": False,
            "old_plan_offset": old_offset,
            "blend_actions_requested": self.blend_actions,
            "blended_actions": blend_count,
            "cosine_take_new_weights": blend_weights,
            "switch_joint_rms_before_rad": before_rad,
            "switch_joint_rms_after_rad": after_rad,
            "switch_joint_rms_before_deg": (
                None if before_rad is None else math.degrees(before_rad)
            ),
            "switch_joint_rms_after_deg": (
                None if after_rad is None else math.degrees(after_rad)
            ),
            "result_age_ms": float(result_age_ms),
            "finish_active_plan_before_install": bool(
                self.finish_active_plan_before_install
            ),
            "deferred_install": bool(accepted and defer_to_boundary),
            "scheduled_activation_tick": int(activation_tick),
            "activation_prefix_actions": int(max(0, activation_prefix)),
            "remaining_suffix_at_activation": int(
                max(0, activation_remaining)
            ),
            "buffered_plan_id": (
                None if self.buffered_plan is None else int(self.buffered_plan.plan_id)
            ),
            "replaced_buffered_plan_id": replaced_buffered_plan_id,
        }
        if include_diagnostic:
            suffix_start = min(max(int(prefix), 0), len(physical))
            old_overlap = None
            old_plan_summary = None
            if old_plan is not None:
                old_plan_summary = {
                    "plan_id": int(old_plan.plan_id),
                    "start_tick": int(old_plan.start_tick),
                    "decoded_physical_actions_rad_binary": (
                        old_plan.physical_actions.copy()
                    ),
                }
                if old_offset is not None and blend_count > 0:
                    old_overlap = old_plan.physical_actions[
                        old_offset : old_offset + blend_count
                    ].copy()
            report["_plan_diagnostic"] = {
                "diagnostic_only_no_control_effect": True,
                "units": {
                    "policy_returned_actions": "driver joint degrees plus gripper percent",
                    "decoded_physical_actions": "joint radians plus binary gripper",
                    "ticks": "successful action command ticks",
                },
                "policy_returned_actions": raw.copy(),
                "decoded_physical_actions_before_rtc": physical_before_blend,
                "aligned_suffix_before_blend": physical_before_blend[
                    suffix_start:
                ].copy(),
                "old_active_plan": old_plan_summary,
                "old_aligned_overlap": old_overlap,
                "new_aligned_overlap_before_blend": physical_before_blend[
                    suffix_start : suffix_start + blend_count
                ].copy(),
                "new_aligned_overlap_after_blend": physical[
                    suffix_start : suffix_start + blend_count
                ].copy(),
                "installed_decoded_physical_actions_full": physical.copy(),
                "effective_suffix_from_install_tick": physical[
                    suffix_start:
                ].copy(),
                "prediction_report": copy.deepcopy(prediction_report),
            }
        return report

    def request_replan(self, reason: str) -> dict[str, Any]:
        """Request one new observation without interrupting the active plan."""
        reason = str(reason).strip()
        if not reason:
            raise ValueError("forced RTC replan reason must be non-empty")
        already_pending = self._forced_replan_reason is not None
        if not already_pending:
            self._forced_replan_reason = reason
        return {
            "requested": not already_pending,
            "already_pending": already_pending,
            "reason": self._forced_replan_reason,
            "action_tick": int(self.action_tick),
        }

    def pending_replan_reason(self) -> str | None:
        return self._forced_replan_reason

    def has_buffered_plan(self) -> bool:
        return self.buffered_plan is not None

    def consume_replan_request(self) -> str | None:
        reason = self._forced_replan_reason
        self._forced_replan_reason = None
        return reason

    def _activate_buffered_plan_if_ready(self) -> dict[str, Any] | None:
        if self.buffered_plan is None:
            return None
        if self.active_plan is not None:
            active_end_tick = int(
                self.active_plan.start_tick + len(self.active_plan.physical_actions)
            )
            if self.action_tick < active_end_tick:
                return None

        buffered = self.buffered_plan
        self.buffered_plan = None
        prefix = int(self.action_tick - buffered.start_tick)
        remaining = int(len(buffered.physical_actions) - prefix)
        if prefix < 0 or remaining < self.min_suffix_actions:
            event = {
                "activated": False,
                "reason": (
                    "buffered_plan_starts_in_the_future"
                    if prefix < 0
                    else "buffered_plan_has_too_few_suffix_actions"
                ),
                "plan_id": int(buffered.plan_id),
                "activation_tick": int(self.action_tick),
                "skipped_prefix_actions": int(max(0, prefix)),
                "remaining_suffix_actions": int(max(0, remaining)),
            }
        else:
            self.active_plan = ActivePlan(
                plan_id=int(buffered.plan_id),
                start_tick=int(buffered.start_tick),
                physical_actions=buffered.physical_actions,
                prediction_report=buffered.prediction_report,
            )
            event = {
                "activated": True,
                "reason": None,
                "plan_id": int(buffered.plan_id),
                "activation_tick": int(self.action_tick),
                "buffered_tick": int(buffered.buffered_tick),
                "buffer_wait_action_ticks": int(
                    self.action_tick - buffered.buffered_tick
                ),
                "skipped_prefix_actions": int(prefix),
                "remaining_suffix_actions": int(remaining),
                "rtc_overlap_blend_actions": 0,
                "continuity_authority": "existing_driver_step_and_acceleration_filter",
            }
        self._last_boundary_activation = event
        return event

    def remaining_actions(self) -> int:
        self._activate_buffered_plan_if_ready()
        if self.active_plan is None:
            return 0
        return max(
            0,
            len(self.active_plan.physical_actions)
            - (self.action_tick - self.active_plan.start_tick),
        )

    def transition_active_plan_to_unbiased(
        self,
        *,
        effective_tick: int,
        blend_actions: int,
    ) -> dict[str, Any]:
        """Remove Z bias from the remaining active plan with a cosine ramp."""
        if self.active_plan is None:
            return {
                "changed": False,
                "reason": "no_active_plan",
                "effective_tick": int(effective_tick),
            }
        original = np.asarray(
            self.active_plan.prediction_report.get(
                "selected_physical_actions_before_cartesian_z_bias"
            ),
            dtype=np.float32,
        )
        if (
            original.shape != self.active_plan.physical_actions.shape
            or not np.isfinite(original).all()
        ):
            raise RuntimeError(
                "active RTC plan cannot remove Cartesian Z bias because its "
                "unbiased decoded chunk is missing"
            )
        start_offset = max(
            0, int(effective_tick) - int(self.active_plan.start_tick)
        )
        if start_offset >= len(original):
            return {
                "changed": False,
                "reason": "effective_tick_after_active_plan",
                "effective_tick": int(effective_tick),
                "start_offset": int(start_offset),
            }
        old = self.active_plan.physical_actions.copy()
        updated = old.copy()
        blend_count = min(
            max(1, int(blend_actions)), len(updated) - start_offset
        )
        weights = []
        for blend_index in range(blend_count):
            alpha = self._take_new_weight(blend_index, blend_count)
            weights.append(alpha)
            index = start_offset + blend_index
            updated[index, :6] = (
                (1.0 - alpha) * old[index, :6]
                + alpha * original[index, :6]
            )
        updated[start_offset + blend_count :, :6] = original[
            start_offset + blend_count :, :6
        ]
        before_delta = old[start_offset:, :6] - original[start_offset:, :6]
        after_delta = updated[start_offset:, :6] - original[start_offset:, :6]
        self.active_plan.physical_actions = updated
        self.active_plan.prediction_report["selected_physical_actions"] = (
            updated.tolist()
        )
        self.active_plan.prediction_report[
            "post_grasp_z_bias_removal"
        ] = {
            "effective_tick": int(effective_tick),
            "start_offset": int(start_offset),
            "blend_actions": int(blend_count),
            "cosine_unbiased_weights": weights,
        }
        return {
            "changed": True,
            "reason": None,
            "plan_id": int(self.active_plan.plan_id),
            "effective_tick": int(effective_tick),
            "start_offset": int(start_offset),
            "blend_actions": int(blend_count),
            "max_abs_correction_before_deg": float(
                np.rad2deg(np.max(np.abs(before_delta)))
            ),
            "max_abs_correction_after_deg": float(
                np.rad2deg(np.max(np.abs(after_delta)))
            ),
        }

    def peek(self, driver_state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        self._activate_buffered_plan_if_ready()
        if self.active_plan is None:
            raise RuntimeError("RTC has no active action plan")
        requested_offset = self.action_tick - self.active_plan.start_tick
        if requested_offset < 0:
            raise RuntimeError(
                "RTC active plan starts in the future: "
                f"tick={self.action_tick}, plan={self.active_plan.plan_id}, "
                f"offset={requested_offset}"
            )
        # A delayed CUDA worker must never turn plan starvation into an abrupt
        # controller exit.  Hold the final decoded joint target and freeze the
        # successful-action clock until a replacement plan is installed.  A
        # stale returned result will be rejected by install(); the frozen tick
        # then lets the next fresh request start with a full suffix.
        plan_exhausted_hold = requested_offset >= len(
            self.active_plan.physical_actions
        )
        offset = min(requested_offset, len(self.active_plan.physical_actions) - 1)
        physical = self.active_plan.physical_actions[offset]
        raw_driver = np.empty(7, dtype=np.float32)
        raw_driver[:6] = self.policy._joint_rad_to_driver(physical[:6])
        raw_driver[6] = float(physical[6] >= 0.5) * 100.0
        safe_action, pending_velocity = self.policy._filter_driver_action(
            raw_driver,
            np.asarray(driver_state[:6], dtype=np.float64),
            self._last_velocity,
        )
        self._pending_velocity = np.asarray(pending_velocity, dtype=np.float64)
        status = self.status(offset=offset)
        status.update(
            {
                "requested_plan_offset": int(requested_offset),
                "plan_exhausted_hold": bool(plan_exhausted_hold),
            }
        )
        return safe_action, status

    def cancel_pending(self) -> None:
        self._pending_velocity = None

    def advance(self) -> None:
        if self._pending_velocity is None:
            raise RuntimeError("RTC cannot advance before peeking an action")
        self._last_velocity = self._pending_velocity.copy()
        self._pending_velocity = None
        self.action_tick += 1

    def status(self, *, offset: int | None = None) -> dict[str, Any]:
        if self.active_plan is None:
            return {
                "action_tick": int(self.action_tick),
                "active_plan_id": None,
                "active_plan_start_tick": None,
                "active_plan_offset": None,
                "remaining_actions": 0,
            }
        if offset is None:
            offset = self.action_tick - self.active_plan.start_tick
        return {
            "action_tick": int(self.action_tick),
            "active_plan_id": int(self.active_plan.plan_id),
            "active_plan_start_tick": int(self.active_plan.start_tick),
            "active_plan_offset": int(offset),
            "remaining_actions": int(self.remaining_actions()),
            "finish_active_plan_before_install": bool(
                self.finish_active_plan_before_install
            ),
            "buffered_plan_id": (
                None if self.buffered_plan is None else int(self.buffered_plan.plan_id)
            ),
            "buffered_plan_start_tick": (
                None if self.buffered_plan is None else int(self.buffered_plan.start_tick)
            ),
            "forced_replan_reason": self._forced_replan_reason,
            "last_boundary_activation": copy.deepcopy(
                self._last_boundary_activation
            ),
        }


class AbsoluteTimeGripper(legacy.IndependentGripperStateMachine):
    """Install full 30-step gripper forecasts at their observation tick."""

    def __init__(
        self,
        *,
        min_open_hold_frames: int,
        open_reached_confirm_frames: int,
        reclose_consensus_horizons: int,
        contact_detection: bool = False,
        contact_max_feedback: float = 70.0,
        contact_stall_frames: int = 15,
        contact_motion_tolerance: float = 0.5,
        contact_min_close_frames: int = 10,
        contact_min_drop: float = 10.0,
        contact_preload_percent: float = 2.0,
        initial_close_guard: bool = False,
        initial_close_consensus_horizons: int = 2,
        initial_close_deadline_latch: bool = False,
        initial_close_deadline_max_wait_frames: int = 10,
        first_close_pose_profile: Path | None = None,
        first_close_max_nearest_rms_deg: float = 3.0,
        first_close_envelope_margin_deg: float = 1.0,
        release_on_any_open_in_active_horizon: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.min_open_hold_frames = int(min_open_hold_frames)
        self.open_reached_confirm_frames = int(open_reached_confirm_frames)
        self.reclose_consensus_horizons = int(reclose_consensus_horizons)
        self.contact_detection = bool(contact_detection)
        self.contact_max_feedback = float(contact_max_feedback)
        self.contact_stall_frames = int(contact_stall_frames)
        self.contact_motion_tolerance = float(contact_motion_tolerance)
        self.contact_min_close_frames = int(contact_min_close_frames)
        self.contact_min_drop = float(contact_min_drop)
        self.contact_preload_percent = float(contact_preload_percent)
        self._contact_close_start_feedback: float | None = None
        self._contact_previous_feedback: float | None = None
        self._contact_stall_streak = 0
        self._contact_latched_target: float | None = None
        self.initial_close_guard = bool(initial_close_guard)
        self.initial_close_consensus_horizons = int(
            initial_close_consensus_horizons
        )
        self.initial_close_deadline_latch = bool(initial_close_deadline_latch)
        self.initial_close_deadline_max_wait_frames = int(
            initial_close_deadline_max_wait_frames
        )
        self._initial_close_forecasts: list[dict[str, Any]] = []
        self._initial_close_deadline_tick: int | None = None
        self._initial_close_deadline_status = (
            "collecting" if self.initial_close_deadline_latch else "disabled"
        )
        self._initial_close_deadline_confirming_chunks: list[int] = []
        self.first_close_max_nearest_rms_deg = float(
            first_close_max_nearest_rms_deg
        )
        self.first_close_envelope_margin_deg = float(
            first_close_envelope_margin_deg
        )
        self.release_on_any_open_in_active_horizon = bool(
            release_on_any_open_in_active_horizon
        )
        self._first_close_samples_deg: np.ndarray | None = None
        self._first_close_lower_deg: np.ndarray | None = None
        self._first_close_upper_deg: np.ndarray | None = None
        if self.initial_close_guard:
            if first_close_pose_profile is None:
                raise RuntimeError(
                    "initial close guard requires a first-grasp pose profile"
                )
            profile_path = Path(first_close_pose_profile)
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            if profile.get("schema") != "assembly_first_grasp_pose_profile_v1":
                raise RuntimeError(
                    f"unsupported first-grasp pose profile: {profile_path}"
                )
            samples = np.asarray(profile.get("samples_deg"), dtype=np.float64)
            percentiles = profile.get("percentiles_deg", {})
            lower = np.asarray(percentiles.get("p01"), dtype=np.float64)
            upper = np.asarray(percentiles.get("p99"), dtype=np.float64)
            if (
                samples.ndim != 2
                or samples.shape[1:] != (6,)
                or samples.shape[0] < 2
                or lower.shape != (6,)
                or upper.shape != (6,)
                or not np.isfinite(samples).all()
                or not np.isfinite(lower).all()
                or not np.isfinite(upper).all()
            ):
                raise RuntimeError(
                    f"invalid first-grasp pose profile arrays: {profile_path}"
                )
            self._first_close_samples_deg = samples
            self._first_close_lower_deg = lower
            self._first_close_upper_deg = upper
        self.release_epoch_tick: int | None = None
        self.open_reached_streak = 0
        self.open_hold_age_frames = 0
        self.reclose_armed = True
        self._purged_horizons_on_release = 0

    def _initial_close_deadline_report(self) -> dict[str, Any]:
        trailing_close_forecasts: list[dict[str, Any]] = []
        for forecast in reversed(self._initial_close_forecasts):
            if forecast["first_future_close_tick"] is None:
                break
            trailing_close_forecasts.append(forecast)
        trailing_close_forecasts.reverse()
        return {
            "initial_close_deadline_latch_enabled": bool(
                self.initial_close_deadline_latch
            ),
            "initial_close_deadline_status": self._initial_close_deadline_status,
            "initial_close_deadline_tick": self._initial_close_deadline_tick,
            "initial_close_deadline_max_wait_frames": int(
                self.initial_close_deadline_max_wait_frames
            ),
            "initial_close_deadline_candidate_streak": int(
                len(trailing_close_forecasts)
            ),
            "initial_close_deadline_confirming_chunks": list(
                self._initial_close_deadline_confirming_chunks
            ),
        }

    def _observe_initial_close_forecast(
        self,
        *,
        chunk_id: int,
        start_tick: int,
        commands: np.ndarray,
    ) -> dict[str, Any]:
        if (
            not self.initial_close_deadline_latch
            or self.has_closed_once
            or self.release_epoch_tick is not None
        ):
            return self._initial_close_deadline_report()

        close_offsets = np.flatnonzero(commands == 0.0)
        future_close_ticks = [
            int(start_tick + int(offset))
            for offset in close_offsets
            if int(start_tick + int(offset)) >= int(self.prediction_tick)
        ]
        self._initial_close_forecasts.append(
            {
                "chunk_id": int(chunk_id),
                "start_tick": int(start_tick),
                "first_future_close_tick": (
                    min(future_close_ticks) if future_close_ticks else None
                ),
                "last_future_close_tick": (
                    max(future_close_ticks) if future_close_ticks else None
                ),
            }
        )
        # Worker results can arrive out of order; observation time is the
        # temporal authority, not installation order.
        self._initial_close_forecasts.sort(
            key=lambda item: (item["start_tick"], item["chunk_id"])
        )
        self._initial_close_forecasts = self._initial_close_forecasts[-64:]

        trailing_close_forecasts: list[dict[str, Any]] = []
        for forecast in reversed(self._initial_close_forecasts):
            if forecast["first_future_close_tick"] is None:
                break
            trailing_close_forecasts.append(forecast)
        trailing_close_forecasts.reverse()
        if (
            len(trailing_close_forecasts)
            >= self.initial_close_consensus_horizons
        ):
            confirming = trailing_close_forecasts[
                -self.initial_close_consensus_horizons :
            ]
            candidate_tick = min(
                int(forecast["first_future_close_tick"])
                for forecast in confirming
            )
            # Once two successive forecasts agree that a close is coming,
            # later replans may bring it forward but may not postpone it.
            if (
                self._initial_close_deadline_tick is None
                or candidate_tick < self._initial_close_deadline_tick
            ):
                self._initial_close_deadline_tick = candidate_tick
                self._initial_close_deadline_confirming_chunks = [
                    int(forecast["chunk_id"]) for forecast in confirming
                ]
            self._initial_close_deadline_status = "locked"
        return self._initial_close_deadline_report()

    def _apply_initial_close_deadline(
        self, command: float
    ) -> tuple[float, dict[str, Any]]:
        report = self._initial_close_deadline_report()
        if (
            not self.initial_close_deadline_latch
            or self.has_closed_once
            or self.release_epoch_tick is not None
        ):
            return float(command), report
        deadline = self._initial_close_deadline_tick
        if deadline is None:
            # Do not let a single transient tail prediction close the gripper.
            return 100.0, report
        wait_age = int(self.prediction_tick - deadline)
        if wait_age < 0:
            return 100.0, report
        if wait_age <= self.initial_close_deadline_max_wait_frames:
            self._initial_close_deadline_status = "due"
            report = self._initial_close_deadline_report()
            report["initial_close_deadline_wait_age_frames"] = wait_age
            return 0.0, report

        # The arm never entered the training-derived pose corridor. Expire the
        # candidate instead of closing late in an unrelated pose.
        self._initial_close_deadline_status = "expired_pose_timeout"
        self._initial_close_deadline_tick = None
        self._initial_close_deadline_confirming_chunks = []
        self._initial_close_forecasts = []
        report = self._initial_close_deadline_report()
        report["initial_close_deadline_wait_age_frames"] = wait_age
        return 100.0, report

    def _purge_before_release_epoch(self) -> int:
        if self.release_epoch_tick is None:
            return 0
        before = len(self._prediction_horizons)
        self._prediction_horizons = [
            plan
            for plan in self._prediction_horizons
            if plan[1] >= self.release_epoch_tick
        ]
        return before - len(self._prediction_horizons)

    def observe_prediction_horizon_at(
        self, chunk_id: int, start_tick: int, model_commands_pct: Any
    ) -> dict[str, Any]:
        commands = self._validate_prediction_horizon(model_commands_pct)
        chunk = int(chunk_id)
        if any(
            existing_chunk == chunk
            for existing_chunk, _, _ in self._prediction_horizons
        ):
            raise RuntimeError(f"duplicate gripper prediction horizon for chunk {chunk}")
        if (
            self.release_epoch_tick is not None
            and int(start_tick) < self.release_epoch_tick
        ):
            return {
                "chunk_id": chunk,
                "accepted": False,
                "reason": "captured_before_release_epoch",
                "absolute_start_tick": int(start_tick),
                "release_epoch_tick": int(self.release_epoch_tick),
                "installed_at_tick": int(self.prediction_tick),
                "expired_prefix_actions": int(
                    max(0, self.prediction_tick - int(start_tick))
                ),
                "horizon_steps": int(commands.size),
                "commands_pct": commands.tolist(),
            }
        self._prediction_horizons.append(
            (chunk, int(start_tick), commands.copy())
        )
        deadline_report = self._observe_initial_close_forecast(
            chunk_id=chunk,
            start_tick=int(start_tick),
            commands=commands,
        )
        close_offsets = np.flatnonzero(commands == 0.0)
        report = {
            "chunk_id": chunk,
            "accepted": True,
            "reason": None,
            "absolute_start_tick": int(start_tick),
            "release_epoch_tick": self.release_epoch_tick,
            "installed_at_tick": int(self.prediction_tick),
            "expired_prefix_actions": int(max(0, self.prediction_tick - start_tick)),
            "horizon_steps": int(commands.size),
            "close_steps": int(close_offsets.size),
            "first_close_offset": int(close_offsets[0]) if close_offsets.size else None,
            "last_close_offset": int(close_offsets[-1]) if close_offsets.size else None,
            "commands_pct": commands.tolist(),
        }
        report.update(deadline_report)
        return report

    def _horizon_command(
        self,
        fallback_model_command_pct: float,
        *,
        active_plan_id: int | None = None,
        active_plan_offset: int | None = None,
    ) -> tuple[float, dict[str, Any]]:
        if self.release_epoch_tick is None:
            active_votes: list[tuple[int, int, int, float]] = []
            retained: list[tuple[int, int, np.ndarray]] = []
            for chunk_id, start_tick, commands in self._prediction_horizons:
                offset = int(self.prediction_tick - start_tick)
                if offset < len(commands):
                    retained.append((chunk_id, start_tick, commands))
                if 0 <= offset < len(commands):
                    active_votes.append(
                        (start_tick, chunk_id, offset, float(commands[offset]))
                    )
            self._prediction_horizons = retained
            active_votes.sort(key=lambda vote: (vote[0], vote[1]))

            # Keep the first grasp on the exact same absolute-time plan as the
            # arm.  Previously the gripper selected the oldest forecast that
            # still covered this tick while the arm used the newest accepted
            # RTC plan.  A stale chunk tail could therefore close the gripper
            # even though the arm's active plan still requested open.
            if active_plan_id is not None:
                matched = [
                    vote for vote in active_votes if vote[1] == int(active_plan_id)
                ]
                matched_vote = matched[-1] if matched else None
                any_open_offsets: list[int] = []
                release_on_any_open_triggered = False
                if matched_vote is not None and active_plan_offset is not None:
                    start_tick, chunk_id, forecast_offset, _ = matched_vote
                    horizon = next(
                        commands
                        for candidate_chunk, _, commands in retained
                        if candidate_chunk == chunk_id
                    )
                    requested_offset = int(active_plan_offset)
                    if 0 <= requested_offset < len(horizon):
                        any_open_offsets = [
                            int(requested_offset + relative_offset)
                            for relative_offset in np.flatnonzero(
                                horizon[requested_offset:] > 50.0
                            )
                        ]
                        release_on_any_open_triggered = bool(
                            self.release_on_any_open_in_active_horizon
                            and self.has_closed_once
                            and any_open_offsets
                        )
                        if release_on_any_open_triggered:
                            command = 100.0
                            source = (
                                "active_rtc_plan_any_open_in_remaining_horizon"
                            )
                        else:
                            command = float(horizon[requested_offset])
                            source = "active_rtc_plan_full_horizon"
                    else:
                        command = float(fallback_model_command_pct)
                        source = "active_rtc_plan_selected_action_fallback"
                else:
                    start_tick = None
                    chunk_id = int(active_plan_id)
                    forecast_offset = None
                    requested_offset = active_plan_offset
                    command = float(fallback_model_command_pct)
                    source = "active_rtc_plan_selected_action_fallback"
                command, deadline_report = self._apply_initial_close_deadline(
                    command
                )
                report = {
                    "prediction_source": source,
                    "prediction_tick": int(self.prediction_tick),
                    "active_prediction_count": int(len(active_votes)),
                    "selected_prediction_chunk_id": int(active_plan_id),
                    "selected_prediction_offset": (
                        None
                        if active_plan_offset is None
                        else int(active_plan_offset)
                    ),
                    "active_plan_horizon_match": bool(matched_vote is not None),
                    "active_plan_forecast_offset": (
                        None
                        if matched_vote is None
                        else int(matched_vote[2])
                    ),
                    "active_plan_offset_aligned": bool(
                        matched_vote is not None
                        and active_plan_offset is not None
                        and int(matched_vote[2]) == int(active_plan_offset)
                    ),
                    "release_on_any_open_in_active_horizon": bool(
                        self.release_on_any_open_in_active_horizon
                    ),
                    "release_on_any_open_triggered": bool(
                        release_on_any_open_triggered
                    ),
                    "remaining_horizon_open_offsets": any_open_offsets,
                    "horizon_votes": [
                        {
                            "absolute_start_tick": start_tick,
                            "chunk_id": chunk_id,
                            "offset": offset,
                            "command_pct": command_pct,
                        }
                        for start_tick, chunk_id, offset, command_pct in active_votes
                    ],
                    "initial_close_consensus_horizons": int(
                        self.initial_close_consensus_horizons
                    ),
                    "initial_close_consensus_vote_count": None,
                    "initial_close_consensus": None,
                }
                report.update(deadline_report)
                return command, report

            # There should always be an active arm plan during execution.  If
            # a caller cannot provide one, fail closed with respect to motion:
            # require agreement from the newest independent forecasts instead
            # of allowing one stale horizon to trigger a grasp.
            consensus_votes = active_votes[
                -self.initial_close_consensus_horizons :
            ]
            enough_fresh_votes = (
                len(consensus_votes) >= self.initial_close_consensus_horizons
            )
            close_consensus = enough_fresh_votes and all(
                abs(command_pct) <= 1e-6
                for _, _, _, command_pct in consensus_votes
            )
            command = 0.0 if close_consensus else 100.0
            command, deadline_report = self._apply_initial_close_deadline(command)
            report = {
                "prediction_source": "pre_grasp_latest_horizon_consensus",
                "prediction_tick": int(self.prediction_tick),
                "active_prediction_count": int(len(active_votes)),
                "selected_prediction_chunk_id": (
                    consensus_votes[-1][1] if consensus_votes else None
                ),
                "selected_prediction_offset": (
                    consensus_votes[-1][2] if consensus_votes else None
                ),
                "horizon_votes": [
                    {
                        "absolute_start_tick": start_tick,
                        "chunk_id": chunk_id,
                        "offset": offset,
                        "command_pct": command_pct,
                    }
                    for start_tick, chunk_id, offset, command_pct in active_votes
                ],
                "initial_close_consensus_horizons": int(
                    self.initial_close_consensus_horizons
                ),
                "initial_close_consensus_vote_count": int(
                    len(consensus_votes)
                ),
                "initial_close_consensus": bool(close_consensus),
            }
            report.update(deadline_report)
            return command, report

        self._purge_before_release_epoch()
        active_votes: list[tuple[int, int, int, float]] = []
        retained: list[tuple[int, int, np.ndarray]] = []
        for chunk_id, start_tick, commands in self._prediction_horizons:
            offset = int(self.prediction_tick - start_tick)
            if offset < len(commands):
                retained.append((chunk_id, start_tick, commands))
            if 0 <= offset < len(commands):
                active_votes.append(
                    (start_tick, chunk_id, offset, float(commands[offset]))
                )
        self._prediction_horizons = retained

        # Result arrival order is not a time ordering guarantee.  Rank votes
        # by the observation tick that produced each forecast so "latest"
        # means newest visual/proprio feedback, even if a worker result arrives
        # out of order.
        active_votes.sort(key=lambda vote: (vote[0], vote[1]))
        consensus_votes = active_votes[-self.reclose_consensus_horizons :]
        enough_fresh_votes = (
            len(consensus_votes) >= self.reclose_consensus_horizons
        )
        close_consensus = enough_fresh_votes and all(
            abs(command_pct) <= 1e-6
            for _, _, _, command_pct in consensus_votes
        )
        command = 0.0 if close_consensus else 100.0
        return command, {
            "prediction_source": "post_release_latest_horizon_consensus",
            "prediction_tick": int(self.prediction_tick),
            "active_prediction_count": int(len(active_votes)),
            "selected_prediction_chunk_id": (
                consensus_votes[-1][1] if consensus_votes else None
            ),
            "selected_prediction_offset": (
                consensus_votes[-1][2] if consensus_votes else None
            ),
            "horizon_votes": [
                {
                    "absolute_start_tick": start_tick,
                    "chunk_id": chunk_id,
                    "offset": offset,
                    "command_pct": command_pct,
                }
                for start_tick, chunk_id, offset, command_pct in active_votes
            ],
            "release_epoch_tick": int(self.release_epoch_tick),
            "reclose_consensus_horizons": int(
                self.reclose_consensus_horizons
            ),
            "reclose_consensus_vote_count": int(len(consensus_votes)),
            "reclose_close_consensus": bool(close_consensus),
        }

    def _first_close_pose_gate(
        self, current_model_joint_deg: Any
    ) -> dict[str, Any]:
        if (
            self._first_close_samples_deg is None
            or self._first_close_lower_deg is None
            or self._first_close_upper_deg is None
        ):
            raise RuntimeError("first-close pose gate was not initialized")
        current = np.asarray(current_model_joint_deg, dtype=np.float64).reshape(-1)
        if current.shape != (6,) or not np.isfinite(current).all():
            raise RuntimeError(
                "first-close pose gate requires six finite model-space joints"
            )
        sample_rms = np.sqrt(
            np.mean(
                np.square(self._first_close_samples_deg - current[None, :]),
                axis=1,
            )
        )
        nearest_index = int(np.argmin(sample_rms))
        nearest_rms = float(sample_rms[nearest_index])
        lower = (
            self._first_close_lower_deg - self.first_close_envelope_margin_deg
        )
        upper = (
            self._first_close_upper_deg + self.first_close_envelope_margin_deg
        )
        within_joint_envelope = np.logical_and(current >= lower, current <= upper)
        accepted = bool(
            nearest_rms <= self.first_close_max_nearest_rms_deg
            and np.all(within_joint_envelope)
        )
        return {
            "first_close_pose_gate_enabled": True,
            "first_close_pose_gate_accepted": accepted,
            "first_close_current_model_joint_deg": current.tolist(),
            "first_close_nearest_sample_index": nearest_index,
            "first_close_nearest_sample_rms_deg": nearest_rms,
            "first_close_max_nearest_rms_deg": float(
                self.first_close_max_nearest_rms_deg
            ),
            "first_close_within_joint_envelope": within_joint_envelope.tolist(),
            "first_close_envelope_lower_deg": lower.tolist(),
            "first_close_envelope_upper_deg": upper.tolist(),
        }

    def update_from_prediction_horizons(
        self,
        fallback_model_command_pct: float,
        feedback_pct: float,
        *,
        current_model_joint_deg: Any | None = None,
        active_plan_id: int | None = None,
        active_plan_offset: int | None = None,
    ) -> dict[str, Any]:
        model_command_pct, horizon_report = self._horizon_command(
            fallback_model_command_pct,
            active_plan_id=active_plan_id,
            active_plan_offset=active_plan_offset,
        )
        gate_report: dict[str, Any] = {
            "first_close_pose_gate_enabled": bool(self.initial_close_guard),
            "first_close_pose_gate_accepted": None,
        }
        if (
            self.initial_close_guard
            and not self.has_closed_once
            and abs(float(model_command_pct)) <= 1e-6
        ):
            if current_model_joint_deg is None:
                raise RuntimeError(
                    "initial close guard requires current model-space joints"
                )
            gate_report = self._first_close_pose_gate(current_model_joint_deg)
            if not gate_report["first_close_pose_gate_accepted"]:
                model_command_pct = 100.0
        report = self.update(model_command_pct, feedback_pct)
        if (
            self.initial_close_deadline_latch
            and report.get("transition") in {"hold->close", "open->close"}
        ):
            self._initial_close_deadline_status = "consumed"
            self._initial_close_deadline_tick = None
            self._initial_close_forecasts = []
            horizon_report.update(self._initial_close_deadline_report())
        report.update(horizon_report)
        report.update(gate_report)
        return report

    def update(self, model_command_pct: float, feedback_pct: float) -> dict[str, Any]:
        previous_state = self.state
        requested_before_guard = self._request_name(model_command_pct)
        opening_guard_active = bool(
            not self.terminal_open_after_release
            and self.state == "open"
            and not self.reclose_armed
        )
        guarded_command = 100.0 if opening_guard_active else model_command_pct
        report = super().update(guarded_command, feedback_pct)

        contact_transition: str | None = None
        contact_motion = None
        contact_drop = None
        if self.state != "close":
            self._contact_close_start_feedback = None
            self._contact_previous_feedback = None
            self._contact_stall_streak = 0
            self._contact_latched_target = None
        else:
            feedback = float(report["feedback_pct"])
            if previous_state != "close" or self._contact_close_start_feedback is None:
                self._contact_close_start_feedback = feedback
                self._contact_previous_feedback = feedback
                self._contact_stall_streak = 0
                self._contact_latched_target = None

            if self._contact_previous_feedback is not None:
                contact_motion = abs(feedback - self._contact_previous_feedback)
            contact_drop = max(
                0.0,
                float(self._contact_close_start_feedback) - feedback,
            )

            if self._contact_latched_target is None and self.contact_detection:
                contact_candidate = bool(
                    self.state_age_frames >= self.contact_min_close_frames
                    and feedback > self.close_feedback_threshold
                    and feedback <= self.contact_max_feedback
                    and contact_drop >= self.contact_min_drop
                    and contact_motion is not None
                    and contact_motion <= self.contact_motion_tolerance
                )
                if contact_candidate:
                    self._contact_stall_streak += 1
                else:
                    self._contact_stall_streak = 0
                if self._contact_stall_streak >= self.contact_stall_frames:
                    self._contact_latched_target = float(
                        np.clip(
                            feedback - self.contact_preload_percent,
                            0.0,
                            100.0,
                        )
                    )
                    contact_transition = "object_contact_latched"

            if self._contact_latched_target is not None:
                # A blocked gripper must not keep chasing the impossible 0%
                # endpoint.  Hold a small preload beyond the measured contact
                # aperture so the configured effort retains the object without
                # repeatedly winding up and releasing the position loop.
                report["command_pct"] = float(self._contact_latched_target)
                report["target_reached"] = True

            self._contact_previous_feedback = feedback

        if (
            report["transition"] == "close->open"
            and not self.terminal_open_after_release
        ):
            # The transition command is sent for prediction_tick; the next
            # observation belongs to the new post-release epoch.
            self.release_epoch_tick = int(self.prediction_tick + 1)
            self._purged_horizons_on_release = self._purge_before_release_epoch()
            self.open_reached_streak = 0
            self.open_hold_age_frames = 0
            self.reclose_armed = False
            self.close_request_streak = 0
            opening_guard_active = True
        elif self.state == "open" and not self.reclose_armed:
            if report["open_reached"]:
                self.open_reached_streak += 1
            else:
                self.open_reached_streak = 0
                self.open_hold_age_frames = 0
            if self.open_reached_streak >= self.open_reached_confirm_frames:
                self.open_hold_age_frames += 1
            if self.open_hold_age_frames >= self.min_open_hold_frames:
                self.reclose_armed = True
                self.close_request_streak = 0
                self.open_request_streak = 0
        elif self.state == "close":
            self.open_reached_streak = 0
            self.open_hold_age_frames = 0

        report.update(
            {
                "model_request_before_open_guard": requested_before_guard,
                "post_release_open_guard_active": bool(opening_guard_active),
                "release_epoch_tick": self.release_epoch_tick,
                "open_reached_streak": int(self.open_reached_streak),
                "open_reached_confirm_frames": int(
                    self.open_reached_confirm_frames
                ),
                "open_hold_age_frames": int(self.open_hold_age_frames),
                "min_open_hold_frames": int(self.min_open_hold_frames),
                "reclose_armed": bool(self.reclose_armed),
                "purged_horizons_on_release": int(
                    self._purged_horizons_on_release
                ),
                "contact_detection_enabled": bool(self.contact_detection),
                "object_contact_latched": bool(
                    self._contact_latched_target is not None
                ),
                "contact_transition": contact_transition,
                "contact_latched_target_pct": self._contact_latched_target,
                "contact_close_start_feedback_pct": (
                    self._contact_close_start_feedback
                ),
                "contact_feedback_motion_pct": contact_motion,
                "contact_feedback_drop_pct": contact_drop,
                "contact_stall_streak": int(self._contact_stall_streak),
                "contact_stall_frames": int(self.contact_stall_frames),
                "contact_motion_tolerance_pct": float(
                    self.contact_motion_tolerance
                ),
                "contact_min_close_frames": int(self.contact_min_close_frames),
                "contact_min_drop_pct": float(self.contact_min_drop),
                "contact_max_feedback_pct": float(self.contact_max_feedback),
                "contact_preload_percent": float(self.contact_preload_percent),
            }
        )
        return report


def run_live_async_rtc(
    args: argparse.Namespace,
    policy,
    log: legacy.JsonlLogger,
    device: torch.device,
    joint_limits: list[tuple[float, float]],
    gripper_stroke_mm: float,
    adapter,
    run_timestamp: str,
    diagnostic_image_view_builder: (
        Callable[[dict[str, Any]], dict[str, np.ndarray]] | None
    ) = None,
) -> None:
    execute = args.mode == "execute"
    if execute and not args.allow_motion:
        raise RuntimeError("execute mode requires --allow-motion")

    initial_pose = legacy.load_hardware_initial_pose(
        args.initial_pose_json,
        adapter,
    )
    firmware_initial_pose = legacy.apply_j2_command_offset_to_initial_pose(
        initial_pose,
        args.j2_command_offset_deg,
        list(adapter.hardware_limits_deg),
    )
    firmware_initial_pose = legacy.apply_j3_command_offset_to_initial_pose(
        firmware_initial_pose,
        args.j3_command_offset_deg,
        list(adapter.hardware_limits_deg),
    )
    firmware_initial_pose = legacy.apply_j5_command_offset_to_initial_pose(
        firmware_initial_pose,
        args.j5_command_offset_deg,
        list(adapter.hardware_limits_deg),
    )
    phase_diagnostic = (
        SecondPickPhaseDiagnostic(args.phase_diagnostic_profile)
        if args.phase_diagnostics
        else None
    )
    z_bias_transform = PhaseGatedCartesianZBiasTransform(
        CartesianZBiasTransform(
            bias_mm=args.end_effector_z_bias_mm,
            joint_limits_deg=joint_limits,
            max_joint_correction_deg=(
                args.end_effector_z_bias_max_joint_correction_deg
            ),
        ),
        disable_after_first_close=(
            args.end_effector_z_bias_disable_after_first_close
        ),
    )
    robot = legacy.make_robot(
        args,
        list(adapter.hardware_limits_deg),
        adapter.hardware_gripper_stroke_mm,
    )
    # Scope the higher torque to this additive entrypoint. The legacy sync
    # wrapper does not expose this option and retains PiperConfig's 1000-unit
    # default.
    robot.config.gripper_effort = int(args.gripper_effort)
    robot.connect(calibrate=False)
    motion_promoted = False
    predictor: BackgroundPredictor | None = None
    image_writer: AsyncDiagnosticImageWriter | None = None
    full_run_image_writer: AsyncFullRunImageWriter | None = None
    plan_writer: AsyncPlanDiagnosticWriter | None = None
    controller = DecodedJointRtcController(
        policy,
        args.rtc_min_suffix_actions,
        args.rtc_blend_actions,
        finish_active_plan_before_install=(
            args.rtc_finish_active_plan_before_install
        ),
    )
    gripper = AbsoluteTimeGripper(
        mode=args.gripper_control_mode,
        close_confirm_frames=args.gripper_close_confirm_frames,
        open_confirm_frames=args.gripper_open_confirm_frames,
        min_close_hold_frames=args.gripper_min_close_hold_frames,
        min_open_hold_frames=args.gripper_min_open_hold_frames,
        open_reached_confirm_frames=args.gripper_open_reached_confirm_frames,
        reclose_consensus_horizons=args.gripper_reclose_consensus_horizons,
        contact_detection=args.gripper_contact_detection,
        contact_max_feedback=args.gripper_contact_max_feedback,
        contact_stall_frames=args.gripper_contact_stall_frames,
        contact_motion_tolerance=args.gripper_contact_motion_tolerance,
        contact_min_close_frames=args.gripper_contact_min_close_frames,
        contact_min_drop=args.gripper_contact_min_drop,
        contact_preload_percent=args.gripper_contact_preload_percent,
        initial_close_guard=args.gripper_initial_close_guard,
        initial_close_consensus_horizons=(
            args.gripper_initial_close_consensus_horizons
        ),
        initial_close_deadline_latch=(
            args.gripper_initial_close_deadline_latch
        ),
        initial_close_deadline_max_wait_frames=(
            args.gripper_initial_close_deadline_max_wait_frames
        ),
        first_close_pose_profile=args.gripper_first_close_pose_profile,
        first_close_max_nearest_rms_deg=(
            args.gripper_first_close_max_nearest_rms_deg
        ),
        first_close_envelope_margin_deg=(
            args.gripper_first_close_envelope_margin_deg
        ),
        release_on_any_open_in_active_horizon=(
            args.gripper_release_on_any_open_in_active_horizon
        ),
        close_feedback_threshold=args.gripper_close_feedback_threshold,
        open_feedback_threshold=args.gripper_open_feedback_threshold,
        terminal_open_after_release=args.gripper_terminal_open_after_release,
    )
    delayed_release_assist = DelayedLowerBoundReleaseAssist(
        enabled=args.gripper_delayed_release_assist,
        profile_path=args.gripper_delayed_release_pose_profile,
        policy_fraction=args.gripper_delayed_release_policy_fraction,
        minimum_close_action_ticks=list(
            args.gripper_delayed_release_min_close_action_ticks
        ),
        max_nearest_pose_rms_deg=(
            args.gripper_delayed_release_max_nearest_pose_rms_deg
        ),
        max_joint_motion_rms_deg=(
            args.gripper_delayed_release_max_joint_motion_rms_deg
        ),
        stable_frames=args.gripper_delayed_release_stable_frames,
        healthy_frames=args.gripper_delayed_release_healthy_frames,
    )
    next_request_id = 1
    next_replan_tick = int(args.rtc_replan_interval)
    interrupted = False
    try:
        calibration_report = legacy.bind_runtime_calibration(
            robot,
            robot.sdk.get_status(),
            list(adapter.hardware_limits_deg),
            adapter.hardware_gripper_stroke_mm,
        )
        log.write(
            {
                "event": "runtime_calibration_bound",
                **calibration_report,
                "joint_space_adapter": adapter.report(),
                "cartesian_z_bias": z_bias_transform.settings_report(),
                "command_sent": False,
            }
        )
        camera_report = legacy.configure_front_camera_auto_controls(robot)
        log.write(
            {
                "event": "camera_config",
                "camera": "435if",
                **camera_report,
                "command_sent": False,
            }
        )

        preflight_observation = robot.get_observation()
        legacy.require_live_robot_feedback(
            robot,
            preflight_observation,
            context="deployment preflight",
        )
        preflight_home_start, preflight_home_report = legacy.validate_home_start_feedback(
            preflight_observation,
            args,
            list(adapter.hardware_limits_deg),
            adapter.hardware_gripper_stroke_mm,
        )
        legacy.require_deployment_pose_inside_limits(
            preflight_home_report,
            execute=execute,
            home_at_start=args.home_at_start,
        )
        preflight_joint_duration = legacy.quintic_joint_duration(
            preflight_home_start.joints_deg,
            firmware_initial_pose.joints_deg,
            args.home_velocity_deg_s,
            args.home_acceleration_deg_s2,
        )
        log.write(
            {
                "event": "live_preflight",
                "front_shape": list(np.asarray(preflight_observation["435if"]).shape),
                "wrist_shape": list(np.asarray(preflight_observation["405"]).shape),
                "side_shape": list(np.asarray(preflight_observation["side"]).shape),
                "raw_state": [
                    float(preflight_observation[f"joint_{index}.pos"])
                    for index in range(7)
                ],
                "initial_pose_json": str(args.initial_pose_json),
                "initial_pose_joints_deg": list(initial_pose.joints_deg),
                "firmware_initial_pose_joints_deg": list(
                    firmware_initial_pose.joints_deg
                ),
                "j2_command_offset_deg": args.j2_command_offset_deg,
                "j3_command_offset_deg": args.j3_command_offset_deg,
                "j5_command_offset_deg": args.j5_command_offset_deg,
                "initial_pose_gripper_mm": initial_pose.gripper_mm,
                "home_start_feedback_validation": preflight_home_report,
                "planned_joint_home_duration_s": preflight_joint_duration,
                "command_sent": False,
            }
        )
        if execute:
            held_position = legacy.adopt_already_enabled_motion_hold_current(robot)
            motion_promoted = True
            log.write(
                {
                    "event": "motion_enable",
                    "promotion_mode": "adopt_already_enabled_no_quick_stop",
                    "held_position": held_position,
                    "command_sent": True,
                }
            )
            if args.hold_current_only_seconds > 0:
                legacy.wait_for_stable_homing_health(robot)
                hold_started = time.perf_counter()
                while time.perf_counter() - hold_started < args.hold_current_only_seconds:
                    tick_started = time.perf_counter()
                    legacy.verify_homing_health(robot, check_controller=False)
                    legacy.latch_current_raw_position(robot)
                    legacy.precise_sleep(
                        CONTROL_PERIOD_S - (time.perf_counter() - tick_started)
                    )
                legacy.wait_for_stable_homing_health(robot)
                log.write(
                    {
                        "event": "hold_current_only_complete",
                        "duration_s": time.perf_counter() - hold_started,
                        "held_position": held_position,
                        "command_sent": True,
                    }
                )
                legacy.latch_current_pose_without_disabling_torque(
                    robot, log, reason="hold_current_only_complete"
                )
                return
            if args.home_at_start:
                legacy.run_initial_pose_home(
                    robot=robot,
                    goal=initial_pose,
                    args=args,
                    log=log,
                    joint_limits=list(adapter.hardware_limits_deg),
                    gripper_stroke_mm=adapter.hardware_gripper_stroke_mm,
                    phase="before_policy",
                )
            if args.home_only:
                log.write({"event": "home_only_complete", "command_sent": True})
                legacy.latch_current_pose_without_disabling_torque(
                    robot, log, reason="home_only_complete"
                )
                return

        warmup_observation = robot.get_observation()
        warmup_model_input_observation = legacy.hardware_observation_for_model(
            warmup_observation,
            args.j2_command_offset_deg,
            args.j3_command_offset_deg,
            args.j5_command_offset_deg,
        )
        warmup_started = time.perf_counter()
        warmup_actions = (
            policy.predict_action_chunk(
                legacy.live_observation_to_batch(
                    warmup_model_input_observation, device, adapter
                )
            )
            .detach()
            .cpu()
            .numpy()
            .copy()
        )
        warmup_report = z_bias_transform.apply_to_prediction_report(
            copy.deepcopy(policy.last_prediction_report)
        )
        install_report = controller.install(
            plan_id=0,
            start_tick=0,
            actions=warmup_actions,
            prediction_report=warmup_report,
            result_age_ms=0.0,
            max_result_age_ms=args.rtc_max_result_age_ms,
        )
        hardware_driver_state = legacy.observation_to_driver_state(
            warmup_observation,
            list(adapter.hardware_limits_deg),
            adapter.hardware_gripper_stroke_mm,
        )
        warmup_model_observation = adapter.hardware_observation_to_model_observation(
            warmup_model_input_observation
        )
        model_driver_state = legacy.observation_to_driver_state(
            warmup_model_observation,
            joint_limits,
            gripper_stroke_mm,
        )
        warmup_action, warmup_queue = controller.peek(model_driver_state)
        continuity = legacy.initial_continuity_report(
            warmup_model_observation,
            torch.from_numpy(warmup_action),
            joint_limits,
            args.max_initial_joint_delta_deg,
        )
        controller.cancel_pending()
        warmup_gripper = None
        if args.gripper_prediction_source == "full-horizon":
            warmup_gripper = gripper.observe_prediction_horizon_at(
                0,
                0,
                legacy.full_gripper_horizon_from_report(warmup_report),
            )
        log.write(
            {
                "event": "warmup",
                "mode": args.mode,
                "total_step_ms": (time.perf_counter() - warmup_started) * 1e3,
                "first_action": warmup_action.tolist(),
                "initial_continuity": continuity,
                "chunk_report": warmup_report,
                "rtc_install": install_report,
                "queue_status": warmup_queue,
                "gripper_horizon_update": warmup_gripper,
                "command_sent": False,
            }
        )
        predictor = BackgroundPredictor(
            policy,
            device,
            adapter,
            j2_command_offset_deg=args.j2_command_offset_deg,
            j3_command_offset_deg=args.j3_command_offset_deg,
            j5_command_offset_deg=args.j5_command_offset_deg,
            prediction_transform=z_bias_transform,
        )
        if args.async_diagnostic_images:
            image_writer = AsyncDiagnosticImageWriter(
                output_dir=args.log_dir / f"diagnostic_images_{run_timestamp}",
                view_builder=(
                    diagnostic_image_view_builder or raw_diagnostic_image_views
                ),
                queue_size=args.async_diagnostic_image_queue_size,
            )
        if args.record_full_run_images:
            full_run_image_writer = AsyncFullRunImageWriter(
                output_dir=args.log_dir / f"full_run_images_{run_timestamp}",
                queue_size=args.full_run_image_queue_size,
                jpeg_quality=args.full_run_image_jpeg_quality,
            )
        if args.rtc_plan_diagnostics:
            plan_writer = AsyncPlanDiagnosticWriter(
                output_dir=args.log_dir / f"plan_diagnostics_{run_timestamp}",
                view_builder=(
                    diagnostic_image_view_builder or raw_diagnostic_image_views
                ),
                queue_size=args.rtc_plan_diagnostic_queue_size,
            )

        started_run = time.perf_counter()
        next_diagnostic_image_tick = 0
        next_diagnostic_image_sample_id = 0
        delayed_release_report = copy.deepcopy(
            delayed_release_assist.last_report
        )
        for frame_index in range(args.max_steps):
            loop_started = time.perf_counter()
            observation = robot.get_observation()
            observation["_captured_ns"] = time.monotonic_ns()
            full_run_image_request = None
            if full_run_image_writer is not None:
                full_run_image_request = full_run_image_writer.submit(
                    FullRunImageRequest(
                        frame=frame_index,
                        action_tick=int(controller.action_tick),
                        captured_ns=int(observation["_captured_ns"]),
                        wall_time_utc=datetime.now(timezone.utc).isoformat(),
                        observation=copy_live_observation(observation),
                    )
                )
            hardware_driver_state = legacy.observation_to_driver_state(
                observation,
                list(adapter.hardware_limits_deg),
                adapter.hardware_gripper_stroke_mm,
            )
            model_input_observation = legacy.hardware_observation_for_model(
                observation,
                args.j2_command_offset_deg,
                args.j3_command_offset_deg,
                args.j5_command_offset_deg,
            )
            model_observation = adapter.hardware_observation_to_model_observation(
                model_input_observation
            )
            model_driver_state = legacy.observation_to_driver_state(
                model_observation,
                joint_limits,
                gripper_stroke_mm,
            )

            diagnostic_image_results = (
                image_writer.poll_all() if image_writer is not None else []
            )
            plan_diagnostic_results = (
                plan_writer.poll_all() if plan_writer is not None else []
            )
            diagnostic_image_request = None
            plan_diagnostic_request = None
            result_event = None
            gripper_horizon_update = None
            new_chunk_report = None
            result = predictor.poll()
            if result is not None:
                if result.error is not None:
                    raise RuntimeError(
                        f"asynchronous inference request {result.request_id} failed"
                    ) from result.error
                result_age_ms = (time.monotonic_ns() - result.captured_ns) / 1e6
                capture_plan_diagnostic = bool(
                    plan_writer is not None
                    and args.rtc_plan_diagnostic_start_id
                    <= result.request_id
                    <= args.rtc_plan_diagnostic_end_id
                )
                reconciled_report, z_bias_reconcile = (
                    z_bias_transform.reconcile_prediction_report(
                        result.prediction_report
                    )
                )
                result_event = controller.install(
                    plan_id=result.request_id,
                    start_tick=result.request_tick,
                    actions=result.actions,
                    prediction_report=reconciled_report,
                    result_age_ms=result_age_ms,
                    max_result_age_ms=args.rtc_max_result_age_ms,
                    include_diagnostic=capture_plan_diagnostic,
                )
                result_event["inference_timing"] = result.timing
                result_event["cartesian_z_bias_phase_gate_reconcile"] = (
                    z_bias_reconcile
                )
                plan_payload = result_event.pop("_plan_diagnostic", None)
                if plan_payload is not None:
                    if result.observation is None:
                        raise RuntimeError(
                            "plan diagnostics requested but inference snapshot is missing"
                        )
                    plan_payload["rtc_install"] = copy.deepcopy(result_event)
                    plan_payload["inference_timing"] = copy.deepcopy(result.timing)
                    plan_diagnostic_request = plan_writer.submit(
                        PlanDiagnosticRequest(
                            plan_id=result.request_id,
                            frame=result.request_frame,
                            request_tick=result.request_tick,
                            install_tick=controller.action_tick,
                            observation=result.observation,
                            payload=plan_payload,
                        )
                    )
                new_chunk_report = reconciled_report
                if result_event["accepted"]:
                    if args.gripper_prediction_source == "full-horizon":
                        gripper_horizon_update = gripper.observe_prediction_horizon_at(
                            result.request_id,
                            result.request_tick,
                            legacy.full_gripper_horizon_from_report(
                                reconciled_report
                            ),
                        )
                else:
                    next_replan_tick = controller.action_tick

            request_event = None
            forced_replan_reason = controller.pending_replan_reason()
            regular_replan_due = (
                controller.action_tick >= next_replan_tick
                or controller.remaining_actions() <= args.rtc_min_suffix_actions
            )
            should_request = (
                forced_replan_reason is not None
                or (
                    not controller.has_buffered_plan()
                    and regular_replan_due
                )
            )
            if should_request and predictor.can_submit():
                snapshot_started = time.perf_counter()
                snapshot = copy_live_observation(observation)
                snapshot_ms = (time.perf_counter() - snapshot_started) * 1e3
                request = InferenceRequest(
                    request_id=next_request_id,
                    request_tick=controller.action_tick,
                    request_frame=frame_index,
                    observation=snapshot,
                    submitted_ns=time.monotonic_ns(),
                    snapshot_ms=snapshot_ms,
                    gripper_policy_override_fraction=(
                        delayed_release_assist.current_override_fraction()
                    ),
                )
                if predictor.submit(request):
                    consumed_forced_replan_reason = (
                        controller.consume_replan_request()
                    )
                    request_event = {
                        "request_id": next_request_id,
                        "request_tick": int(controller.action_tick),
                        "active_plan_id": controller.status()["active_plan_id"],
                        "active_remaining_actions": int(controller.remaining_actions()),
                        "snapshot_ms": snapshot_ms,
                        "forced_replan_reason": consumed_forced_replan_reason,
                        "gripper_policy_override_fraction": (
                            request.gripper_policy_override_fraction
                        ),
                    }
                    next_request_id += 1
                    next_replan_tick = (
                        controller.action_tick + args.rtc_replan_interval
                    )
                    if (
                        image_writer is not None
                        and controller.action_tick >= next_diagnostic_image_tick
                    ):
                        diagnostic_image_request = image_writer.submit(
                            DiagnosticImageRequest(
                                sample_id=next_diagnostic_image_sample_id,
                                frame=frame_index,
                                action_tick=int(controller.action_tick),
                                request_id=request.request_id,
                                observation=snapshot,
                            )
                        )
                        if diagnostic_image_request["scheduled"]:
                            next_diagnostic_image_sample_id += 1
                            next_diagnostic_image_tick = (
                                controller.action_tick
                                + args.async_diagnostic_image_interval_frames
                            )

            model_action_array, queue_status = controller.peek(model_driver_state)
            model_action = legacy.action_to_robot_dict(
                torch.from_numpy(model_action_array)
            )
            if args.gripper_prediction_source == "full-horizon":
                gripper_state = gripper.update_from_prediction_horizons(
                    model_action["joint_6.pos"],
                    hardware_driver_state[6],
                    current_model_joint_deg=[
                        float(model_observation[f"joint_{index}.pos"]) / 1000.0
                        for index in range(6)
                    ],
                    active_plan_id=queue_status["active_plan_id"],
                    active_plan_offset=queue_status["active_plan_offset"],
                )
            else:
                gripper_state = gripper.update(
                    model_action["joint_6.pos"], hardware_driver_state[6]
                )
            contact_replan_request = None
            if (
                args.rtc_replan_after_contact
                and gripper_state.get("contact_transition")
                == "object_contact_latched"
            ):
                contact_replan_request = controller.request_replan(
                    "gripper_object_contact_latched"
                )
                next_replan_tick = min(
                    int(next_replan_tick), int(controller.action_tick + 1)
                )
            z_bias_phase_transition = None
            if (
                args.end_effector_z_bias_disable_after_first_close
                and gripper_state.get("transition")
                in {"hold->close", "open->close"}
                and z_bias_transform.active
            ):
                disable_report = z_bias_transform.disable_for_first_close(
                    action_tick=int(controller.action_tick)
                )
                active_plan_report = (
                    controller.transition_active_plan_to_unbiased(
                        effective_tick=int(controller.action_tick + 1),
                        blend_actions=(
                            args.end_effector_z_bias_disable_blend_actions
                        ),
                    )
                )
                next_replan_tick = min(
                    int(next_replan_tick), int(controller.action_tick + 1)
                )
                z_bias_phase_transition = {
                    "event": "first_close_disable_cartesian_z_bias",
                    "disable": disable_report,
                    "active_plan": active_plan_report,
                    "next_replan_tick": int(next_replan_tick),
                }
            frame_elapsed_s = time.perf_counter() - started_run
            phase_report = None
            if phase_diagnostic is not None:
                phase_report = phase_diagnostic.update(
                    frame=frame_index,
                    action_tick=int(controller.action_tick),
                    elapsed_s=frame_elapsed_s,
                    joints_deg=[
                        float(model_observation[f"joint_{index}.pos"]) / 1000.0
                        for index in range(6)
                    ],
                    gripper_transition=gripper_state.get("transition"),
                )
            unshifted_robot_action = (
                adapter.model_driver_action_to_hardware_action(model_action)
            )
            robot_action, j2_command_mapping = (
                legacy.apply_j2_command_offset_to_hardware_action(
                    unshifted_robot_action,
                    args.j2_command_offset_deg,
                    list(adapter.hardware_limits_deg),
                )
            )
            robot_action, j3_command_mapping = (
                legacy.apply_j3_command_offset_to_hardware_action(
                    robot_action,
                    args.j3_command_offset_deg,
                    list(adapter.hardware_limits_deg),
                )
            )
            robot_action, j5_command_mapping = (
                legacy.apply_j5_command_offset_to_hardware_action(
                    robot_action,
                    args.j5_command_offset_deg,
                    list(adapter.hardware_limits_deg),
                )
            )
            robot_action["joint_6.pos"] = gripper_state["command_pct"]
            if execute:
                legacy.verify_homing_health(
                    robot,
                    context="deployment control loop",
                )
                sent_action = robot.send_action(
                    robot_action,
                    present_position=legacy.normalized_present_position(
                        hardware_driver_state
                    ),
                )
            else:
                sent_action = None
            arm_clipped = None
            gripper_clipped = None
            clip_report = None
            if sent_action is not None:
                clip_report = command_clip_report(
                    robot_action,
                    sent_action,
                    arm_tolerance=args.arm_command_clip_tolerance,
                )
                arm_clipped = clip_report["arm_clipped"]
                gripper_clipped = clip_report["gripper_clipped"]
            # When the inference worker misses the current 30-action suffix,
            # keep commanding the last target but do not consume policy time.
            # This gives a fresh replan a stable absolute tick to attach to.
            plan_exhausted_hold = bool(
                queue_status.get("plan_exhausted_hold", False)
            )
            action_advanced = (
                (not execute or not arm_clipped)
                and not plan_exhausted_hold
            )
            if action_advanced:
                controller.advance()
                gripper.advance_prediction_tick()
            else:
                controller.cancel_pending()
            delayed_release_report = delayed_release_assist.observe(
                action_tick=int(controller.action_tick),
                current_model_joint_deg=[
                    float(model_observation[f"joint_{index}.pos"]) / 1000.0
                    for index in range(6)
                ],
                gripper_state=gripper_state,
                arm_clipped_or_stalled=bool(
                    arm_clipped or plan_exhausted_hold
                ),
            )
            delayed_release_replan_request = None
            if delayed_release_report.get("newly_activated"):
                delayed_release_replan_request = controller.request_replan(
                    "gripper_delayed_release_assist_activated"
                )
                next_replan_tick = min(
                    int(next_replan_tick), int(controller.action_tick + 1)
                )

            image_paths = None
            if request_event is not None and args.save_inference_images:
                image_paths = legacy.save_inference_images(
                    args.log_dir,
                    run_timestamp,
                    request_event["request_id"],
                    observation["435if"],
                    observation["405"],
                    observation["side"],
                )
            log.write(
                {
                    "mode": args.mode,
                    "frame": frame_index,
                    "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_s": frame_elapsed_s,
                    "total_step_ms": (time.perf_counter() - loop_started) * 1e3,
                    "raw_state": [
                        float(observation[f"joint_{index}.pos"])
                        for index in range(7)
                    ],
                    "model_input_state": [
                        float(model_input_observation[f"joint_{index}.pos"])
                        for index in range(7)
                    ],
                    "hardware_driver_state": hardware_driver_state.tolist(),
                    "model_driver_state": model_driver_state.tolist(),
                    "model_action": model_action,
                    "unshifted_hardware_action": unshifted_robot_action,
                    "j2_command_mapping": j2_command_mapping,
                    "j3_command_mapping": j3_command_mapping,
                    "j5_command_mapping": j5_command_mapping,
                    "proposed_action": robot_action,
                    "sent_action": sent_action,
                    "arm_command_was_clipped": arm_clipped,
                    "gripper_command_was_clipped": gripper_clipped,
                    "command_clip_report": clip_report,
                    "action_advanced": action_advanced,
                    "queue_status": controller.status(),
                    "gripper_state_machine": gripper_state,
                    "gripper_delayed_release_assist": delayed_release_report,
                    "gripper_delayed_release_replan_request": (
                        delayed_release_replan_request
                    ),
                    "gripper_contact_replan_request": contact_replan_request,
                    "cartesian_z_bias_phase_transition": (
                        z_bias_phase_transition
                    ),
                    "gripper_horizon_update": gripper_horizon_update,
                    "async_request": request_event,
                    "async_result": result_event,
                    "async_worker": predictor.status(),
                    "new_chunk_report": new_chunk_report,
                    "inference_images": image_paths,
                    "async_diagnostic_image_request": diagnostic_image_request,
                    "async_diagnostic_image_results": diagnostic_image_results,
                    "async_diagnostic_image_writer": (
                        image_writer.status() if image_writer is not None else None
                    ),
                    "full_run_image_request": full_run_image_request,
                    "full_run_image_writer": (
                        full_run_image_writer.status()
                        if full_run_image_writer is not None
                        else None
                    ),
                    "rtc_plan_diagnostic_request": plan_diagnostic_request,
                    "rtc_plan_diagnostic_results": plan_diagnostic_results,
                    "rtc_plan_diagnostic_writer": (
                        plan_writer.status() if plan_writer is not None else None
                    ),
                    "phase_diagnostic": phase_report,
                    "command_sent": bool(execute and sent_action is not None),
                }
            )
            legacy.precise_sleep(
                CONTROL_PERIOD_S - (time.perf_counter() - loop_started)
            )

        predictor.close()
        predictor = None
        if execute and args.home_at_end:
            legacy.run_initial_pose_home(
                robot=robot,
                goal=initial_pose,
                args=args,
                log=log,
                joint_limits=list(adapter.hardware_limits_deg),
                gripper_stroke_mm=adapter.hardware_gripper_stroke_mm,
                phase="after_policy",
            )
        if execute:
            legacy.latch_current_pose_without_disabling_torque(
                robot, log, reason="normal_completion"
            )
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        if execute and motion_promoted:
            try:
                legacy.latch_current_pose_without_disabling_torque(
                    robot, log, reason="exception_abort"
                )
            except BaseException as hold_error:
                logger.error("failed to latch current pose after abort: %s", hold_error)
        if interrupted:
            print(
                "\nCtrl+C received: current pose is latched. Finishing worker, "
                "camera, and CAN cleanup; additional Ctrl+C is ignored until "
                "cleanup completes.",
                file=sys.stderr,
                flush=True,
            )
        raise
    finally:
        previous_sigint_handler = None
        if interrupted and threading.current_thread() is threading.main_thread():
            previous_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if predictor is not None:
                try:
                    predictor.close()
                except BaseException as predictor_close_error:
                    # Never replace the control-loop exception with a secondary
                    # worker-shutdown timeout.  The worker is daemonized; the
                    # already-latched arm remains the safety authority here.
                    logger.error(
                        "failed to close asynchronous inference worker: %s",
                        predictor_close_error,
                    )
            if image_writer is not None:
                try:
                    final_image_results = image_writer.close()
                    log.write(
                        {
                            "event": "async_diagnostic_image_writer_closed",
                            "status": image_writer.status(),
                            "final_results": final_image_results,
                            "command_sent": False,
                        }
                    )
                except BaseException as image_error:
                    logger.error(
                        "failed to close asynchronous diagnostic image writer: %s",
                        image_error,
                    )
            if full_run_image_writer is not None:
                try:
                    final_full_run_status = full_run_image_writer.close()
                    log.write(
                        {
                            "event": "full_run_image_writer_closed",
                            "status": final_full_run_status,
                            "command_sent": False,
                        }
                    )
                except BaseException as full_run_image_error:
                    logger.error(
                        "failed to close full-run image writer: %s",
                        full_run_image_error,
                    )
            if plan_writer is not None:
                try:
                    final_plan_results = plan_writer.close()
                    log.write(
                        {
                            "event": "rtc_plan_diagnostic_writer_closed",
                            "status": plan_writer.status(),
                            "final_results": final_plan_results,
                            "command_sent": False,
                        }
                    )
                except BaseException as plan_error:
                    logger.error(
                        "failed to close asynchronous plan diagnostic writer: %s",
                        plan_error,
                    )
            robot.disconnect()
        finally:
            if previous_sigint_handler is not None:
                signal.signal(signal.SIGINT, previous_sigint_handler)


def parse_args() -> argparse.Namespace:
    rtc_parser = argparse.ArgumentParser(add_help=False)
    rtc_parser.add_argument("--rtc-replan-interval", type=int, default=10)
    rtc_parser.add_argument("--rtc-blend-actions", type=int, default=8)
    rtc_parser.add_argument("--rtc-min-suffix-actions", type=int, default=8)
    rtc_parser.add_argument("--rtc-max-result-age-ms", type=float, default=500.0)
    rtc_parser.add_argument(
        "--rtc-finish-active-plan-before-install",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "buffer returned arm chunks until the currently active 30-action "
            "chunk has completed; the buffered chunk remains anchored to its "
            "capture tick and its expired prefix is skipped at activation"
        ),
    )
    rtc_parser.add_argument(
        "--rtc-replan-after-contact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "request one fresh observation on confirmed gripper object contact; "
            "with finish-active mode, the result is buffered and does not "
            "interrupt the current arm chunk"
        ),
    )
    rtc_parser.add_argument(
        "--end-effector-z-bias-mm",
        type=float,
        default=0.0,
        help=(
            "signed base-frame gripper_base Z correction applied to decoded "
            "joint chunks before RTC; positive raises, negative lowers; "
            "start real-hardware testing at -5mm"
        ),
    )
    rtc_parser.add_argument(
        "--end-effector-z-bias-max-joint-correction-deg",
        type=float,
        default=5.0,
        help=(
            "per-joint absolute correction bound for local Cartesian Z bias"
        ),
    )
    rtc_parser.add_argument(
        "--end-effector-z-bias-disable-after-first-close",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "apply the configured Cartesian Z bias only until the first "
            "confirmed gripper-close transition, then permanently remove it"
        ),
    )
    rtc_parser.add_argument(
        "--end-effector-z-bias-disable-blend-actions",
        type=int,
        default=8,
        help=(
            "successful action ticks used to cosine-ramp the active plan from "
            "biased to unbiased joints after the first close"
        ),
    )
    rtc_parser.add_argument(
        "--arm-command-clip-tolerance",
        type=float,
        default=1e-4,
        help=(
            "minimum proposed-vs-sent difference in normalized PIPER driver "
            "position units that counts as a material arm clamp; smaller "
            "float round-trip noise still advances the action tick"
        ),
    )
    rtc_parser.add_argument("--gripper-min-open-hold-frames", type=int, default=60)
    rtc_parser.add_argument(
        "--gripper-open-reached-confirm-frames", type=int, default=3
    )
    rtc_parser.add_argument(
        "--gripper-reclose-consensus-horizons", type=int, default=2
    )
    rtc_parser.add_argument(
        "--gripper-release-on-any-open-in-active-horizon",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after the gripper has closed at least once, request release as "
            "soon as any still-valid action in the active RTC chunk predicts "
            "open; no contact or release-pose gate is applied"
        ),
    )
    rtc_parser.add_argument(
        "--gripper-delayed-release-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after contact, a training-derived minimum close duration, a "
            "release-pose match and stable healthy tracking, run a second "
            "inference with only gripper proprio set to the training lower "
            "bound; use only its gripper forecast and retain the normal 6-D arm path"
        ),
    )
    rtc_parser.add_argument(
        "--gripper-delayed-release-policy-fraction",
        type=float,
        default=0.58,
        help="training-space gripper aperture fraction used only by the auxiliary policy pass",
    )
    rtc_parser.add_argument(
        "--gripper-delayed-release-min-close-action-ticks",
        type=int,
        nargs=3,
        default=(262, 289, 227),
        metavar=("FIRST", "SECOND", "THIRD"),
        help=(
            "training-derived minimum successful action ticks from each close "
            "to its release; three values are required"
        ),
    )
    rtc_parser.add_argument(
        "--gripper-delayed-release-pose-profile",
        type=Path,
        default=DEFAULT_DELAYED_RELEASE_POSE_PROFILE,
    )
    rtc_parser.add_argument(
        "--gripper-delayed-release-max-nearest-pose-rms-deg",
        type=float,
        default=1.0,
    )
    rtc_parser.add_argument(
        "--gripper-delayed-release-max-joint-motion-rms-deg",
        type=float,
        default=0.2,
        help="largest frame-to-frame 6-D joint RMS eligible for release",
    )
    rtc_parser.add_argument(
        "--gripper-delayed-release-stable-frames",
        type=int,
        default=5,
    )
    rtc_parser.add_argument(
        "--gripper-delayed-release-healthy-frames",
        type=int,
        default=5,
        help="consecutive frames without arm clipping or plan-exhausted hold",
    )
    rtc_parser.add_argument(
        "--gripper-contact-detection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "detect a stable, partially open gripper after a close command as "
            "object contact and latch a small position preload instead of "
            "continuing to chase the unreachable 0 percent endpoint"
        ),
    )
    rtc_parser.add_argument(
        "--gripper-contact-max-feedback",
        type=float,
        default=70.0,
        help="largest feedback aperture percent eligible for object contact",
    )
    rtc_parser.add_argument(
        "--gripper-contact-stall-frames",
        type=int,
        default=15,
        help="consecutive stable feedback frames required to latch contact",
    )
    rtc_parser.add_argument(
        "--gripper-contact-motion-tolerance",
        type=float,
        default=0.5,
        help="maximum frame-to-frame feedback motion in percent while stalled",
    )
    rtc_parser.add_argument(
        "--gripper-contact-min-close-frames",
        type=int,
        default=10,
        help="minimum close-state age before contact may be detected",
    )
    rtc_parser.add_argument(
        "--gripper-contact-min-drop",
        type=float,
        default=10.0,
        help="minimum aperture decrease in percent since close began",
    )
    rtc_parser.add_argument(
        "--gripper-contact-preload-percent",
        type=float,
        default=2.0,
        help=(
            "position preload below measured contact aperture; effort remains "
            "the configured --gripper-effort (5000 means the existing 5 N setting)"
        ),
    )
    rtc_parser.add_argument(
        "--gripper-initial-close-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "require recent independent gripper forecasts and a training-derived "
            "model-space joint-pose corridor before the first close"
        ),
    )
    rtc_parser.add_argument(
        "--gripper-initial-close-consensus-horizons", type=int, default=2
    )
    rtc_parser.add_argument(
        "--gripper-initial-close-deadline-latch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after successive horizons predict an upcoming first close, lock "
            "the earliest absolute action tick so later replans cannot "
            "postpone the grasp indefinitely"
        ),
    )
    rtc_parser.add_argument(
        "--gripper-initial-close-deadline-max-wait-frames",
        type=int,
        default=10,
        help=(
            "maximum successful action ticks after the locked deadline to "
            "wait for the first-grasp pose gate before expiring the candidate"
        ),
    )
    rtc_parser.add_argument(
        "--gripper-first-close-pose-profile",
        type=Path,
        default=DEFAULT_FIRST_GRASP_POSE_PROFILE,
    )
    rtc_parser.add_argument(
        "--gripper-first-close-max-nearest-rms-deg", type=float, default=3.0
    )
    rtc_parser.add_argument(
        "--gripper-first-close-envelope-margin-deg", type=float, default=1.0
    )
    rtc_parser.add_argument(
        "--gripper-effort",
        type=int,
        default=5000,
        help="PIPER gripper torque in 0.001 N*m SDK units (0..5000)",
    )
    rtc_parser.add_argument(
        "--async-diagnostic-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "save low-rate inference snapshots in a bounded background writer; "
            "never blocks or changes the control command"
        ),
    )
    rtc_parser.add_argument(
        "--async-diagnostic-image-interval-frames", type=int, default=30
    )
    rtc_parser.add_argument(
        "--async-diagnostic-image-queue-size", type=int, default=2
    )
    rtc_parser.add_argument(
        "--record-full-run-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "record front/wrist/side JPEGs for every policy control frame in "
            "a bounded background writer"
        ),
    )
    rtc_parser.add_argument(
        "--full-run-image-queue-size",
        type=int,
        default=64,
        help="maximum copied camera triplets waiting for background JPEG output",
    )
    rtc_parser.add_argument(
        "--full-run-image-jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for full-run camera recording (1..95)",
    )
    rtc_parser.add_argument(
        "--rtc-plan-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "save selected complete raw/aligned/blended RTC plans and the exact "
            "front/wrist/side model inputs in a background writer; diagnostic only"
        ),
    )
    rtc_parser.add_argument(
        "--rtc-plan-diagnostic-start-id", type=int, default=5
    )
    rtc_parser.add_argument(
        "--rtc-plan-diagnostic-end-id", type=int, default=10
    )
    rtc_parser.add_argument(
        "--rtc-plan-diagnostic-queue-size", type=int, default=4
    )
    rtc_parser.add_argument(
        "--phase-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="log event-aligned second-pick pose errors without gating motion",
    )
    rtc_parser.add_argument(
        "--phase-diagnostic-profile",
        type=Path,
        default=DEFAULT_PHASE_DIAGNOSTIC_PROFILE,
    )
    rtc_args, remaining = rtc_parser.parse_known_args()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = legacy.parse_args()
    finally:
        sys.argv = old_argv
    for key, value in vars(rtc_args).items():
        setattr(args, key, value)
    if args.list_checkpoint_profiles:
        return args
    if args.temporal_ensemble:
        raise ValueError(
            "do not combine built-in --temporal-ensemble with decoded-joint RTC"
        )
    if not 1 <= args.rtc_replan_interval < args.query_frequency:
        raise ValueError(
            "--rtc-replan-interval must be in [1, query-frequency-1]"
        )
    if not 1 <= args.rtc_blend_actions <= args.query_frequency:
        raise ValueError("--rtc-blend-actions must be in [1, query-frequency]")
    if not 1 <= args.rtc_min_suffix_actions <= args.query_frequency:
        raise ValueError(
            "--rtc-min-suffix-actions must be in [1, query-frequency]"
        )
    if not math.isfinite(args.rtc_max_result_age_ms) or args.rtc_max_result_age_ms <= 0:
        raise ValueError("--rtc-max-result-age-ms must be finite and positive")
    if (
        not math.isfinite(args.end_effector_z_bias_mm)
        or abs(args.end_effector_z_bias_mm) > 20.0
    ):
        raise ValueError("--end-effector-z-bias-mm must be within [-20,20]")
    if (
        not math.isfinite(args.end_effector_z_bias_max_joint_correction_deg)
        or not 0.0 < args.end_effector_z_bias_max_joint_correction_deg <= 10.0
    ):
        raise ValueError(
            "--end-effector-z-bias-max-joint-correction-deg must be in (0,10]"
        )
    if not (
        1
        <= args.end_effector_z_bias_disable_blend_actions
        <= args.query_frequency
    ):
        raise ValueError(
            "--end-effector-z-bias-disable-blend-actions must be in "
            "[1, query-frequency]"
        )
    if (
        args.end_effector_z_bias_disable_after_first_close
        and abs(args.end_effector_z_bias_mm) <= 1e-12
    ):
        raise ValueError(
            "--end-effector-z-bias-disable-after-first-close requires a "
            "non-zero --end-effector-z-bias-mm"
        )
    if (
        not math.isfinite(args.arm_command_clip_tolerance)
        or args.arm_command_clip_tolerance <= 0
    ):
        raise ValueError("--arm-command-clip-tolerance must be finite and positive")
    if args.gripper_min_open_hold_frames < 0:
        raise ValueError("--gripper-min-open-hold-frames must be non-negative")
    if args.gripper_open_reached_confirm_frames < 1:
        raise ValueError(
            "--gripper-open-reached-confirm-frames must be positive"
        )
    if args.gripper_reclose_consensus_horizons < 1:
        raise ValueError(
            "--gripper-reclose-consensus-horizons must be positive"
        )
    if not 0.0 <= args.gripper_delayed_release_policy_fraction <= 1.0:
        raise ValueError(
            "--gripper-delayed-release-policy-fraction must be in [0,1]"
        )
    if len(args.gripper_delayed_release_min_close_action_ticks) != 3 or any(
        int(value) < 1
        for value in args.gripper_delayed_release_min_close_action_ticks
    ):
        raise ValueError(
            "--gripper-delayed-release-min-close-action-ticks requires three positive integers"
        )
    if (
        not math.isfinite(
            args.gripper_delayed_release_max_nearest_pose_rms_deg
        )
        or args.gripper_delayed_release_max_nearest_pose_rms_deg <= 0
    ):
        raise ValueError(
            "--gripper-delayed-release-max-nearest-pose-rms-deg must be finite and positive"
        )
    if (
        not math.isfinite(
            args.gripper_delayed_release_max_joint_motion_rms_deg
        )
        or args.gripper_delayed_release_max_joint_motion_rms_deg < 0
    ):
        raise ValueError(
            "--gripper-delayed-release-max-joint-motion-rms-deg must be finite and non-negative"
        )
    if args.gripper_delayed_release_stable_frames < 1:
        raise ValueError(
            "--gripper-delayed-release-stable-frames must be positive"
        )
    if args.gripper_delayed_release_healthy_frames < 1:
        raise ValueError(
            "--gripper-delayed-release-healthy-frames must be positive"
        )
    if args.gripper_delayed_release_assist:
        if not args.gripper_contact_detection:
            raise ValueError(
                "--gripper-delayed-release-assist requires --gripper-contact-detection"
            )
        if args.gripper_release_on_any_open_in_active_horizon:
            raise ValueError(
                "delayed release assist requires absolute-time gripper execution; "
                "disable --gripper-release-on-any-open-in-active-horizon"
            )
        if not args.gripper_delayed_release_pose_profile.is_file():
            raise FileNotFoundError(args.gripper_delayed_release_pose_profile)
    if not 0 <= args.gripper_contact_max_feedback <= 100:
        raise ValueError("--gripper-contact-max-feedback must be in [0,100]")
    if args.gripper_contact_stall_frames < 1:
        raise ValueError("--gripper-contact-stall-frames must be positive")
    if not math.isfinite(args.gripper_contact_motion_tolerance) or (
        args.gripper_contact_motion_tolerance < 0
    ):
        raise ValueError(
            "--gripper-contact-motion-tolerance must be finite and non-negative"
        )
    if args.gripper_contact_min_close_frames < 0:
        raise ValueError(
            "--gripper-contact-min-close-frames must be non-negative"
        )
    if not 0 <= args.gripper_contact_min_drop <= 100:
        raise ValueError("--gripper-contact-min-drop must be in [0,100]")
    if not 0 <= args.gripper_contact_preload_percent <= 100:
        raise ValueError(
            "--gripper-contact-preload-percent must be in [0,100]"
        )
    if (
        args.gripper_contact_detection
        and args.gripper_contact_max_feedback
        <= args.gripper_close_feedback_threshold
    ):
        raise ValueError(
            "--gripper-contact-max-feedback must exceed "
            "--gripper-close-feedback-threshold when contact detection is enabled"
        )
    if args.gripper_initial_close_consensus_horizons < 1:
        raise ValueError(
            "--gripper-initial-close-consensus-horizons must be positive"
        )
    if args.gripper_initial_close_deadline_max_wait_frames < 0:
        raise ValueError(
            "--gripper-initial-close-deadline-max-wait-frames must be "
            "non-negative"
        )
    if (
        args.gripper_initial_close_deadline_latch
        and not args.gripper_initial_close_guard
    ):
        raise ValueError(
            "--gripper-initial-close-deadline-latch requires "
            "--gripper-initial-close-guard"
        )
    if (
        args.gripper_initial_close_deadline_latch
        and args.gripper_initial_close_deadline_max_wait_frames + 1
        < args.gripper_close_confirm_frames
    ):
        raise ValueError(
            "the initial-close deadline wait window must contain at least "
            "--gripper-close-confirm-frames action ticks"
        )
    if (
        not math.isfinite(args.gripper_first_close_max_nearest_rms_deg)
        or args.gripper_first_close_max_nearest_rms_deg <= 0
    ):
        raise ValueError(
            "--gripper-first-close-max-nearest-rms-deg must be finite and positive"
        )
    if (
        not math.isfinite(args.gripper_first_close_envelope_margin_deg)
        or args.gripper_first_close_envelope_margin_deg < 0
    ):
        raise ValueError(
            "--gripper-first-close-envelope-margin-deg must be finite and non-negative"
        )
    if (
        args.gripper_initial_close_guard
        and not args.gripper_first_close_pose_profile.is_file()
    ):
        raise ValueError(
            "--gripper-first-close-pose-profile must exist when the guard is enabled"
        )
    if not 0 <= args.gripper_effort <= 5000:
        raise ValueError("--gripper-effort must be in [0,5000]")
    if args.async_diagnostic_image_interval_frames < 1:
        raise ValueError(
            "--async-diagnostic-image-interval-frames must be positive"
        )
    if args.async_diagnostic_image_queue_size < 1:
        raise ValueError("--async-diagnostic-image-queue-size must be positive")
    if args.full_run_image_queue_size < 1:
        raise ValueError("--full-run-image-queue-size must be positive")
    if not 1 <= args.full_run_image_jpeg_quality <= 95:
        raise ValueError("--full-run-image-jpeg-quality must be in [1,95]")
    if args.rtc_plan_diagnostic_start_id < 1:
        raise ValueError("--rtc-plan-diagnostic-start-id must be at least 1")
    if args.rtc_plan_diagnostic_end_id < args.rtc_plan_diagnostic_start_id:
        raise ValueError(
            "--rtc-plan-diagnostic-end-id must be >= start id"
        )
    if args.rtc_plan_diagnostic_queue_size < 1:
        raise ValueError("--rtc-plan-diagnostic-queue-size must be positive")
    if args.phase_diagnostics and not args.phase_diagnostic_profile.is_file():
        raise FileNotFoundError(args.phase_diagnostic_profile)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.list_checkpoint_profiles:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    actual_sha256 = legacy.sha256_file(args.checkpoint)
    if actual_sha256 != args.checkpoint_sha256:
        raise RuntimeError(
            "checkpoint SHA-256 mismatch: "
            f"expected {args.checkpoint_sha256}, got {actual_sha256}"
        )
    joint_limits, gripper_stroke_mm = legacy.load_calibration(args.calibration_json)
    adapter = legacy.JointSpaceAdapter.from_files(
        mapping_path=args.role_mapping_json,
        model_calibration_path=args.calibration_json,
        hardware_calibration_path=args.hardware_calibration_json,
        joint_zero_offset_path=args.joint_zero_offset_json,
        allow_experimental_joint_mapping=args.allow_experimental_joint_mapping,
        mapping_mode=args.joint_mapping_mode,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    policy = legacy.make_policy(args, joint_limits, gripper_stroke_mm)
    if args.mode == "checkpoint":
        logger.info(
            "Async RTC checkpoint loaded without constructing a robot: "
            "sha256=%s qf=%d rtc_interval=%d contract=%s",
            actual_sha256,
            args.query_frequency,
            args.rtc_replan_interval,
            policy.checkpoint_report,
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = legacy.JsonlLogger(
        args.log_dir / f"async-rtc_{args.mode}_{timestamp}.jsonl"
    )
    try:
        log.write(
            {
                "event": "start",
                "mode": args.mode,
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": actual_sha256,
                "checkpoint_contract": policy.checkpoint_report,
                "scheduler": "asynchronous_inference_absolute_tick_rtc",
                "rtc_space": "decoded_physical_6d_joint_radians",
                "query_frequency": args.query_frequency,
                "inference_steps": args.inference_steps,
                "rtc_replan_interval": args.rtc_replan_interval,
                "rtc_blend_actions": args.rtc_blend_actions,
                "rtc_min_suffix_actions": args.rtc_min_suffix_actions,
                "rtc_max_result_age_ms": args.rtc_max_result_age_ms,
                "rtc_finish_active_plan_before_install": (
                    args.rtc_finish_active_plan_before_install
                ),
                "rtc_replan_after_contact": args.rtc_replan_after_contact,
                "gripper_rtc_guidance": False,
                "gripper_effort_sdk_units": args.gripper_effort,
                "gripper_effort_nm": args.gripper_effort * 0.001,
                "gripper_contact_detection": args.gripper_contact_detection,
                "gripper_contact_max_feedback": (
                    args.gripper_contact_max_feedback
                ),
                "gripper_contact_stall_frames": (
                    args.gripper_contact_stall_frames
                ),
                "gripper_contact_motion_tolerance": (
                    args.gripper_contact_motion_tolerance
                ),
                "gripper_contact_min_close_frames": (
                    args.gripper_contact_min_close_frames
                ),
                "gripper_contact_min_drop": args.gripper_contact_min_drop,
                "gripper_contact_preload_percent": (
                    args.gripper_contact_preload_percent
                ),
                "gripper_release_on_any_open_in_active_horizon": (
                    args.gripper_release_on_any_open_in_active_horizon
                ),
                "gripper_delayed_release_assist": (
                    args.gripper_delayed_release_assist
                ),
                "gripper_delayed_release_policy_fraction": (
                    args.gripper_delayed_release_policy_fraction
                ),
                "gripper_delayed_release_min_close_action_ticks": list(
                    args.gripper_delayed_release_min_close_action_ticks
                ),
                "gripper_delayed_release_pose_profile": str(
                    args.gripper_delayed_release_pose_profile
                ),
                "gripper_delayed_release_max_nearest_pose_rms_deg": (
                    args.gripper_delayed_release_max_nearest_pose_rms_deg
                ),
                "gripper_delayed_release_max_joint_motion_rms_deg": (
                    args.gripper_delayed_release_max_joint_motion_rms_deg
                ),
                "gripper_delayed_release_stable_frames": (
                    args.gripper_delayed_release_stable_frames
                ),
                "gripper_delayed_release_healthy_frames": (
                    args.gripper_delayed_release_healthy_frames
                ),
                "gripper_initial_close_guard": (
                    args.gripper_initial_close_guard
                ),
                "gripper_initial_close_consensus_horizons": (
                    args.gripper_initial_close_consensus_horizons
                ),
                "gripper_initial_close_deadline_latch": (
                    args.gripper_initial_close_deadline_latch
                ),
                "gripper_initial_close_deadline_max_wait_frames": (
                    args.gripper_initial_close_deadline_max_wait_frames
                ),
                "async_diagnostic_images": args.async_diagnostic_images,
                "async_diagnostic_image_interval_frames": (
                    args.async_diagnostic_image_interval_frames
                ),
                "async_diagnostic_image_queue_size": (
                    args.async_diagnostic_image_queue_size
                ),
                "record_full_run_images": args.record_full_run_images,
                "full_run_image_queue_size": args.full_run_image_queue_size,
                "full_run_image_jpeg_quality": args.full_run_image_jpeg_quality,
                "rtc_plan_diagnostics": args.rtc_plan_diagnostics,
                "rtc_plan_diagnostic_start_id": (
                    args.rtc_plan_diagnostic_start_id
                ),
                "rtc_plan_diagnostic_end_id": args.rtc_plan_diagnostic_end_id,
                "rtc_plan_diagnostic_queue_size": (
                    args.rtc_plan_diagnostic_queue_size
                ),
                "phase_diagnostics": args.phase_diagnostics,
                "phase_diagnostic_profile": (
                    str(args.phase_diagnostic_profile)
                    if args.phase_diagnostics
                    else None
                ),
                "seed": args.seed,
                "j2_command_offset_deg": args.j2_command_offset_deg,
                "j2_mapping": {
                    "model_proprio": "firmware_feedback_minus_offset",
                    "firmware_action": "model_action_plus_offset",
                    "firmware_home": "recorded_home_plus_offset",
                },
                "j3_command_offset_deg": args.j3_command_offset_deg,
                "j3_mapping": {
                    "model_proprio": "firmware_feedback_minus_offset",
                    "firmware_action": "model_action_plus_offset",
                    "firmware_home": "recorded_home_plus_offset",
                },
                "j5_command_offset_deg": args.j5_command_offset_deg,
                "j5_mapping": {
                    "model_proprio": "firmware_feedback_minus_offset",
                    "firmware_action": "model_action_plus_offset",
                    "firmware_home": "recorded_home_plus_offset",
                },
                "home_at_start": args.home_at_start,
                "home_at_end": args.home_at_end,
                "model_calibration_json": str(args.calibration_json),
                "hardware_calibration_json": str(args.hardware_calibration_json),
                "role_mapping_json": str(args.role_mapping_json),
                "joint_zero_offset_json": str(args.joint_zero_offset_json),
                "allow_experimental_joint_mapping": (
                    args.allow_experimental_joint_mapping
                ),
                "joint_space_adapter": adapter.report(),
                "command_sent": False,
            }
        )
        if args.mode == "offline":
            legacy.run_offline(args, policy, log, device)
        else:
            run_live_async_rtc(
                args,
                policy,
                log,
                device,
                joint_limits,
                gripper_stroke_mm,
                adapter,
                timestamp,
            )
    finally:
        log.close()
        logger.info("wrote async RTC audit log to %s", log.path)


if __name__ == "__main__":
    main()
