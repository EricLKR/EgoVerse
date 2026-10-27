#!/usr/bin/env python3
"""Map model physical joint angles onto the replacement PIPER hardware.

Assembly New1 checkpoints predict the recorded follower's joint coordinates.
The stable default keeps those physical angles unchanged.  A separately
selectable compatibility mode reconstructs collection-time leader commands by
preserving each joint's calibrated percentile between the recorded follower
and leader ranges.  The latter is required when replaying follower-labelled
data on the original collection leader whose calibration endpoints differ.

Both mappings are bidirectional: commands use model -> hardware and proprio
uses the exact hardware -> model inverse.  An explicitly validated residual
zero offset, if one is active, is applied in hardware degrees after the base
mapping.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


JOINT_ZERO_OFFSET_SCHEMA = "assembly_new1_joint_zero_offset_v1"
JOINT_MAPPING_MODES = (
    "physical-identity",
    "recorded-percentile-affine",
)


def load_joint_zero_offset(
    path: Path | None,
    *,
    allow_experimental: bool = False,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Load an explicitly gated model-to-hardware joint zero offset.

    An inactive file is a documented identity mapping.  A non-zero mapping is
    accepted only after it has been marked ``approved_active`` *and* contains a
    passing multi-pose validation record.  A Home-only calibration candidate
    is therefore impossible to activate accidentally.
    """
    if path is None:
        return np.zeros(6, dtype=np.float64), "implicit_inactive", {
            "required": True,
            "passed": False,
            "pose_count": 0,
        }
    with path.open() as stream:
        payload = json.load(stream)
    if payload.get("schema") != JOINT_ZERO_OFFSET_SCHEMA:
        raise ValueError(
            f"unsupported joint zero-offset schema in {path}: {payload.get('schema')!r}"
        )
    status = str(payload.get("status"))
    offset = np.asarray(
        payload.get("hardware_minus_model_joint_offset_deg"), dtype=np.float64
    )
    if offset.shape != (6,) or not np.isfinite(offset).all():
        raise ValueError(f"invalid six-axis joint zero offset in {path}: {offset.tolist()}")
    if float(np.max(np.abs(offset))) > 10.0:
        raise ValueError(
            f"joint zero offset exceeds the audited 10 degree bound in {path}: {offset.tolist()}"
        )
    validation = payload.get("multi_pose_validation")
    if not isinstance(validation, dict):
        raise ValueError(f"joint zero offset lacks multi_pose_validation in {path}")
    if status == "inactive":
        if not np.allclose(offset, 0.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"inactive joint zero offset must be exactly zero in {path}")
        return offset, status, validation
    if status == "experimental_unvalidated":
        if not allow_experimental:
            raise ValueError(
                "experimental joint zero offset requires the explicit "
                f"deployment opt-in in {path}"
            )
        if bool(validation.get("passed")):
            raise ValueError(
                f"experimental joint zero offset must remain marked unvalidated in {path}"
            )
        return offset, status, validation
    if status != "approved_active":
        raise ValueError(
            f"joint zero offset is not deployable in {path}: status={status!r}"
        )
    if not bool(validation.get("passed")) or int(validation.get("pose_count", 0)) < 3:
        raise ValueError(
            "approved joint zero offset requires passing validation at three or "
            f"more poses in {path}: {validation}"
        )
    return offset, status, validation


def load_calibration_limits(path: Path) -> tuple[tuple[tuple[float, float], ...], float]:
    with path.open() as stream:
        payload = json.load(stream)
    limits = tuple(
        (
            float(payload[f"joint_{index}"]["range_min"]) / 1000.0,
            float(payload[f"joint_{index}"]["range_max"]) / 1000.0,
        )
        for index in range(6)
    )
    stroke = float(payload["gripper"]["range_max"]) / 1000.0
    if any(not math.isfinite(value) for pair in limits for value in pair):
        raise ValueError(f"non-finite calibration in {path}")
    if any(high <= low for low, high in limits):
        raise ValueError(f"invalid joint limits in {path}: {limits}")
    if not math.isfinite(stroke) or stroke <= 0.0:
        raise ValueError(f"invalid gripper stroke in {path}: {stroke}")
    return limits, stroke


def _as_limits(values: Any, name: str) -> tuple[tuple[float, float], ...]:
    limits = tuple(tuple(float(item) for item in pair) for pair in values)
    if len(limits) != 6 or any(len(pair) != 2 for pair in limits):
        raise ValueError(f"{name} must contain six [min,max] pairs")
    if any(high <= low for low, high in limits):
        raise ValueError(f"invalid {name}: {limits}")
    return limits


def _normalise(degrees: Sequence[float], limits: Sequence[tuple[float, float]]) -> np.ndarray:
    values = np.asarray(degrees, dtype=np.float64)
    low = np.asarray([pair[0] for pair in limits], dtype=np.float64)
    high = np.asarray([pair[1] for pair in limits], dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError(f"invalid six-axis pose: {values.tolist()}")
    return (values - low) / (high - low) * 2.0 - 1.0


def _denormalise(unit: Sequence[float], limits: Sequence[tuple[float, float]]) -> np.ndarray:
    values = np.asarray(unit, dtype=np.float64)
    low = np.asarray([pair[0] for pair in limits], dtype=np.float64)
    high = np.asarray([pair[1] for pair in limits], dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError(f"invalid six-axis normalized pose: {values.tolist()}")
    return low + (values + 1.0) * 0.5 * (high - low)


def _require_inside_limits(
    degrees: Sequence[float],
    limits: Sequence[tuple[float, float]],
    *,
    name: str,
    tolerance_deg: float = 0.05,
) -> np.ndarray:
    values = np.asarray(degrees, dtype=np.float64)
    low = np.asarray([pair[0] for pair in limits], dtype=np.float64)
    high = np.asarray([pair[1] for pair in limits], dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError(f"invalid {name}: {values.tolist()}")
    outside = np.flatnonzero(
        (values < low - tolerance_deg) | (values > high + tolerance_deg)
    )
    if len(outside):
        details = {
            int(index): {
                "degrees": float(values[index]),
                "allowed": [float(low[index]), float(high[index])],
            }
            for index in outside
        }
        raise RuntimeError(f"{name} exceeds original-leader physical limits: {details}")
    # Recorded integer feedback can exceed a calibration endpoint by one or a
    # few 0.001-degree counts.  Admit only this audited rounding band, then
    # clamp exactly to the hardware endpoint so driver percentages stay in
    # [-100,100].  Material limit violations still fail above.
    return np.clip(values, low, high)


@dataclass(frozen=True)
class JointSpaceAdapter:
    """Bidirectional model-follower <-> original-leader conversion."""

    model_limits_deg: tuple[tuple[float, float], ...]
    hardware_limits_deg: tuple[tuple[float, float], ...]
    mirror_sign: tuple[float, ...]
    model_gripper_stroke_mm: float
    hardware_gripper_stroke_mm: float
    model_home_deg: tuple[float, ...]
    hardware_home_deg: tuple[float, ...]
    model_home_gripper_mm: float
    hardware_home_gripper_mm: float
    hardware_minus_model_zero_offset_deg: tuple[float, ...]
    joint_zero_offset_status: str
    joint_zero_offset_validation: dict[str, Any]
    joint_zero_offset_path: Path | None
    mapping_mode: str
    mapping_path: Path
    model_calibration_path: Path
    hardware_calibration_path: Path

    @classmethod
    def from_files(
        cls,
        *,
        mapping_path: Path,
        model_calibration_path: Path,
        hardware_calibration_path: Path,
        joint_zero_offset_path: Path | None = None,
        allow_experimental_joint_mapping: bool = False,
        mapping_mode: str = "physical-identity",
    ) -> "JointSpaceAdapter":
        if mapping_mode not in JOINT_MAPPING_MODES:
            raise ValueError(
                f"unsupported joint mapping mode {mapping_mode!r}; "
                f"expected one of {JOINT_MAPPING_MODES}"
            )
        with mapping_path.open() as stream:
            payload = json.load(stream)
        initial = payload["initial_pose"]
        recorded_model_limits = _as_limits(
            initial["follower_joint_limits_deg"], "follower_joint_limits_deg"
        )
        recorded_hardware_limits = _as_limits(
            initial["leader_joint_limits_deg"], "leader_joint_limits_deg"
        )
        model_limits, model_stroke = load_calibration_limits(model_calibration_path)
        hardware_limits, hardware_stroke = load_calibration_limits(
            hardware_calibration_path
        )
        if not np.allclose(model_limits, recorded_model_limits, rtol=0.0, atol=1e-9):
            raise ValueError(
                "model calibration differs from the recorded follower limits: "
                f"calibration={model_limits}, recording={recorded_model_limits}"
            )
        if not np.allclose(
            hardware_limits, recorded_hardware_limits, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                "hardware calibration differs from the recorded leader limits: "
                f"calibration={hardware_limits}, recording={recorded_hardware_limits}"
            )
        mirror = tuple(float(value) for value in initial["mirror_sign"])
        if len(mirror) != 6 or any(value not in (-1.0, 1.0) for value in mirror):
            raise ValueError(f"mirror_sign must contain six +/-1 values: {mirror}")

        if any(value != 1.0 for value in mirror):
            raise ValueError(
                "physical-angle deployment currently requires six +1 mirror signs; "
                f"recording has {mirror}"
            )

        zero_offset, zero_status, zero_validation = load_joint_zero_offset(
            joint_zero_offset_path,
            allow_experimental=allow_experimental_joint_mapping,
        )

        adapter = cls(
            model_limits_deg=model_limits,
            hardware_limits_deg=hardware_limits,
            mirror_sign=mirror,
            model_gripper_stroke_mm=model_stroke,
            hardware_gripper_stroke_mm=hardware_stroke,
            model_home_deg=tuple(
                float(value) for value in initial["follower_joint_positions_deg"]
            ),
            hardware_home_deg=tuple(
                float(value) for value in initial["leader_joint_positions_deg"]
            ),
            model_home_gripper_mm=float(initial["follower_gripper_opening_mm"]),
            hardware_home_gripper_mm=float(initial["leader_gripper_opening_mm"]),
            hardware_minus_model_zero_offset_deg=tuple(float(value) for value in zero_offset),
            joint_zero_offset_status=zero_status,
            joint_zero_offset_validation=zero_validation,
            joint_zero_offset_path=joint_zero_offset_path,
            mapping_mode=mapping_mode,
            mapping_path=mapping_path,
            model_calibration_path=model_calibration_path,
            hardware_calibration_path=hardware_calibration_path,
        )
        adapter._verify_camera_home()
        return adapter

    def _recorded_percentile_model_to_hardware(
        self, degrees: Sequence[float]
    ) -> np.ndarray:
        model_unit = _normalise(degrees, self.model_limits_deg)
        hardware_unit = model_unit * np.asarray(self.mirror_sign, dtype=np.float64)
        return _denormalise(hardware_unit, self.hardware_limits_deg)

    def _base_model_to_hardware(self, degrees: Sequence[float]) -> np.ndarray:
        values = np.asarray(degrees, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"invalid model six-axis pose: {values.tolist()}")
        if self.mapping_mode == "physical-identity":
            return values.copy()
        if self.mapping_mode == "recorded-percentile-affine":
            return self._recorded_percentile_model_to_hardware(values)
        raise AssertionError(f"unhandled joint mapping mode: {self.mapping_mode}")

    def _base_hardware_to_model(self, degrees: Sequence[float]) -> np.ndarray:
        values = np.asarray(degrees, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"invalid hardware six-axis pose: {values.tolist()}")
        if self.mapping_mode == "physical-identity":
            return values.copy()
        if self.mapping_mode == "recorded-percentile-affine":
            hardware_unit = _normalise(values, self.hardware_limits_deg)
            model_unit = hardware_unit * np.asarray(self.mirror_sign, dtype=np.float64)
            return _denormalise(model_unit, self.model_limits_deg)
        raise AssertionError(f"unhandled joint mapping mode: {self.mapping_mode}")

    def _verify_camera_home(self) -> None:
        _require_inside_limits(
            self._base_model_to_hardware(self.model_home_deg)
            + np.asarray(self.hardware_minus_model_zero_offset_deg, dtype=np.float64),
            self.hardware_limits_deg,
            name=f"{self.mapping_mode} hardware camera Home",
        )

    def model_degrees_to_hardware_degrees(
        self, degrees: Sequence[float]
    ) -> np.ndarray:
        hardware = self._base_model_to_hardware(degrees) + np.asarray(
            self.hardware_minus_model_zero_offset_deg, dtype=np.float64
        )
        return _require_inside_limits(
            hardware,
            self.hardware_limits_deg,
            name="model physical joint target",
        ).copy()

    def hardware_degrees_to_model_degrees(
        self, degrees: Sequence[float]
    ) -> np.ndarray:
        values = np.asarray(degrees, dtype=np.float64) - np.asarray(
            self.hardware_minus_model_zero_offset_deg, dtype=np.float64
        )
        # Present proprio in the recorded follower/model coordinate system.
        # This is the exact inverse of the selected command mapping.
        return self._base_hardware_to_model(values)

    def hardware_observation_to_model_observation(
        self, observation: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a shallow copy whose seven state values match training space."""
        hardware_deg = [
            float(observation[f"joint_{index}.pos"]) / 1000.0
            for index in range(6)
        ]
        model_deg = self.hardware_degrees_to_model_degrees(hardware_deg)
        converted = dict(observation)
        for index, value in enumerate(model_deg):
            converted[f"joint_{index}.pos"] = float(value * 1000.0)
        hardware_gripper_mm = float(observation["joint_6.pos"]) / 1000.0
        gripper_pct = np.clip(
            hardware_gripper_mm / self.hardware_gripper_stroke_mm, 0.0, 1.0
        )
        converted["joint_6.pos"] = float(
            gripper_pct * self.model_gripper_stroke_mm * 1000.0
        )
        return converted

    def model_driver_action_to_hardware_action(
        self, action: dict[str, float]
    ) -> dict[str, float]:
        converted = dict(action)
        model_driver = np.asarray(
            [action[f"joint_{index}.pos"] for index in range(6)], dtype=np.float64
        )
        if not np.isfinite(model_driver).all() or np.any(np.abs(model_driver) > 100.0001):
            raise RuntimeError(
                f"invalid model driver command before joint-space adaptation: {model_driver.tolist()}"
            )
        model_degrees = _denormalise(model_driver / 100.0, self.model_limits_deg)
        hardware_degrees = self.model_degrees_to_hardware_degrees(model_degrees)
        hardware_driver = _normalise(
            hardware_degrees, self.hardware_limits_deg
        ) * 100.0
        if np.any(np.abs(hardware_driver) > 100.0001):
            raise RuntimeError(
                "physical-angle target cannot be represented by original-leader "
                f"driver calibration: {hardware_driver.tolist()}"
            )
        for index, value in enumerate(hardware_driver):
            converted[f"joint_{index}.pos"] = float(value)
        # Both grippers use 0..100 aperture percentage; the SDK applies the
        # hardware stroke after this adapter.
        converted["joint_6.pos"] = float(action["joint_6.pos"])
        return converted

    def report(self) -> dict[str, Any]:
        physical_home = self.model_degrees_to_hardware_degrees(self.model_home_deg)
        percentile_home = self._recorded_percentile_model_to_hardware(
            self.model_home_deg
        )
        return {
            "mapping": self.mapping_mode,
            "mapping_path": str(self.mapping_path),
            "model_calibration_path": str(self.model_calibration_path),
            "hardware_calibration_path": str(self.hardware_calibration_path),
            "joint_zero_offset_path": (
                str(self.joint_zero_offset_path)
                if self.joint_zero_offset_path is not None
                else None
            ),
            "joint_zero_offset_status": self.joint_zero_offset_status,
            "hardware_minus_model_zero_offset_deg": list(
                self.hardware_minus_model_zero_offset_deg
            ),
            "joint_zero_offset_multi_pose_validation": self.joint_zero_offset_validation,
            "model_joint_limits_deg": [list(pair) for pair in self.model_limits_deg],
            "hardware_joint_limits_deg": [
                list(pair) for pair in self.hardware_limits_deg
            ],
            "mirror_sign": list(self.mirror_sign),
            "model_gripper_stroke_mm": self.model_gripper_stroke_mm,
            "hardware_gripper_stroke_mm": self.hardware_gripper_stroke_mm,
            "model_home_deg": list(self.model_home_deg),
            "recorded_teleoperation_leader_home_deg": list(self.hardware_home_deg),
            "recorded_percentile_mapped_home_deg": percentile_home.tolist(),
            "physical_camera_home_deg": physical_home.tolist(),
            "recorded_percentile_home_pair_max_error_deg": float(
                np.max(
                    np.abs(
                        percentile_home
                        - np.asarray(self.hardware_home_deg, dtype=np.float64)
                    )
                )
            ),
            "physical_vs_recorded_leader_home_delta_deg": (
                physical_home
                - np.asarray(self.hardware_home_deg, dtype=np.float64)
            ).tolist(),
        }
