#!/usr/bin/env python3
"""Audited local Cartesian-Z correction for decoded PIPER joint chunks.

The learned policy emits absolute physical joint angles.  A signed Z bias is
therefore applied *after* decoding and *before* RTC overlap blending.  The
correction uses a batched damped-least-squares Jacobian solve seeded at every
predicted joint target; it never performs random restarts or changes IK branch.

Positive ``bias_mm`` raises ``gripper_base`` in the arm base frame and negative
values lower it.  Zero is an exact no-op and does not import the optional
kinematics dependency.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


class CartesianZBiasError(RuntimeError):
    """Raised when a requested bias cannot be represented safely."""


def _rotation_vector(rotation: torch.Tensor) -> torch.Tensor:
    """Return batched SO(3) logarithms in the same frame as ``rotation``."""
    vector = 0.5 * torch.stack(
        (
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ),
        dim=-1,
    )
    sine = torch.linalg.vector_norm(vector, dim=-1)
    cosine = (
        (rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    ).clamp(-1.0, 1.0)
    angle = torch.atan2(sine, cosine)
    scale = torch.where(sine > 1e-9, angle / sine, torch.ones_like(sine))
    return vector * scale[:, None]


class CartesianZBiasTransform:
    """Translate decoded joint targets along base-frame Z with local IK."""

    def __init__(
        self,
        *,
        bias_mm: float,
        joint_limits_deg: Sequence[tuple[float, float]],
        max_joint_correction_deg: float = 5.0,
        position_tolerance_mm: float = 1.0,
        rotation_tolerance_deg: float = 1.0,
        damping: float = 2e-3,
        iterations: int = 12,
        adaptive_search_iterations: int = 7,
        chain: Any | None = None,
    ) -> None:
        self.bias_mm = float(bias_mm)
        self.bias_m = self.bias_mm * 1e-3
        self.max_joint_correction_deg = float(max_joint_correction_deg)
        self.position_tolerance_mm = float(position_tolerance_mm)
        self.rotation_tolerance_deg = float(rotation_tolerance_deg)
        self.damping = float(damping)
        self.iterations = int(iterations)
        self.adaptive_search_iterations = int(adaptive_search_iterations)
        limits = np.asarray(joint_limits_deg, dtype=np.float64)
        if limits.shape != (6, 2) or not np.isfinite(limits).all():
            raise ValueError("joint_limits_deg must be six finite [low,high] pairs")
        if np.any(limits[:, 1] <= limits[:, 0]):
            raise ValueError("joint_limits_deg contains a non-increasing pair")
        if not math.isfinite(self.bias_mm) or abs(self.bias_mm) > 20.0:
            raise ValueError("--end-effector-z-bias-mm must be within [-20,20]")
        if (
            not math.isfinite(self.max_joint_correction_deg)
            or self.max_joint_correction_deg <= 0.0
            or self.max_joint_correction_deg > 10.0
        ):
            raise ValueError(
                "--end-effector-z-bias-max-joint-correction-deg must be in (0,10]"
            )
        if self.position_tolerance_mm <= 0.0 or self.rotation_tolerance_deg <= 0.0:
            raise ValueError("Cartesian Z-bias tolerances must be positive")
        if (
            self.damping <= 0.0
            or self.iterations < 1
            or self.adaptive_search_iterations < 1
        ):
            raise ValueError("Cartesian Z-bias solver settings are invalid")
        self._low = torch.as_tensor(
            np.deg2rad(limits[:, 0]), dtype=torch.float64
        )
        self._high = torch.as_tensor(
            np.deg2rad(limits[:, 1]), dtype=torch.float64
        )
        self._chain = chain
        if self.enabled and self._chain is None:
            try:
                import pytorch_kinematics as pk
                from egomimic.robot.piper.piper_kinematics import DEFAULT_URDF_PATH
            except ImportError as exc:  # pragma: no cover - deployment dependency
                raise RuntimeError(
                    "non-zero Cartesian Z bias requires pytorch-kinematics and "
                    "the bundled EgoVerse PIPER DH URDF"
                ) from exc
            urdf_path = Path(DEFAULT_URDF_PATH)
            with urdf_path.open("rb") as stream:
                self._chain = pk.build_serial_chain_from_urdf(
                    stream.read(),
                    end_link_name="gripper_base",
                    root_link_name="base_link",
                )
        if self._chain is not None:
            self._chain = self._chain.to(device="cpu", dtype=torch.float64)

    @property
    def enabled(self) -> bool:
        return abs(self.bias_mm) > 1e-12

    def settings_report(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "bias_mm": float(self.bias_mm),
            "direction": (
                "up" if self.bias_mm > 0 else "down" if self.bias_mm < 0 else "off"
            ),
            "frame": "PIPER base_link Z",
            "end_link": "gripper_base",
            "max_joint_correction_deg": float(self.max_joint_correction_deg),
            "position_tolerance_mm": float(self.position_tolerance_mm),
            "rotation_tolerance_deg": float(self.rotation_tolerance_deg),
            "solver": (
                "batched_local_damped_least_squares_with_bounded_global_"
                "bias_backoff_no_random_restart"
            ),
            "adaptive_global_bias_backoff": True,
            "adaptive_search_iterations": int(self.adaptive_search_iterations),
        }

    def _solve(
        self,
        q0: torch.Tensor,
        original_matrix: torch.Tensor,
        bias_scale: float,
    ) -> dict[str, torch.Tensor | float]:
        """Solve one constant scaled bias for every action in the chunk."""
        if self._chain is None:
            raise AssertionError("enabled Cartesian Z bias has no kinematic chain")
        q = q0.clone()
        target_position = original_matrix[:, :3, 3].clone()
        target_position[:, 2] += self.bias_m * float(bias_scale)
        target_rotation = original_matrix[:, :3, :3].clone()
        batch = q.shape[0]
        identity = torch.eye(6, dtype=q.dtype).expand(batch, -1, -1)
        per_iteration_limit = math.radians(2.0)
        total_limit = math.radians(self.max_joint_correction_deg)

        for _ in range(self.iterations):
            matrix = self._chain.forward_kinematics(q).get_matrix()
            position_error = target_position - matrix[:, :3, 3]
            rotation_error = _rotation_vector(
                target_rotation @ matrix[:, :3, :3].transpose(-1, -2)
            )
            twist_error = torch.cat((position_error, rotation_error), dim=-1)
            jacobian = self._chain.jacobian(q)
            if jacobian.shape != (batch, 6, 6):
                raise CartesianZBiasError(
                    f"unexpected PIPER Jacobian shape: {tuple(jacobian.shape)}"
                )
            lhs = jacobian @ jacobian.transpose(-1, -2)
            lhs = lhs + (self.damping**2) * identity
            delta = jacobian.transpose(-1, -2) @ torch.linalg.solve(
                lhs, twist_error[..., None]
            )
            delta = delta[..., 0].clamp(-per_iteration_limit, per_iteration_limit)
            q = q + delta
            q = torch.maximum(torch.minimum(q, self._high), self._low)
            q = torch.maximum(torch.minimum(q, q0 + total_limit), q0 - total_limit)

        final_matrix = self._chain.forward_kinematics(q).get_matrix()
        position_error_mm = (
            torch.linalg.vector_norm(
                target_position - final_matrix[:, :3, 3], dim=-1
            )
            * 1000.0
        )
        rotation_error_deg = torch.rad2deg(
            torch.linalg.vector_norm(
                _rotation_vector(
                    target_rotation @ final_matrix[:, :3, :3].transpose(-1, -2)
                ),
                dim=-1,
            )
        )
        return {
            "q": q,
            "actual_delta_mm": (
                final_matrix[:, :3, 3] - original_matrix[:, :3, 3]
            )
            * 1000.0,
            "joint_correction_deg": torch.rad2deg(q - q0),
            "position_error_mm": position_error_mm,
            "rotation_error_deg": rotation_error_deg,
            "max_position_error_mm": float(position_error_mm.max().item()),
            "max_rotation_error_deg": float(rotation_error_deg.max().item()),
        }

    def _solution_is_valid(self, solution: dict[str, torch.Tensor | float]) -> bool:
        return bool(
            float(solution["max_position_error_mm"]) <= self.position_tolerance_mm
            and float(solution["max_rotation_error_deg"])
            <= self.rotation_tolerance_deg
        )

    def apply(self, physical_actions: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        actions = np.asarray(physical_actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError(f"expected finite decoded action chunk (T,7), got {actions.shape}")
        output = actions.copy()
        if not self.enabled:
            return output.astype(np.float32), self.settings_report()
        if self._chain is None:
            raise AssertionError("enabled Cartesian Z bias has no kinematic chain")

        q0 = torch.as_tensor(actions[:, :6], dtype=torch.float64)
        with torch.no_grad():
            original_matrix = self._chain.forward_kinematics(q0).get_matrix()
            requested_solution = self._solve(q0, original_matrix, 1.0)
            solution = requested_solution
            applied_scale = 1.0
            if not self._solution_is_valid(requested_solution):
                low = 0.0
                high = 1.0
                # Scale zero is the original decoded action and is therefore
                # an exact, branch-preserving fallback.  Search for the
                # largest *constant* bias that every action can realize so no
                # within-chunk Z step is introduced.
                solution = self._solve(q0, original_matrix, low)
                for _ in range(self.adaptive_search_iterations):
                    candidate_scale = 0.5 * (low + high)
                    candidate = self._solve(q0, original_matrix, candidate_scale)
                    if self._solution_is_valid(candidate):
                        low = candidate_scale
                        solution = candidate
                    else:
                        high = candidate_scale
                applied_scale = low

        q = solution["q"]
        actual_delta_mm = solution["actual_delta_mm"]
        joint_correction_deg = solution["joint_correction_deg"]
        if not isinstance(q, torch.Tensor):
            raise AssertionError("Cartesian Z-bias solver returned invalid joints")
        if not isinstance(actual_delta_mm, torch.Tensor) or not isinstance(
            joint_correction_deg, torch.Tensor
        ):
            raise AssertionError("Cartesian Z-bias solver returned invalid diagnostics")
        max_position_error_mm = float(solution["max_position_error_mm"])
        max_rotation_error_deg = float(solution["max_rotation_error_deg"])
        if not self._solution_is_valid(solution):
            raise CartesianZBiasError(
                "Cartesian Z bias adaptive fallback failed validation: "
                f"position_error={max_position_error_mm:.3f}mm, "
                f"rotation_error={max_rotation_error_deg:.3f}deg"
            )
        achieved_z = actual_delta_mm[:, 2].cpu().numpy()
        report = self.settings_report()
        report.update(
            {
                "actions": int(len(actions)),
                "requested_z_delta_mm": float(self.bias_mm),
                "applied_z_delta_mm": float(self.bias_mm * applied_scale),
                "applied_bias_fraction": float(applied_scale),
                "adaptive_reduction_applied": bool(applied_scale < 1.0 - 1e-12),
                "requested_max_position_error_mm": float(
                    requested_solution["max_position_error_mm"]
                ),
                "requested_max_rotation_error_deg": float(
                    requested_solution["max_rotation_error_deg"]
                ),
                "achieved_z_delta_mm_min": float(np.min(achieved_z)),
                "achieved_z_delta_mm_mean": float(np.mean(achieved_z)),
                "achieved_z_delta_mm_max": float(np.max(achieved_z)),
                "max_abs_xy_delta_mm": float(
                    np.max(np.abs(actual_delta_mm[:, :2].cpu().numpy()))
                ),
                "max_position_error_mm": max_position_error_mm,
                "max_rotation_error_deg": max_rotation_error_deg,
                "max_abs_joint_correction_deg": float(
                    torch.max(torch.abs(joint_correction_deg)).item()
                ),
                "joint_correction_rms_deg": float(
                    torch.sqrt(torch.mean(torch.square(joint_correction_deg))).item()
                ),
            }
        )
        output[:, :6] = q.cpu().numpy()
        return output.astype(np.float32), report

    def apply_to_prediction_report(
        self, prediction_report: dict[str, Any] | None
    ) -> dict[str, Any]:
        if prediction_report is None:
            raise CartesianZBiasError("prediction report is missing")
        report = dict(prediction_report)
        physical = np.asarray(report.get("selected_physical_actions"))
        transformed, bias_report = self.apply(physical)
        report["selected_physical_actions"] = transformed.tolist()
        report["cartesian_z_bias"] = bias_report
        return report
