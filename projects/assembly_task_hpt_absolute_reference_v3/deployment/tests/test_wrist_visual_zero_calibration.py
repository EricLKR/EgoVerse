from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_ROOT / "robot"))

from wrist_visual_zero_calibration import (
    alignment_converged,
    damped_joint_update,
    parse_active_joints,
    weighted_alignment_error,
)


class WristVisualZeroCalibrationTest(unittest.TestCase):
    def test_known_linear_problem_reduces_visual_error(self):
        jacobian = np.asarray(
            [
                [8.0, -1.0, 0.5],
                [0.5, 7.0, -2.0],
                [1.0, 0.5, 4.0],
            ],
            dtype=np.float64,
        )
        error = np.asarray([-40.0, 12.0, -5.0], dtype=np.float64)
        result = damped_joint_update(
            error,
            jacobian,
            damping=0.25,
            max_step_deg=1.0,
            current_offset_deg=np.zeros(3),
            max_total_offset_deg=8.0,
        )
        self.assertLess(result.predicted_error_after, result.predicted_error_before)
        self.assertLessEqual(float(np.max(np.abs(result.step_deg))), 1.0 + 1e-12)
        self.assertEqual(result.rank, 3)

    def test_rank_deficient_probe_is_rejected(self):
        jacobian = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        with self.assertRaisesRegex(RuntimeError, "rank deficient"):
            damped_joint_update(
                [10.0, 5.0, 0.0],
                jacobian,
                damping=0.25,
                max_step_deg=1.0,
                current_offset_deg=np.zeros(3),
                max_total_offset_deg=8.0,
            )

    def test_total_offset_trust_region_is_enforced(self):
        result = damped_joint_update(
            [-10.0, -10.0, -10.0],
            np.eye(3),
            damping=0.01,
            max_step_deg=1.0,
            current_offset_deg=[7.8, 0.0, 0.0],
            max_total_offset_deg=8.0,
        )
        self.assertLessEqual(
            float(np.max(np.abs(np.asarray([7.8, 0.0, 0.0]) + result.step_deg))),
            8.0 + 1e-12,
        )

    def test_report_weighting_and_convergence(self):
        report = {
            "tx_px": 1.0,
            "ty_px": -2.0,
            "translation_norm_px": np.hypot(1.0, -2.0),
            "rotation_deg": 0.25,
        }
        error = weighted_alignment_error(report, rotation_weight_px_per_deg=10.0)
        self.assertTrue(np.allclose(error, [1.0, -2.0, 2.5]))
        self.assertTrue(
            alignment_converged(
                report,
                translation_tolerance_px=3.0,
                rotation_tolerance_deg=0.5,
            )
        )

    def test_only_wrist_joints_are_accepted(self):
        self.assertEqual(parse_active_joints("4,5,6"), (3, 4, 5))
        with self.assertRaises(Exception):
            parse_active_joints("1,2,3")


if __name__ == "__main__":
    unittest.main()
