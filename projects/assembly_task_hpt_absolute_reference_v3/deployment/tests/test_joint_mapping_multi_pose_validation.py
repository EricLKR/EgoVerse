from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = BUNDLE_ROOT / "robot/joint_mapping_multi_pose_validation.py"
SPEC = importlib.util.spec_from_file_location("joint_mapping_validator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class JointMappingMultiPoseValidationTest(unittest.TestCase):
    def test_static_spec_has_three_safe_mapped_poses(self) -> None:
        spec, candidate_path, candidate, offset, poses = validator.load_and_validate_spec(
            BUNDLE_ROOT / "assets/joint_mapping_validation_poses.json",
            BUNDLE_ROOT / "assets/piper_original_leader_calibration.json",
            joint_limit_margin_deg=2.0,
            minimum_audited_route_z_mm=200.0,
        )
        self.assertEqual(spec["episode_index"], 0)
        self.assertTrue(candidate_path.is_file())
        self.assertEqual(candidate["status"], "candidate_not_activated")
        self.assertEqual(offset.shape, (6,))
        self.assertEqual(len(poses), 3)
        self.assertTrue(all(pose["audited_route_min_z_mm"] >= 200 for pose in poses))

    def test_result_summary_requires_all_three(self) -> None:
        passed = validator.summarize_results(
            [
                {"pose_id": "a", "passed": True},
                {"pose_id": "b", "passed": True},
                {"pose_id": "c", "passed": True},
            ]
        )
        self.assertTrue(passed["passed"])
        failed = validator.summarize_results(
            [
                {"pose_id": "a", "passed": True},
                {"pose_id": "b", "passed": False},
                {"pose_id": "c", "passed": True},
            ]
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["failed_pose_ids"], ["b"])

    def test_wrist_gate_checks_translation_rotation_and_scale(self) -> None:
        good = {"translation_norm_px": 4.9, "rotation_deg": -0.7, "scale": 1.019}
        self.assertTrue(
            validator.wrist_report_passes(
                good,
                translation_tolerance_px=5.0,
                rotation_tolerance_deg=0.75,
                scale_tolerance=0.02,
            )
        )
        for key, value in (
            ("translation_norm_px", 5.1),
            ("rotation_deg", 0.8),
            ("scale", 1.021),
        ):
            bad = dict(good)
            bad[key] = value
            self.assertFalse(
                validator.wrist_report_passes(
                    bad,
                    translation_tolerance_px=5.0,
                    rotation_tolerance_deg=0.75,
                    scale_tolerance=0.02,
                )
            )

    def test_approval_payload_requires_unchanged_candidate_and_three_passes(self) -> None:
        source_candidate = BUNDLE_ROOT / "assets/joint_zero_offset_home_candidate_20261020.json"
        candidate = validator.load_candidate(source_candidate)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidate.json"
            candidate_path.write_bytes(source_candidate.read_bytes())
            report_path = root / "report.json"
            report = {
                "candidate_sha256": validator.sha256_file(candidate_path),
                "summary": {"passed": True, "pose_count": 3},
                "pose_results": [
                    {
                        "pose_id": name,
                        "passed": True,
                        "wrist_alignment": {
                            "translation_norm_px": 1.0,
                            "rotation_deg": 0.1,
                            "scale": 1.0,
                        },
                    }
                    for name in ("a", "b", "c")
                ],
            }
            report_path.write_text(json.dumps(report), encoding="utf-8")
            payload = validator.build_approval_payload(
                candidate_path=candidate_path,
                candidate=candidate,
                report_path=report_path,
                report=report,
            )
            self.assertEqual(payload["status"], "approved_active")
            self.assertEqual(payload["multi_pose_validation"]["pose_count"], 3)
            candidate_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate changed"):
                validator.build_approval_payload(
                    candidate_path=candidate_path,
                    candidate=candidate,
                    report_path=report_path,
                    report=report,
                )

    def test_model_and_hardware_mapping_are_exact_inverses(self) -> None:
        offset = np.asarray(
            validator.load_candidate(
                BUNDLE_ROOT / "assets/joint_zero_offset_home_candidate_20261020.json"
            )["hardware_minus_model_joint_offset_deg"]
        )
        model = np.asarray([12.0, 80.0, -70.0, -20.0, 60.0, 30.0])
        hardware = model + offset
        np.testing.assert_allclose(hardware - offset, model, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
