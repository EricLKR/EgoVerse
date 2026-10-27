from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_bundle", BUNDLE_ROOT / "tools/validate_bundle.py"
)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)


class BundleContractTest(unittest.TestCase):
    def test_complete_static_contract_without_hardware(self):
        report = VALIDATOR.validate(require_pi05_assets=False)
        self.assertEqual(len(report["profiles"]), 5)
        self.assertIn(
            "assembly-new1-hpt-absolute-reference-v3-30k",
            report["profiles"],
        )
        self.assertTrue(
            report["initial_pose_matches_training_follower_physical_home"]
        )
        self.assertFalse(report["robot_constructed"])
        self.assertFalse(report["camera_opened"])
        self.assertFalse(report["can_opened"])


if __name__ == "__main__":
    unittest.main()
