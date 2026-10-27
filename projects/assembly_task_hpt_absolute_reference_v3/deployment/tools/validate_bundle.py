#!/usr/bin/env python3
"""Static Assembly New1 bundle validation; never imports robot drivers."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import py_compile
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate(require_pi05_assets: bool = False) -> dict:
    manifest = json.loads((BUNDLE_ROOT / "manifest.json").read_text())
    checked_hashes: dict[str, str] = {}
    for profile_name, profile in manifest["profiles"].items():
        relative = profile["checkpoint"]
        path = BUNDLE_ROOT / relative
        require(path.is_file(), f"missing checkpoint for {profile_name}: {path}")
        actual = sha256_file(path)
        require(actual == profile["sha256"], f"checkpoint SHA mismatch: {profile_name}")
        checked_hashes[relative] = actual
    for relative, expected in manifest["immutable_runtime_files"].items():
        path = BUNDLE_ROOT / relative
        require(path.is_file(), f"missing immutable runtime file: {path}")
        actual = sha256_file(path)
        require(actual == expected, f"runtime SHA mismatch: {relative}")
        checked_hashes[relative] = actual

    camera_contract = json.loads(
        (BUNDLE_ROOT / "assets/camera_contract.json").read_text()
    )
    expected_serials = {
        name: spec["serial"] for name, spec in manifest["cameras"].items()
    }
    collection_serials = {
        "head": expected_serials["front"],
        "wrist": expected_serials["wrist"],
        "side": expected_serials["side"],
    }
    require(
        camera_contract["serials"] == collection_serials,
        "camera serial contract differs from deployment manifest",
    )
    require(
        camera_contract["included_source_keys"]
        == [
            "observation.images.head",
            "observation.images.wrist",
            "observation.images.side",
        ],
        "three-camera source ordering changed",
    )

    reset = json.loads((BUNDLE_ROOT / "assets/reset_control.json").read_text())
    follower_initial = json.loads(
        (BUNDLE_ROOT / "assets/initial_pose.json").read_text()
    )
    recorded_leader_initial = json.loads(
        (BUNDLE_ROOT / "assets/initial_pose_original_leader.json").read_text()
    )
    initial = json.loads(
        (
            BUNDLE_ROOT
            / "assets/initial_pose_follower_physical_on_original_leader.json"
        ).read_text()
    )
    require(
        follower_initial["joint_positions_deg"]
        == reset["initial_pose"]["follower_joint_positions_deg"],
        "saved model-space Home does not match collection follower pose",
    )
    require(
        follower_initial["gripper_opening_mm"]
        == reset["initial_pose"]["follower_gripper_opening_mm"],
        "saved model-space gripper Home does not match collection follower pose",
    )
    require(
        recorded_leader_initial["joint_positions_deg"]
        == reset["initial_pose"]["leader_joint_positions_deg"],
        "recorded leader Home provenance does not match collection metadata",
    )
    require(
        recorded_leader_initial["gripper_opening_mm"]
        == reset["initial_pose"]["leader_gripper_opening_mm"],
        "recorded leader gripper Home provenance does not match collection metadata",
    )
    require(
        initial["joint_positions_deg"]
        == reset["initial_pose"]["follower_joint_positions_deg"],
        "saved model-space Home does not preserve follower physical joint angles",
    )
    require(
        initial["gripper_opening_mm"]
        == reset["initial_pose"]["follower_gripper_opening_mm"],
        "deployment camera Home does not preserve follower physical gripper aperture",
    )

    source_expectations = {
        "robot/deploy_assembly_new1_hpt_joint.py": [
            '"side": RealSenseCameraConfig(',
            'observation["side"]',
            "enable_motion_on_connect=False",
            "execute mode requires --allow-motion",
            "def adopt_already_enabled_motion_hold_current",
            'context="deployment control loop"',
            '"adopt_already_enabled_no_quick_stop"',
            "JointSpaceAdapter.from_files",
            "adapter.hardware_observation_to_model_observation",
            "adapter.model_driver_action_to_hardware_action",
        ],
        "robot/joint_space_adapter.py": [
            '"physical-identity"',
            '"recorded-percentile-affine"',
            "_base_hardware_to_model",
            "approved_active",
            "pose_count",
            "hardware_observation_to_model_observation",
            "model_driver_action_to_hardware_action",
            "model physical joint target",
        ],
        "src/lerobot/policies/egoverse_hpt/modeling_egoverse_hpt_joint.py": [
            '"observations.images.side_img"',
            '"hpt_expert_action_v3"',
            '"piper_absolute_reference_joint_v3"',
            '"AbsoluteReferenceJointPolicyV3"',
            "direct_absolute_joint_radians_no_ik",
            'data["_proprio_norm_mode"] = norm_mode',
        ],
        "robot/deploy_assembly_new1_pi05_mixed_v2_joint.py": [
            '"right_wrist_0_rgb": "side_img"',
            "assembly-new1-pi05-mixed-v2-30k",
            "PIPER_RECORDING_EXTRINSICS_JSON",
            'sys.modules["deploy_egoverse_hpt_joint"] = assembly_legacy',
            "JointSpaceAdapter.from_files",
        ],
        "robot/deploy_assembly_new1_hpt_joint_async_rtc.py": [
            "class BackgroundPredictor",
            "class DecodedJointRtcController",
            "class AsyncFullRunImageWriter",
            'observation["side"]',
            '"decoded_physical_6d_joint_radians"',
            "CartesianZBiasTransform",
            '"--end-effector-z-bias-mm"',
            '"--record-full-run-images"',
            '"full_run_image_writer_closed"',
            "legacy.adopt_already_enabled_motion_hold_current(robot)",
            'context="deployment control loop"',
            '"adopt_already_enabled_no_quick_stop"',
        ],
        "robot/cartesian_z_bias.py": [
            "class CartesianZBiasTransform",
            "DEFAULT_URDF_PATH",
            '"requested_z_delta_mm"',
            '"max_position_error_mm"',
            '"max_rotation_error_deg"',
        ],
        "robot/deploy_assembly_new1_pi05_mixed_v2_joint_async_rtc.py": [
            'sys.modules["deploy_egoverse_hpt_joint"] = assembly_legacy',
            "EgoVersePI05MixedCamFrameV5JointPolicy",
            "rtc.run_live_async_rtc",
            '"single_worker_async_absolute_action_tick_rtc"',
        ],
        "robot/home_assembly_new1.py": [
            '"assembly_new1_model_free_piper_sdk_interface"',
            'choices=("check", "execute")',
            '"checkpoint_loaded": False',
            '"policy_constructed": False',
            '"camera_opened": False',
            "enable_motion=False",
            '"--gripper-effort"',
            '"gripper_effort_nm"',
        ],
        "robot/gripper_only_control.py": [
            'choices=("check", "execute")',
            'choices=("open", "close")',
            "adopt_already_enabled_motion_hold_current",
            "gripper_effort_nm",
            "object_contact",
            "disable_torque=False",
        ],
        "runtime/lerobot/src/lerobot/robots/piper/config_piper.py": [
            "gripper_effort: int = 5000",
            "0.001 N*m SDK units",
        ],
        "runtime/lerobot/src/lerobot/robots/piper/piper.py": [
            "gripper_effort=self.config.gripper_effort",
        ],
        "runtime/lerobot/src/lerobot/robots/piper/piper_sdk_interface.py": [
            "self.gripper_effort = int(gripper_effort)",
            "GripperCtrl(scaled[6], self.gripper_effort",
            "GripperCtrl(gripper, self.gripper_effort",
        ],
        "robot/wrist_visual_zero_calibration.py": [
            'choices=("check", "execute")',
            "central differences",
            "damped_joint_update",
            "candidate_not_activated",
            "deployment_files_modified\": False",
            "enable_motion=False",
        ],
        "robot/joint_mapping_multi_pose_validation.py": [
            'choices=("check", "execute", "approve")',
            "validated_pending_approval",
            "APPROVE_VALIDATED_MAPPING",
            "verify_fk_routes",
            "active_mapping_modified",
            "execute mode requires --allow-motion",
        ],
        "training_code/EgoVerse_pi05/egomimic/algo/pi_egoverse_mixed_camframe_v5.py": [
            "side_image: torch.Tensor",
            '"observations.images.side_img": side_image',
        ],
        "training_code/EgoVerse_pi05/egomimic/utils/action_utils.py": [
            "class HumanRightXYZ6DActionConverter",
            "class PiperJointActionConverter",
        ],
        "training_code/EgoVerse/egomimic/models/residual_joint_policy_clean_v2.py": [
            "class ResidualJointPolicyCleanV2",
            'contract_name = "piper_residual_joint_clean_v2"',
        ],
        "training_code/EgoVerse/egomimic/models/absolute_reference_joint_policy_v3.py": [
            "class AbsoluteReferenceJointPolicyV3",
            'contract_name = "piper_absolute_reference_joint_v3"',
            "joint_reference_center_rad",
            "joint_correction_scale_rad",
        ],
        "training_code/EgoVerse/egomimic/algo/hpt_absolute_reference_v3.py": [
            "class HPTAbsoluteReferenceV3",
            "compute_recovery_losses",
        ],
    }
    for relative, markers in source_expectations.items():
        text = (BUNDLE_ROOT / relative).read_text()
        for marker in markers:
            require(marker in text, f"{relative} lacks required marker {marker!r}")

    compile_paths = list((BUNDLE_ROOT / "robot").glob("*.py"))
    compile_paths += list(
        (BUNDLE_ROOT / "src/lerobot/policies/egoverse_hpt").glob("*.py")
    )
    compile_paths += [BUNDLE_ROOT / "tools/validate_bundle.py"]
    compile_paths += list((BUNDLE_ROOT / "overlays/egomimic/algo").glob("*.py"))
    compile_paths += list((BUNDLE_ROOT / "overlays/egomimic/models").glob("*.py"))
    for path in compile_paths:
        py_compile.compile(str(path), doraise=True)

    async_launchers = [
        "run_hpt_mixed_v2_async_rtc.sh",
        "run_hpt_expert_action_v3_async_rtc.sh",
        "run_hpt_absolute_reference_v3_25k_async_rtc.sh",
        "run_hpt_absolute_reference_v3_30k_async_rtc.sh",
        "run_hpt_absolute_reference_v3_30k_async_rtc_gripper_deadline.sh",
        "run_pi05_mixed_v2_async_rtc.sh",
    ]
    for relative in async_launchers:
        text = (BUNDLE_ROOT / relative).read_text(encoding="utf-8")
        require("--query-frequency 30" in text, f"{relative} does not select 30 RTC ticks")
        require(
            "--gripper-prediction-source full-horizon" in text,
            f"{relative} does not preserve the full binary gripper horizon",
        )
    deadline_launcher = (
        BUNDLE_ROOT
        / "run_hpt_absolute_reference_v3_30k_async_rtc_gripper_deadline.sh"
    )
    deadline_text = deadline_launcher.read_text(encoding="utf-8")
    require(
        "--gripper-initial-close-deadline-latch" in deadline_text
        and "--gripper-initial-close-guard" in deadline_text,
        "gripper-deadline launcher does not enable both deadline and pose gates",
    )

    full_run_image_launcher = (
        BUNDLE_ROOT / "run_hpt_mixed_v2_async_rtc_record_images.sh"
    )
    require(
        full_run_image_launcher.is_file(),
        "missing full-run three-camera recording launcher",
    )
    full_run_image_launcher_text = full_run_image_launcher.read_text(
        encoding="utf-8"
    )
    require(
        "run_hpt_mixed_v2_async_rtc.sh" in full_run_image_launcher_text
        and "--record-full-run-images" in full_run_image_launcher_text
        and "--full-run-image-queue-size 64" in full_run_image_launcher_text,
        "full-run image launcher does not wrap the audited HPT RTC launcher",
    )

    percentile_launcher = (
        BUNDLE_ROOT / "run_hpt_mixed_v2_async_rtc_percentile_mapping.sh"
    )
    require(percentile_launcher.is_file(), "missing percentile mapping A/B launcher")
    percentile_text = percentile_launcher.read_text(encoding="utf-8")
    require(
        "run_hpt_mixed_v2_async_rtc.sh" in percentile_text
        and "--joint-mapping-mode recorded-percentile-affine" in percentile_text,
        "percentile mapping launcher does not pin the audited HPT RTC base and mapping mode",
    )
    percentile_home_launcher = BUNDLE_ROOT / "run_home_percentile_mapping_only.sh"
    require(percentile_home_launcher.is_file(), "missing percentile mapping Home launcher")
    percentile_home_text = percentile_home_launcher.read_text(encoding="utf-8")
    require(
        "run_home_only.sh" in percentile_home_text
        and "assets/initial_pose_original_leader.json" in percentile_home_text,
        "percentile mapping Home launcher does not pin the recorded leader Home",
    )

    home_launcher = BUNDLE_ROOT / "run_home_only.sh"
    require(home_launcher.is_file(), "missing model-free home launcher")
    home_source = (BUNDLE_ROOT / "robot/home_assembly_new1.py").read_text()
    home_tree = ast.parse(home_source)
    imported_modules: list[str] = []
    for node in ast.walk(home_tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    forbidden_import_prefixes = (
        "torch",
        "lerobot.policies",
        "lerobot.cameras",
        "egomimic",
    )
    require(
        not any(
            module == prefix or module.startswith(prefix + ".")
            for module in imported_modules
            for prefix in forbidden_import_prefixes
        ),
        f"model-free home imports model/camera code: {imported_modules}",
    )
    # The audit field ``checkpoint_loaded=False`` is intentional; no
    # checkpoint path or loader may exist in this entry point.
    require(
        "--checkpoint" not in home_source and "load_checkpoint" not in home_source,
        "model-free home entry point unexpectedly exposes checkpoint loading",
    )
    require(
        'assets/initial_pose_follower_physical_on_original_leader.json'
        in home_source,
        "model-free Home does not target the recorded follower physical pose",
    )

    replay_launcher = BUNDLE_ROOT / "run_episode_replay.sh"
    replay_source_path = BUNDLE_ROOT / "robot/replay_assembly_new1_episode.py"
    replay_trajectory = BUNDLE_ROOT / "replay/assembly_new1_ep0000_original_leader.npz"
    require(replay_launcher.is_file(), "missing model-free episode replay launcher")
    require(replay_source_path.is_file(), "missing model-free episode replay source")
    require(replay_trajectory.is_file(), "missing audited episode-0 replay trajectory")
    replay_source = replay_source_path.read_text(encoding="utf-8")
    require(
        "checkpoint" not in replay_source.lower() or "checkpoint_loaded" in replay_source,
        "episode replay unexpectedly exposes checkpoint loading",
    )
    require(
        'choices=("check", "live-dry-run", "execute")' in replay_source,
        "episode replay lacks the three-stage safety ladder",
    )
    require(
        "assembly_new1_physical_joint_episode_replay_v2" in replay_source,
        "episode replay does not require the measured follower-state schema",
    )
    require(
        "--hold-gripper-closed" in replay_source
        and "hold_closed_zero_percent" in replay_source
        and "latch_replay_pose" in replay_source,
        "episode replay does not preserve an explicit closed clamp target",
    )

    calibration_launcher = BUNDLE_ROOT / "run_wrist_zero_calibration.sh"
    require(calibration_launcher.is_file(), "missing wrist zero calibration launcher")
    calibration_source = (
        BUNDLE_ROOT / "robot/wrist_visual_zero_calibration.py"
    ).read_text(encoding="utf-8")
    require(
        "execute mode requires --allow-motion" in calibration_source,
        "wrist zero calibration lacks explicit execute authorization",
    )
    require(
        "--checkpoint" not in calibration_source and "load_checkpoint" not in calibration_source,
        "model-free wrist zero calibration unexpectedly exposes checkpoint loading",
    )

    mapping_launcher = BUNDLE_ROOT / "run_joint_mapping_validation.sh"
    require(mapping_launcher.is_file(), "missing joint mapping validation launcher")
    mapping_spec = json.loads(
        (BUNDLE_ROOT / "assets/joint_mapping_validation_poses.json").read_text()
    )
    require(
        mapping_spec.get("schema")
        == "assembly_new1_joint_mapping_validation_poses_v1",
        "invalid joint mapping validation pose schema",
    )
    require(
        len(mapping_spec.get("poses", [])) >= 3,
        "joint mapping validation requires at least three poses",
    )
    for pose in mapping_spec["poses"]:
        require(
            set(pose["reference_images"]) == {"front", "wrist", "side"},
            f"mapping pose {pose.get('id')} lacks three-camera references",
        )
        for relative in pose["reference_images"].values():
            require((BUNDLE_ROOT / relative).is_file(), f"missing mapping reference {relative}")
    mapping_source = (
        BUNDLE_ROOT / "robot/joint_mapping_multi_pose_validation.py"
    ).read_text(encoding="utf-8")
    require(
        "--checkpoint" not in mapping_source and "load_checkpoint" not in mapping_source,
        "model-free mapping validator unexpectedly exposes checkpoint loading",
    )

    pi05_assets = {
        "base_weights": (BUNDLE_ROOT / "pi05_runtime/pi05_base_pytorch").is_dir(),
        "tokenizer": (BUNDLE_ROOT / "pi05_runtime/paligemma_tokenizer.model").is_file(),
        "python_overlay": (BUNDLE_ROOT / "pi05_runtime/python_overlay").is_dir(),
    }
    if require_pi05_assets:
        require(all(pi05_assets.values()), f"pi0.5 external assets incomplete: {pi05_assets}")

    return {
        "bundle_contract": manifest["bundle_contract"],
        "profiles": list(manifest["profiles"]),
        "verified_sha256_files": checked_hashes,
        "camera_serials": expected_serials,
        "initial_pose_matches_training_follower_physical_home": True,
        "python_sources_compile": True,
        "async_rtc_launchers": async_launchers,
        "full_run_image_launcher": str(
            full_run_image_launcher.relative_to(BUNDLE_ROOT)
        ),
        "model_free_home_launcher": str(home_launcher.relative_to(BUNDLE_ROOT)),
        "model_free_episode_replay_launcher": str(
            replay_launcher.relative_to(BUNDLE_ROOT)
        ),
        "model_free_wrist_zero_calibration_launcher": str(
            calibration_launcher.relative_to(BUNDLE_ROOT)
        ),
        "model_free_joint_mapping_validation_launcher": str(
            mapping_launcher.relative_to(BUNDLE_ROOT)
        ),
        "pi05_external_assets": pi05_assets,
        "robot_constructed": False,
        "camera_opened": False,
        "can_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-pi05-assets", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.require_pi05_assets), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
