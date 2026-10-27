#!/usr/bin/env python3
"""CPU-only integrity checks for the Assembly New1 three-camera bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    required = [
        ROOT / "assets/camera_contract.json",
        ROOT / "assets/piper_follower_calibration.json",
        ROOT / "assets/piper_front_extrinsics_recording_20261016.json",
        ROOT / "assets/formal_extrinsics_acceptance_PENDING.json",
        ROOT / "assets/piper_intrinsics_head_wrist_640x480.json",
        ROOT / "assets/split_90_5_5.json",
        ROOT / "assets/ego_split_54_3_3.json",
        ROOT / "scripts/prepare_ego_split.py",
        ROOT / "scripts/convert_ego_gen2_embedded_vio.py",
        ROOT / "scripts/finalize_ego_conversion.py",
        ROOT / "scripts/plot_ego_trajectories.py",
        ROOT / "patches/aria_utils_gen2_embedded_vio.patch",
        ROOT / "scripts/convert_robot_3cam.py",
        ROOT / "scripts/preflight_robot_3cam.py",
        ROOT / "configs/hpt_data_assembly_new1_90_5_5.yaml",
        ROOT / "configs/hpt_data_assembly_new1_90_5_5_test.yaml",
        ROOT / "configs/hpt_model_assembly_new1_3cam.yaml",
        ROOT / "configs/hpt_evaluator_assembly_new1_right.yaml",
        ROOT / "configs/pi05_data_assembly_new1_90_5_5.yaml",
        ROOT / "configs/pi05_data_assembly_new1_90_5_5_test.yaml",
        ROOT / "configs/pi05_model_assembly_new1_3cam.yaml",
        ROOT / "code_overrides/egomimic/rldb/embodiment/piper_human_joint_three_camera.py",
        ROOT / "code_overrides/egomimic/rldb/embodiment/pi05_expert_action_v3_three_camera.py",
        ROOT / "code_overrides/egomimic/rldb/embodiment/test_piper_human_joint_three_camera.py",
        ROOT / "hpc/train_hpt_assembly_new1_v2.slurm",
        ROOT / "hpc/smoke_pi05_assembly_new1_v2.slurm",
        ROOT / "hpc/train_pi05_assembly_new1_v2_30k.slurm",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing bundle files: {missing}")

    split = json.loads((ROOT / "assets/split_90_5_5.json").read_text())
    groups = split["episodes"]
    if {name: len(values) for name, values in groups.items()} != {
        "train": 90,
        "valid": 5,
        "test": 5,
    }:
        raise ValueError("split is not 90/5/5")
    flat = [index for values in groups.values() for index in values]
    if sorted(flat) != list(range(100)) or len(set(flat)) != 100:
        raise ValueError("split does not cover each source episode exactly once")

    ego_split = json.loads((ROOT / "assets/ego_split_54_3_3.json").read_text())
    ego_groups = ego_split["episodes"]
    if {name: len(values) for name, values in ego_groups.items()} != {
        "train": 54,
        "valid": 3,
        "test": 3,
    }:
        raise ValueError("Ego split is not 54/3/3")
    ego_flat = [index for values in ego_groups.values() for index in values]
    if sorted(ego_flat) != list(range(1, 61)) or len(set(ego_flat)) != 60:
        raise ValueError("Ego split does not cover each source episode exactly once")

    camera = json.loads((ROOT / "assets/camera_contract.json").read_text())
    if camera.get("phase_sampling") is not False:
        raise ValueError("phase sampling must remain disabled")
    if camera.get("excluded_source_keys") != ["observation.images.aria"]:
        raise ValueError("robot Aria exclusion contract changed")
    if camera.get("included_source_keys") != [
        "observation.images.head",
        "observation.images.wrist",
        "observation.images.side",
    ]:
        raise ValueError("source camera contract is not exactly head/wrist/side")
    if set(camera.get("zarr_mapping", {}).values()) != {
        "images.front_1",
        "images.wrist_1",
        "images.side_1",
    }:
        raise ValueError("Zarr camera contract is not exactly front/wrist/side")

    extrinsics_path = ROOT / "assets/piper_front_extrinsics_recording_20261016.json"
    extrinsics = json.loads(extrinsics_path.read_text())
    if extrinsics.get("status") != "recording_calibration":
        raise ValueError("recording extrinsics provenance was altered")
    acceptance = json.loads(
        (ROOT / "assets/formal_extrinsics_acceptance_PENDING.json").read_text()
    )
    if acceptance.get("extrinsics_sha256") != sha256(extrinsics_path):
        raise ValueError("formal acceptance refers to a different extrinsics file")
    if acceptance.get("accepted") not in (False, True):
        raise ValueError("formal acceptance flag must be boolean")
    if acceptance.get("accepted") is True:
        if not acceptance.get("accepted_at") or not acceptance.get("accepted_by"):
            raise ValueError("accepted extrinsics require accepted_at and accepted_by")

    for path in (ROOT / "configs").glob("*.yaml"):
        payload = yaml.safe_load(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: YAML root is not a mapping")
        if "phase" in path.read_text().lower() or "stage" in path.read_text().lower():
            raise ValueError(f"{path}: phase/stage sampling text is forbidden in this bundle")

    hpt = yaml.safe_load((ROOT / "configs/hpt_data_assembly_new1_90_5_5.yaml").read_text())
    hpt_model = yaml.safe_load((ROOT / "configs/hpt_model_assembly_new1_3cam.yaml").read_text())
    pi05 = yaml.safe_load((ROOT / "configs/pi05_data_assembly_new1_90_5_5.yaml").read_text())
    pi05_test = yaml.safe_load(
        (ROOT / "configs/pi05_data_assembly_new1_90_5_5_test.yaml").read_text()
    )
    for label, payload in (("HPT", hpt), ("pi0.5", pi05)):
        for split_name in ("train_datasets", "valid_datasets"):
            if set(payload[split_name]) != {"piper_left_arm", "human_right_arm"}:
                raise ValueError(f"{label} {split_name} is not mixed robot+human")
            robot = payload[split_name]["piper_left_arm"]
            target = robot["resolver"]["key_map"]["_target_"]
            if "three_camera" not in target:
                raise ValueError(f"{label} {split_name} does not use three-camera adapter")
    if int(pi05["valid_future_horizon"]) != 30:
        raise ValueError("pi0.5 horizon must remain 30")
    if float(
        hpt_model["robomimic_model"]["head_specs"]["piper_left_arm_actions_joints"]["max_residual_deg"]
    ) != 110.0:
        raise ValueError("HPT New1 residual scale must cover the measured 101.77deg J5 target")
    if hpt_model["robomimic_model"].get("domains") != ["piper_left_arm", "human_right_arm"]:
        raise ValueError("HPT model does not activate the new right-hand human domain")
    pi05_model = yaml.safe_load((ROOT / "configs/pi05_model_assembly_new1_3cam.yaml").read_text())
    if pi05_model["robomimic_model"].get("domains") != ["piper_left_arm", "human_right_arm"]:
        raise ValueError("pi0.5 model does not activate the new right-hand human domain")
    if pi05_model.get("defaults", [None])[0] != "pi0.5_base":
        raise ValueError("pi0.5 New1 config must not inherit the legacy human-left V5 preset")
    pi05_rules = pi05_model["robomimic_model"]["action_converters"]["rules"]
    if set(pi05_rules) != {"PIPER_LEFT_ARM", "HUMAN_RIGHT_ARM"}:
        raise ValueError("pi0.5 New1 converter rules contain a stale or missing embodiment")
    if "human_left_arm" in (ROOT / "configs/pi05_model_assembly_new1_3cam.yaml").read_text():
        raise ValueError("pi0.5 New1 config still contains the retired human-left domain")
    hpt_evaluator = yaml.safe_load(
        (ROOT / "configs/hpt_evaluator_assembly_new1_right.yaml").read_text()
    )
    if set(hpt_evaluator["viz_func"]) != {"piper_left_arm", "human_right_arm"}:
        raise ValueError("HPT evaluator does not exactly match the New1 domains")
    if any("aria_assembly_1010_zarr" in path.read_text() for path in (ROOT / "configs").glob("*.yaml")):
        raise ValueError("configuration still references the retired human dataset")
    if pi05["_target_"] != "egomimic.pl_utils.pl_data_utils_v5.EpisodeBalancedMultiDataModuleWrapper":
        raise ValueError("pi0.5 loader contract changed")
    if pi05_test.get("defaults", [None])[0] != "pi05_assembly_new1_mixed_90_5_5":
        raise ValueError("pi0.5 test config inherits the wrong installed config name")

    print(json.dumps({"passed": True, "files_checked": len(required), "phase_sampling": False}, indent=2))


if __name__ == "__main__":
    main()
