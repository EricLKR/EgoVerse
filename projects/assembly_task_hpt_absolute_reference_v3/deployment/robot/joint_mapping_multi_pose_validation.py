#!/usr/bin/env python3
"""Validate and explicitly approve an old-follower -> current-arm joint offset.

This is a model-free commissioning tool for the case where Assembly New1 was
recorded with a follower that is no longer available and its original leader
is now used as the only deployment arm.  It reuses three open-gripper,
high-clearance poses and camera frames from the existing dataset; it does not
require a second robot or any new demonstration.

``check`` is static: no camera, CAN, driver, checkpoint, or motion code is
opened. ``execute`` requires ``--allow-motion`` and validates one constant
six-axis offset at three poses.  A passing execute run creates only a
``validated_pending_approval`` artifact. ``approve`` is a separate static
operation with an exact confirmation token; only it may replace the active
``assets/joint_zero_offset.json`` file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = BUNDLE_ROOT / "assets/joint_mapping_validation_poses.json"
DEFAULT_CALIBRATION = BUNDLE_ROOT / "assets/piper_original_leader_calibration.json"
DEFAULT_INITIAL_POSE = (
    BUNDLE_ROOT / "assets/initial_pose_follower_physical_on_original_leader.json"
)
DEFAULT_ACTIVE_OFFSET = BUNDLE_ROOT / "assets/joint_zero_offset.json"
DEFAULT_MANIFEST = BUNDLE_ROOT / "manifest.json"
DEFAULT_OUTPUT_ROOT = BUNDLE_ROOT / "validation/joint_mapping_multi_pose"
APPROVAL_TOKEN = "APPROVE_VALIDATED_MAPPING"
CONTROL_HZ = 30.0
CAMERA_SERIALS = {
    "front": "337322071261",
    "wrist": "427622273943",
    "side": "427622272585",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_bundle_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BUNDLE_ROOT / path


def _six_finite(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain six finite values: {result.tolist()}")
    return result


def load_candidate(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema") != "assembly_new1_wrist_visual_zero_offset_candidate_v1":
        raise ValueError(f"unsupported Home candidate schema in {path}")
    if payload.get("status") != "candidate_not_activated":
        raise ValueError(f"Home candidate is not inactive in {path}")
    offset = _six_finite(
        payload.get("hardware_minus_model_joint_offset_deg"), "joint offset"
    )
    if float(np.max(np.abs(offset))) > 10.0:
        raise ValueError(f"candidate exceeds the audited 10 degree bound: {offset.tolist()}")
    if bool(payload.get("deployment_files_modified")):
        raise ValueError("candidate claims deployment files were already modified")
    return payload


def load_limits(path: Path) -> list[tuple[float, float]]:
    payload = load_json(path)
    limits = [
        (
            float(payload[f"joint_{index}"]["range_min"]) / 1000.0,
            float(payload[f"joint_{index}"]["range_max"]) / 1000.0,
        )
        for index in range(6)
    ]
    if any(not math.isfinite(value) for pair in limits for value in pair):
        raise ValueError(f"non-finite calibration in {path}")
    if any(high <= low for low, high in limits):
        raise ValueError(f"invalid calibration limits in {path}: {limits}")
    return limits


def validate_pose_goal(
    model_deg: Sequence[float],
    offset_deg: Sequence[float],
    limits: Sequence[tuple[float, float]],
    *,
    margin_deg: float,
) -> np.ndarray:
    goal = _six_finite(model_deg, "model pose") + _six_finite(offset_deg, "offset")
    for index, (value, (low, high)) in enumerate(zip(goal, limits, strict=True)):
        if not low + margin_deg <= value <= high - margin_deg:
            raise RuntimeError(
                f"mapped joint {index + 1}={value:.3f} deg violates the "
                f"{margin_deg:.3f} deg margin [{low:.3f},{high:.3f}]"
            )
    return goal


def verify_fk_routes(
    *,
    origin_deg: Sequence[float],
    poses: Sequence[dict[str, Any]],
    minimum_route_z_mm: float,
    samples: int = 101,
) -> list[dict[str, Any]]:
    """Offline end-effector FK audit; this never opens CAN or constructs a robot."""
    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics

    if samples < 2:
        raise ValueError("FK route audit requires at least two samples")
    fk = C_PiperForwardKinematics(dh_is_offset=1)
    origin = _six_finite(origin_deg, "route origin")
    reports: list[dict[str, Any]] = []
    for pose in poses:
        goal = _six_finite(
            pose["mapped_hardware_joint_positions_deg"], "mapped hardware pose"
        )
        xyz: list[list[float]] = []
        for fraction in np.linspace(0.0, 1.0, samples):
            joints_rad = np.deg2rad(origin + (goal - origin) * float(fraction))
            end = fk.CalFK(joints_rad.tolist())[-1]
            xyz.append([float(end[0]), float(end[1]), float(end[2])])
        xyz_array = np.asarray(xyz, dtype=np.float64)
        minimum_z = float(np.min(xyz_array[:, 2]))
        if minimum_z < minimum_route_z_mm:
            raise RuntimeError(
                f"FK route for {pose['id']} drops below {minimum_route_z_mm:.1f}mm: "
                f"minimum z={minimum_z:.1f}mm"
            )
        reports.append(
            {
                "pose_id": pose["id"],
                "sample_count": samples,
                "minimum_z_mm": minimum_z,
                "goal_xyz_mm": xyz_array[-1].tolist(),
                "audit_scope": "end_effector_only_operator_must_clear_full_sweep",
            }
        )
    return reports


def load_and_validate_spec(
    spec_path: Path,
    calibration_path: Path,
    *,
    joint_limit_margin_deg: float,
    minimum_audited_route_z_mm: float,
) -> tuple[dict[str, Any], Path, dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    spec = load_json(spec_path)
    if spec.get("schema") != "assembly_new1_joint_mapping_validation_poses_v1":
        raise ValueError(f"unsupported pose-spec schema in {spec_path}")
    candidate_path = resolve_bundle_path(spec["candidate"])
    candidate = load_candidate(candidate_path)
    offset = _six_finite(
        candidate["hardware_minus_model_joint_offset_deg"], "joint offset"
    )
    limits = load_limits(calibration_path)
    poses = spec.get("poses")
    if not isinstance(poses, list) or len(poses) < 3:
        raise ValueError("multi-pose validation requires at least three poses")
    identifiers: set[str] = set()
    checked: list[dict[str, Any]] = []
    for raw in poses:
        if not isinstance(raw, dict):
            raise ValueError("each pose entry must be a JSON object")
        identifier = str(raw.get("id", ""))
        if not identifier or identifier in identifiers:
            raise ValueError(f"invalid or duplicate pose id: {identifier!r}")
        identifiers.add(identifier)
        references = raw.get("reference_images")
        if not isinstance(references, dict) or set(references) != set(CAMERA_SERIALS):
            raise ValueError(f"pose {identifier} must have front/wrist/side references")
        resolved_references = {
            name: resolve_bundle_path(references[name]) for name in CAMERA_SERIALS
        }
        for name, path in resolved_references.items():
            if not path.is_file():
                raise FileNotFoundError(f"missing {name} reference for {identifier}: {path}")
            with Image.open(path) as image:
                if image.size != (640, 480):
                    raise ValueError(
                        f"reference {path} has size {image.size}, expected 640x480"
                    )
        audited_z = float(raw.get("audited_route_min_z_mm", float("nan")))
        if not math.isfinite(audited_z) or audited_z < minimum_audited_route_z_mm:
            raise RuntimeError(
                f"pose {identifier} lacks the {minimum_audited_route_z_mm:.1f}mm "
                f"audited route clearance: {audited_z}"
            )
        goal = validate_pose_goal(
            raw["model_joint_positions_deg"],
            offset,
            limits,
            margin_deg=joint_limit_margin_deg,
        )
        checked.append(
            {
                **raw,
                "mapped_hardware_joint_positions_deg": goal.tolist(),
                "resolved_reference_images": {
                    name: str(path) for name, path in resolved_references.items()
                },
            }
        )
    return spec, candidate_path, candidate, offset, checked


def wrist_report_passes(
    report: dict[str, Any],
    *,
    translation_tolerance_px: float,
    rotation_tolerance_deg: float,
    scale_tolerance: float,
) -> bool:
    return bool(
        float(report["translation_norm_px"]) <= translation_tolerance_px
        and abs(float(report["rotation_deg"])) <= rotation_tolerance_deg
        and abs(float(report["scale"]) - 1.0) <= scale_tolerance
    )


def summarize_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    passed = [bool(item.get("passed")) for item in results]
    return {
        "passed": bool(len(results) >= 3 and all(passed)),
        "pose_count": len(results),
        "passed_pose_count": sum(passed),
        "failed_pose_ids": [
            str(item.get("pose_id")) for item in results if not bool(item.get("passed"))
        ],
    }


def build_approval_payload(
    *,
    candidate_path: Path,
    candidate: dict[str, Any],
    report_path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    summary = report.get("summary")
    if not isinstance(summary, dict) or not bool(summary.get("passed")):
        raise ValueError("validation report did not pass")
    if int(summary.get("pose_count", 0)) < 3:
        raise ValueError("validation report contains fewer than three poses")
    expected_candidate_sha = str(report.get("candidate_sha256", ""))
    actual_candidate_sha = sha256_file(candidate_path)
    if expected_candidate_sha != actual_candidate_sha:
        raise ValueError(
            "candidate changed after validation: "
            f"report={expected_candidate_sha}, current={actual_candidate_sha}"
        )
    results = report.get("pose_results")
    if not isinstance(results, list) or len(results) < 3 or not all(
        bool(item.get("passed")) for item in results
    ):
        raise ValueError("validation report pose results are incomplete or failed")
    offset = _six_finite(
        candidate["hardware_minus_model_joint_offset_deg"], "joint offset"
    )
    return {
        "schema": "assembly_new1_joint_zero_offset_v1",
        "status": "approved_active",
        "hardware_minus_model_joint_offset_deg": offset.tolist(),
        "multi_pose_validation": {
            "required": True,
            "passed": True,
            "pose_count": len(results),
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "candidate": str(candidate_path),
            "candidate_sha256": actual_candidate_sha,
            "approved_utc": datetime.now(timezone.utc).isoformat(),
            "pose_results": [
                {
                    "pose_id": str(item["pose_id"]),
                    "wrist_alignment": item["wrist_alignment"],
                    "passed": True,
                }
                for item in results
            ],
        },
    }


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("x", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self._stream.write(json.dumps(payload, allow_nan=False) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


class CameraRig:
    """Three persistent RGB streams; capture is sequential while pose is held."""

    def __init__(self, *, warmup_frames: int) -> None:
        import pyrealsense2 as rs

        self._pipelines: dict[str, Any] = {}
        try:
            for name, serial in CAMERA_SERIALS.items():
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                pipeline.start(config)
                self._pipelines[name] = pipeline
            for _ in range(warmup_frames):
                for name, pipeline in self._pipelines.items():
                    frame = pipeline.wait_for_frames(5000).get_color_frame()
                    if not frame:
                        raise RuntimeError(f"{name} camera returned no RGB warmup frame")
        except BaseException:
            self.close()
            raise

    def capture_all(
        self,
        *,
        sample_frames: int,
        hold_callback: Callable[[], None],
    ) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for name, pipeline in self._pipelines.items():
            frames: list[np.ndarray] = []
            for _ in range(sample_frames):
                frame = pipeline.wait_for_frames(5000).get_color_frame()
                if not frame:
                    raise RuntimeError(f"{name} camera returned no RGB sample frame")
                frames.append(np.asanyarray(frame.get_data()).copy())
                hold_callback()
            output[name] = np.median(np.stack(frames), axis=0).astype(np.uint8)
        return output

    def close(self) -> None:
        for pipeline in reversed(tuple(self._pipelines.values())):
            try:
                pipeline.stop()
            except BaseException:
                pass
        self._pipelines.clear()


def save_pair(reference: np.ndarray, current: np.ndarray, path: Path, title: str) -> None:
    height, width = reference.shape[:2]
    canvas = Image.new("RGB", (width * 2, height + 36), "white")
    canvas.paste(Image.fromarray(reference), (0, 36))
    canvas.paste(Image.fromarray(current), (width, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), f"TRAINING {title}", fill="black")
    draw.text((width + 6, 8), f"CURRENT {title} (human audit)", fill="black")
    canvas.save(path, quality=95)


def _alignment_module() -> Any:
    sys.path.insert(0, str(BUNDLE_ROOT / "tools"))
    import check_wrist_visual_alignment

    return check_wrist_visual_alignment


def _wrist_calibration_module() -> Any:
    sys.path.insert(0, str(BUNDLE_ROOT / "robot"))
    import wrist_visual_zero_calibration

    return wrist_visual_zero_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "execute", "approve"), default="check")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--initial-pose-json", type=Path, default=DEFAULT_INITIAL_POSE)
    parser.add_argument("--active-offset-json", type=Path, default=DEFAULT_ACTIVE_OFFSET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--approval-token")
    parser.add_argument("--can-port")
    parser.add_argument("--translation-tolerance-px", type=float, default=5.0)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=0.75)
    parser.add_argument("--scale-tolerance", type=float, default=0.02)
    parser.add_argument("--joint-limit-margin-deg", type=float, default=2.0)
    parser.add_argument("--minimum-audited-route-z-mm", type=float, default=200.0)
    parser.add_argument("--start-pose-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--minimum-open-gripper-mm", type=float, default=90.0)
    parser.add_argument("--move-velocity-deg-s", type=float, default=3.0)
    parser.add_argument("--move-acceleration-deg-s2", type=float, default=6.0)
    parser.add_argument("--move-tolerance-deg", type=float, default=0.3)
    parser.add_argument("--move-timeout-s", type=float, default=60.0)
    parser.add_argument("--settle-s", type=float, default=0.75)
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--max-relative-target", type=float, default=0.25)
    parser.add_argument("--max-gripper-relative-target", type=float, default=100.0)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--sample-frames", type=int, default=9)
    parser.add_argument("--allow-motion", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.mode == "execute":
        if not args.allow_motion:
            parser.error("execute mode requires --allow-motion")
        if not args.can_port:
            parser.error("execute mode requires --can-port")
    elif args.allow_motion:
        parser.error("--allow-motion is valid only in execute mode")
    if args.mode == "approve":
        if args.validation_report is None:
            parser.error("approve mode requires --validation-report")
        if args.approval_token != APPROVAL_TOKEN:
            parser.error(f"approve mode requires --approval-token {APPROVAL_TOKEN}")
    elif args.approval_token is not None:
        parser.error("--approval-token is valid only in approve mode")
    for path in (args.spec, args.calibration_json, args.initial_pose_json):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.mode == "approve" and not args.validation_report.is_file():
        parser.error(f"validation report does not exist: {args.validation_report}")
    if not 1 <= args.speed_percent <= 10:
        parser.error("this commissioning tool limits --speed-percent to [1,10]")
    if args.max_relative_target <= 0 or args.max_relative_target > 0.5:
        parser.error("--max-relative-target must be in (0,0.5]")
    for name in (
        "translation_tolerance_px",
        "rotation_tolerance_deg",
        "scale_tolerance",
        "joint_limit_margin_deg",
        "minimum_audited_route_z_mm",
        "start_pose_tolerance_deg",
        "minimum_open_gripper_mm",
        "move_velocity_deg_s",
        "move_acceleration_deg_s2",
        "move_tolerance_deg",
        "move_timeout_s",
        "settle_s",
        "max_gripper_relative_target",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    spec, candidate_path, candidate, offset, poses = load_and_validate_spec(
        args.spec,
        args.calibration_json,
        joint_limit_margin_deg=args.joint_limit_margin_deg,
        minimum_audited_route_z_mm=args.minimum_audited_route_z_mm,
    )
    report = {
        "event": "joint_mapping_static_check",
        "schema": spec["schema"],
        "candidate": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "hardware_minus_model_joint_offset_deg": offset.tolist(),
        "pose_count": len(poses),
        "poses": [
            {
                "pose_id": pose["id"],
                "frame_index": pose["frame_index"],
                "mapped_hardware_joint_positions_deg": pose[
                    "mapped_hardware_joint_positions_deg"
                ],
                "audited_route_min_z_mm": pose["audited_route_min_z_mm"],
            }
            for pose in poses
        ],
        "checkpoint_loaded": False,
        "policy_constructed": False,
        "camera_opened": False,
        "can_opened": False,
        "motion_command_sent": False,
        "active_mapping_modified": False,
    }
    print(json.dumps(report, indent=2))
    print("CHECK PASSED: static only; no camera, CAN, model, or motion was opened.")
    return report


def run_approve(args: argparse.Namespace) -> dict[str, Any]:
    _, candidate_path, candidate, _, _ = load_and_validate_spec(
        args.spec,
        args.calibration_json,
        joint_limit_margin_deg=args.joint_limit_margin_deg,
        minimum_audited_route_z_mm=args.minimum_audited_route_z_mm,
    )
    report = load_json(args.validation_report)
    payload = build_approval_payload(
        candidate_path=candidate_path,
        candidate=candidate,
        report_path=args.validation_report,
        report=report,
    )
    try:
        relative_active = str(
            args.active_offset_json.resolve().relative_to(BUNDLE_ROOT.resolve())
        )
    except ValueError as error:
        raise ValueError("approved active mapping must remain inside the bundle") from error
    manifest = load_json(DEFAULT_MANIFEST)
    if relative_active not in manifest.get("immutable_runtime_files", {}):
        raise ValueError(
            f"active mapping {relative_active} is absent from manifest immutable files"
        )
    backup = args.active_offset_json.with_name(
        f"{args.active_offset_json.stem}.before_approval_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    if args.active_offset_json.exists():
        backup.write_bytes(args.active_offset_json.read_bytes())
    manifest_backup = DEFAULT_MANIFEST.with_name(
        f"{DEFAULT_MANIFEST.stem}.before_mapping_approval_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    manifest_backup.write_bytes(DEFAULT_MANIFEST.read_bytes())
    atomic_write_json(args.active_offset_json, payload)
    manifest["immutable_runtime_files"][relative_active] = sha256_file(
        args.active_offset_json
    )
    atomic_write_json(DEFAULT_MANIFEST, manifest)
    result = {
        "event": "joint_mapping_approved",
        "active_offset_json": str(args.active_offset_json),
        "active_offset_sha256": sha256_file(args.active_offset_json),
        "backup": str(backup) if backup.exists() else None,
        "manifest_backup": str(manifest_backup),
        "pose_count": payload["multi_pose_validation"]["pose_count"],
        "motion_command_sent": False,
    }
    print(json.dumps(result, indent=2))
    return result


def run_execute(args: argparse.Namespace) -> dict[str, Any]:
    spec, candidate_path, candidate, offset, poses = load_and_validate_spec(
        args.spec,
        args.calibration_json,
        joint_limit_margin_deg=args.joint_limit_margin_deg,
        minimum_audited_route_z_mm=args.minimum_audited_route_z_mm,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / f"execute_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log = JsonlLogger(run_dir / "events.jsonl")
    visual = _alignment_module()
    wrist = _wrist_calibration_module()
    home = wrist._home_module()
    joint_limits, gripper_stroke_mm = home.load_calibration(args.calibration_json)
    saved_home = home.load_initial_pose(
        args.initial_pose_json, joint_limits, gripper_stroke_mm
    )
    saved_home_deg = np.asarray(saved_home.joints_deg, dtype=np.float64)
    fk_route_reports = verify_fk_routes(
        origin_deg=saved_home_deg,
        poses=poses,
        minimum_route_z_mm=args.minimum_audited_route_z_mm,
    )
    sdk: Any | None = None
    cameras: CameraRig | None = None
    motion_enabled = False
    current_goal = saved_home_deg.copy()

    def hold_current() -> None:
        if sdk is None or not motion_enabled:
            return
        measured = home.raw_feedback_pose(sdk.get_status())
        pose = home.PhysicalPose(tuple(float(v) for v in current_goal), measured.gripper_mm)
        home.bounded_send(
            sdk,
            home.physical_pose_to_normalized(pose, joint_limits, gripper_stroke_mm),
            max_arm_relative_target=args.max_relative_target,
            max_gripper_relative_target=args.max_gripper_relative_target,
        )

    try:
        sdk = home.PiperSDKInterface(
            port=args.can_port,
            gripper_stroke_mm=gripper_stroke_mm,
            speed_percent=args.speed_percent,
            joint_limits_deg=joint_limits,
            enable_motion=False,
        )
        measured = home.raw_feedback_pose(sdk.get_status())
        measured_deg = np.asarray(measured.joints_deg, dtype=np.float64)
        start_error = float(np.max(np.abs(measured_deg - saved_home_deg)))
        motors = home.verify_health(sdk, require_enabled=False, check_controller=False)
        status = sdk.piper.GetArmStatus().arm_status
        preflight = {
            "event": "joint_mapping_execute_preflight",
            "current_joints_deg": measured_deg.tolist(),
            "saved_unoffset_home_joints_deg": saved_home_deg.tolist(),
            "max_start_home_error_deg": start_error,
            "current_gripper_mm": float(measured.gripper_mm),
            "motor_enabled": [bool(motor["driver_enabled"]) for motor in motors],
            "motor_voltage_v": [float(motor["voltage_v"]) for motor in motors],
            "controller_status": str(status),
            "candidate_sha256": sha256_file(candidate_path),
            "offline_fk_route_audit": fk_route_reports,
            "command_sent": False,
        }
        log.write(preflight)
        print(json.dumps(preflight, indent=2))
        if start_error > args.start_pose_tolerance_deg:
            raise RuntimeError(
                f"validation must start at saved unoffset Home: max error "
                f"{start_error:.3f} > {args.start_pose_tolerance_deg:.3f} deg"
            )
        if measured.gripper_mm < args.minimum_open_gripper_mm:
            raise RuntimeError(
                f"gripper must be open before validation: {measured.gripper_mm:.1f}mm "
                f"< {args.minimum_open_gripper_mm:.1f}mm"
            )
        if status.err_code or status.arm_status or status.ctrl_mode not in (0, 1):
            raise RuntimeError(f"controller is not NORMAL STANDBY/CAN: {status}")
        enabled = [bool(motor["driver_enabled"]) for motor in motors]
        if all(enabled):
            home.adopt_already_enabled_motion_hold_current(sdk)
        elif not any(enabled):
            home.enable_disabled_motion_hold_current(sdk)
        else:
            raise RuntimeError(f"partially enabled arm is not allowed: {enabled}")
        motion_enabled = True
        cameras = CameraRig(warmup_frames=args.warmup_frames)

        results: list[dict[str, Any]] = []
        for index, pose in enumerate(poses):
            goal = np.asarray(pose["mapped_hardware_joint_positions_deg"], dtype=np.float64)
            print(f"[{index + 1}/{len(poses)}] moving to {pose['id']} at low speed")
            wrist._move_to_joint_pose(
                sdk,
                goal_deg=goal,
                joint_limits=joint_limits,
                gripper_stroke_mm=gripper_stroke_mm,
                args=args,
                home=home,
                log=log,
                label=f"pose_{pose['id']}",
            )
            current_goal = goal.copy()
            captured = cameras.capture_all(
                sample_frames=args.sample_frames, hold_callback=hold_current
            )
            references = {
                name: np.asarray(
                    Image.open(path).convert("RGB")
                )
                for name, path in pose["resolved_reference_images"].items()
            }
            for name, image in captured.items():
                Image.fromarray(image).save(run_dir / f"{pose['id']}_{name}_current.png")
            try:
                wrist_report = visual.estimate_alignment(
                    references["wrist"], captured["wrist"]
                )
                wrist_report["robot_or_can_opened"] = True
                wrist_report["motion_command_sent"] = True
                passed = wrist_report_passes(
                    wrist_report,
                    translation_tolerance_px=args.translation_tolerance_px,
                    rotation_tolerance_deg=args.rotation_tolerance_deg,
                    scale_tolerance=args.scale_tolerance,
                )
                visual.save_montage(
                    references["wrist"],
                    captured["wrist"],
                    wrist_report,
                    run_dir / f"{pose['id']}_wrist_alignment.jpg",
                )
            except RuntimeError as alignment_error:
                # A low-texture or scene-mismatched image is a failed
                # validation sample, not a controller emergency. Record it,
                # preserve the images, return Home, and finish the complete
                # report without turning the failure into an unsafe retry.
                wrist_report = {
                    "alignment_error": str(alignment_error),
                    "robot_or_can_opened": True,
                    "motion_command_sent": True,
                }
                passed = False
                save_pair(
                    references["wrist"],
                    captured["wrist"],
                    run_dir / f"{pose['id']}_wrist_alignment_failed.jpg",
                    "wrist alignment failed",
                )
            for name in ("front", "side"):
                save_pair(
                    references[name],
                    captured[name],
                    run_dir / f"{pose['id']}_{name}_human_audit.jpg",
                    name,
                )
            result = {
                "pose_id": pose["id"],
                "frame_index": pose["frame_index"],
                "model_joint_positions_deg": pose["model_joint_positions_deg"],
                "mapped_hardware_joint_positions_deg": goal.tolist(),
                "wrist_alignment": wrist_report,
                "front_and_side_saved_for_human_audit": True,
                "passed": passed,
            }
            results.append(result)
            log.write({"event": "joint_mapping_pose_result", **result})
            print(json.dumps(result, indent=2))
            print(f"[{index + 1}/{len(poses)}] returning to saved Home")
            wrist._move_to_joint_pose(
                sdk,
                goal_deg=saved_home_deg,
                joint_limits=joint_limits,
                gripper_stroke_mm=gripper_stroke_mm,
                args=args,
                home=home,
                log=log,
                label=f"return_home_after_{pose['id']}",
            )
            current_goal = saved_home_deg.copy()

        summary = summarize_results(results)
        report = {
            "schema": "assembly_new1_joint_mapping_multi_pose_report_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": str(candidate_path),
            "candidate_sha256": sha256_file(candidate_path),
            "hardware_minus_model_joint_offset_deg": offset.tolist(),
            "thresholds": {
                "translation_tolerance_px": args.translation_tolerance_px,
                "rotation_tolerance_deg": args.rotation_tolerance_deg,
                "scale_tolerance": args.scale_tolerance,
            },
            "offline_fk_route_audit": fk_route_reports,
            "pose_results": results,
            "summary": summary,
            "active_mapping_modified": False,
        }
        report_path = run_dir / "validation_report.json"
        atomic_write_json(report_path, report)
        if summary["passed"]:
            pending = {
                **build_approval_payload(
                    candidate_path=candidate_path,
                    candidate=candidate,
                    report_path=report_path,
                    report=report,
                ),
                "status": "validated_pending_approval",
            }
            pending_path = run_dir / "validated_joint_zero_offset.json"
            atomic_write_json(pending_path, pending)
            message = (
                "VALIDATION PASSED: active mapping was NOT changed. Review all "
                f"montages, then use separate approve mode. report={report_path}"
            )
        else:
            message = (
                "VALIDATION FAILED: constant joint offset is not accepted; active "
                f"mapping remains unchanged. report={report_path}"
            )
        home.latch_current_pose(sdk)
        log.write({"event": "joint_mapping_validation_complete", **summary})
        print(json.dumps(report, indent=2))
        print(message)
        return report
    except BaseException as error:
        log.write(
            {
                "event": "joint_mapping_validation_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "command_sent": motion_enabled,
            }
        )
        if sdk is not None and motion_enabled:
            try:
                wrist._move_to_joint_pose(
                    sdk,
                    goal_deg=saved_home_deg,
                    joint_limits=joint_limits,
                    gripper_stroke_mm=gripper_stroke_mm,
                    args=args,
                    home=home,
                    log=log,
                    label="failure_return_to_saved_home",
                )
                current_goal = saved_home_deg.copy()
            except BaseException as return_error:
                log.write(
                    {
                        "event": "failure_return_failed",
                        "error": repr(return_error),
                        "command_sent": True,
                    }
                )
            try:
                home.latch_current_pose(sdk)
            except BaseException as hold_error:
                log.write(
                    {
                        "event": "current_pose_latch_failed",
                        "error": repr(hold_error),
                        "command_sent": False,
                    }
                )
        raise
    finally:
        if cameras is not None:
            cameras.close()
        if sdk is not None:
            sdk.disconnect(home_first=False, disable_torque=False)
        log.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if args.mode == "check":
        run_check(args)
    elif args.mode == "approve":
        run_approve(args)
    else:
        run_execute(args)


if __name__ == "__main__":
    main()
