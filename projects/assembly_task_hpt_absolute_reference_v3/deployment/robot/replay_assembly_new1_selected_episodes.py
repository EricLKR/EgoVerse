#!/usr/bin/env python3
"""Guarded batch replay for selected one-based Assembly New1 entries.

The default sequence is entry 1, 10, 20, ..., 100. Dataset entry numbers are
one-based for the operator, while LeRobot ``episode_index`` is zero-based.
Every trajectory is validated offline before execute mode can send motion.
Each episode is then delegated to the audited single-episode replay program.

Execute mode pauses before every episode so the operator can restore the
workpieces, confirm the sweep area is clear, and keep the emergency stop in
reach. A failed episode stops the batch immediately.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import replay_assembly_new1_episode as replay


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_DIR = BUNDLE_ROOT / "replay/assembly_new1_selected_episodes"
DEFAULT_LOG_DIR = BUNDLE_ROOT / "logs/episode_replay_batch"
DEFAULT_ENTRIES = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def trajectory_path(directory: Path, entry_number: int) -> Path:
    episode_index = int(entry_number) - 1
    return directory / (
        f"assembly_new1_entry_{entry_number:04d}_"
        f"episode_index_{episode_index:04d}.npz"
    )


def validate_entries(entries: list[int]) -> list[int]:
    if not entries:
        raise ValueError("at least one one-based dataset entry is required")
    normalized = [int(value) for value in entries]
    if any(value < 1 or value > 100 for value in normalized):
        raise ValueError("dataset entry numbers must be within [1,100]")
    if len(set(normalized)) != len(normalized):
        raise ValueError("dataset entry numbers must be unique")
    return normalized


def offline_preflight(args: argparse.Namespace) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    calibration = json.loads(args.calibration_json.read_text(encoding="utf-8"))
    hardware_low = np.asarray(
        [calibration[f"joint_{index}"]["range_min"] / 1000.0 for index in range(6)],
        dtype=np.float64,
    )
    hardware_high = np.asarray(
        [calibration[f"joint_{index}"]["range_max"] / 1000.0 for index in range(6)],
        dtype=np.float64,
    )
    for entry in args.entries:
        path = trajectory_path(args.trajectory_dir, entry)
        if not path.is_file():
            raise FileNotFoundError(f"missing replay trajectory: {path}")
        trajectory = replay.load_trajectory(path, args.calibration_json)
        expected_index = entry - 1
        actual_index = int(trajectory.metadata["episode_index"])
        if actual_index != expected_index:
            raise RuntimeError(
                f"entry {entry} expected episode_index={expected_index}, "
                f"trajectory contains {actual_index}"
            )
        joints, gripper, frames = replay.select_frames(
            trajectory,
            0,
            None,
            gripper_command_mode=args.gripper_command_mode,
        )
        resample_gripper_step = (
            100.0
            if args.gripper_command_mode == "binary-action"
            else args.max_gripper_step_percent
        )
        replay_joints, replay_gripper, _ = replay.resample_bounded(
            joints,
            gripper,
            frames,
            time_scale=args.time_scale,
            max_joint_step_deg=args.max_joint_step_deg,
            max_gripper_step_percent=resample_gripper_step,
        )
        firmware_command_joints = replay.replay_actions_to_firmware_commands(
            replay_joints,
            args.j2_command_offset_deg,
            args.j3_command_offset_deg,
            args.j5_command_offset_deg,
        )
        return_error = np.abs(joints[-1] - joints[0])
        limit_overshoot = np.maximum(
            np.maximum(hardware_low[None, :] - firmware_command_joints, 0.0),
            np.maximum(firmware_command_joints - hardware_high[None, :], 0.0),
        )
        max_limit_overshoot = float(limit_overshoot.max(initial=0.0))
        if max_limit_overshoot > args.max_trajectory_joint_limit_overshoot_deg:
            raise RuntimeError(
                f"entry {entry} exceeds the configured hardware calibration "
                f"by {max_limit_overshoot:.3f} deg; limit is "
                f"{args.max_trajectory_joint_limit_overshoot_deg:.3f} deg"
            )
        reports.append(
            {
                "entry_number": entry,
                "episode_index": actual_index,
                "trajectory": str(path),
                "source_frames": int(len(joints)),
                "replay_ticks": int(len(replay_joints)),
                "estimated_duration_s": float(len(replay_joints) / replay.CONTROL_HZ),
                "first_joints_deg": joints[0].tolist(),
                "last_joints_deg": joints[-1].tolist(),
                "first_firmware_command_joints_deg": firmware_command_joints[0].tolist(),
                "last_firmware_command_joints_deg": firmware_command_joints[-1].tolist(),
                "j2_feedback_offset_deg": args.j2_feedback_offset_deg,
                "j2_action_offset_deg": args.j2_command_offset_deg,
                "j3_action_offset_deg": args.j3_command_offset_deg,
                "j5_action_offset_deg": args.j5_command_offset_deg,
                "start_to_end_max_joint_delta_deg": float(return_error.max()),
                "first_gripper_percent": float(gripper[0]),
                "last_gripper_percent": float(gripper[-1]),
                "gripper_command_mode": args.gripper_command_mode,
                "gripper_binary_threshold_percent": (
                    args.gripper_binary_threshold_percent
                ),
                "gripper_binary_switch_count": int(
                    np.count_nonzero(
                        (gripper[1:] >= args.gripper_binary_threshold_percent)
                        != (gripper[:-1] >= args.gripper_binary_threshold_percent)
                    )
                ),
                "hardware_calibration_limit_overshoot_count": int(
                    np.count_nonzero(limit_overshoot)
                ),
                "hardware_calibration_max_limit_overshoot_deg": (
                    max_limit_overshoot
                ),
                "hardware_calibration_max_limit_overshoot_per_axis_deg": (
                    limit_overshoot.max(axis=0, initial=0.0).tolist()
                ),
                "replay_max_joint_step_deg": (
                    float(np.max(np.abs(np.diff(replay_joints, axis=0))))
                    if len(replay_joints) > 1
                    else 0.0
                ),
                "replay_max_gripper_step_percent": (
                    float(np.max(np.abs(np.diff(replay_gripper))))
                    if len(replay_gripper) > 1
                    else 0.0
                ),
            }
        )
    for previous, current in zip(reports, reports[1:], strict=False):
        previous_last = np.asarray(previous["last_joints_deg"], dtype=np.float64)
        current_first = np.asarray(current["first_joints_deg"], dtype=np.float64)
        current["error_from_previous_end_to_this_start_deg"] = np.abs(
            previous_last - current_first
        ).tolist()
        current["max_error_from_previous_end_to_this_start_deg"] = float(
            np.max(np.abs(previous_last - current_first))
        )
    return reports


def child_command(
    args: argparse.Namespace,
    *,
    entry: int,
    mode: str,
) -> list[str]:
    command = [
        sys.executable,
        str(BUNDLE_ROOT / "robot/replay_assembly_new1_episode.py"),
        "--mode",
        mode,
        "--trajectory",
        str(trajectory_path(args.trajectory_dir, entry)),
        "--calibration-json",
        str(args.calibration_json),
        "--model-calibration-json",
        str(args.model_calibration_json),
        "--role-mapping-json",
        str(args.role_mapping_json),
        "--joint-mapping-mode",
        args.joint_mapping_mode,
        "--time-scale",
        str(args.time_scale),
        "--max-joint-step-deg",
        str(args.max_joint_step_deg),
        "--max-gripper-step-percent",
        str(args.max_gripper_step_percent),
        "--max-start-joint-error-deg",
        str(args.max_start_joint_error_deg),
        "--max-trajectory-joint-limit-overshoot-deg",
        str(args.max_trajectory_joint_limit_overshoot_deg),
        "--max-tracking-error-deg",
        str(args.max_tracking_error_deg),
        "--max-tracking-error-cycles",
        str(args.max_tracking_error_cycles),
        "--j2-feedback-offset-deg",
        str(args.j2_feedback_offset_deg),
        "--j2-command-offset-deg",
        str(args.j2_command_offset_deg),
        "--j3-command-offset-deg",
        str(args.j3_command_offset_deg),
        "--j5-command-offset-deg",
        str(args.j5_command_offset_deg),
        "--speed-percent",
        str(args.speed_percent),
        "--max-relative-target",
        str(args.max_relative_target),
        "--max-gripper-relative-target",
        str(args.max_gripper_relative_target),
        "--gripper-effort",
        str(args.gripper_effort),
        "--gripper-command-mode",
        args.gripper_command_mode,
        "--gripper-binary-threshold-percent",
        str(args.gripper_binary_threshold_percent),
    ]
    if mode != "check":
        command.extend(("--can-port", args.can_port))
    if mode == "execute":
        command.append("--allow-motion")
    return command


def append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, allow_nan=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("check", "live-dry-run", "execute"), default="check"
    )
    parser.add_argument(
        "--entries", type=int, nargs="+", default=list(DEFAULT_ENTRIES)
    )
    parser.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--calibration-json", type=Path, default=replay.DEFAULT_CALIBRATION)
    parser.add_argument(
        "--model-calibration-json",
        type=Path,
        default=replay.DEFAULT_MODEL_CALIBRATION,
    )
    parser.add_argument(
        "--role-mapping-json", type=Path, default=replay.DEFAULT_ROLE_MAPPING
    )
    parser.add_argument(
        "--joint-mapping-mode",
        choices=("physical-identity", "recorded-percentile-affine"),
        default="physical-identity",
    )
    parser.add_argument("--can-port")
    parser.add_argument("--time-scale", type=float, default=4.0)
    parser.add_argument("--max-joint-step-deg", type=float, default=0.25)
    parser.add_argument("--max-gripper-step-percent", type=float, default=5.0)
    parser.add_argument("--max-start-joint-error-deg", type=float, default=1.0)
    parser.add_argument(
        "--max-trajectory-joint-limit-overshoot-deg", type=float, default=2.5
    )
    parser.add_argument("--max-tracking-error-deg", type=float, default=8.0)
    parser.add_argument("--max-tracking-error-cycles", type=int, default=15)
    parser.add_argument(
        "--j2-feedback-offset-deg",
        type=float,
        default=0.0,
        help=(
            "value added to raw firmware J2 feedback; replay action commands are unchanged"
        ),
    )
    parser.add_argument(
        "--j2-command-offset-deg",
        type=float,
        default=0.0,
        help="value added only to replay J2 commands; raw feedback is unchanged",
    )
    parser.add_argument(
        "--j3-command-offset-deg",
        type=float,
        default=0.0,
        help=(
            "value added only to replay J3 commands; raw feedback is "
            "inverse-mapped for replay comparisons"
        ),
    )
    parser.add_argument(
        "--j5-command-offset-deg",
        type=float,
        default=0.0,
        help=(
            "value added only to replay J5 commands; raw feedback is "
            "inverse-mapped for replay comparisons"
        ),
    )
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--max-relative-target", type=float, default=0.5)
    parser.add_argument("--max-gripper-relative-target", type=float, default=100.0)
    parser.add_argument(
        "--gripper-command-mode",
        choices=("recorded-aperture", "binary-action"),
        default="binary-action",
        help="batch default emits only fully open/fully closed gripper commands",
    )
    parser.add_argument(
        "--gripper-binary-threshold-percent", type=float, default=50.0
    )
    parser.add_argument(
        "--gripper-effort",
        type=int,
        default=5000,
        help="fixed 5 N*m maximum PIPER gripper setting (5000 SDK units)",
    )
    parser.add_argument(
        "--pause-before-each",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require Enter before each episode so objects can be reset",
    )
    parser.add_argument("--allow-motion", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.entries = validate_entries(args.entries)
    except ValueError as error:
        parser.error(str(error))
    for path_name in (
        "calibration_json",
        "model_calibration_json",
        "role_mapping_json",
    ):
        if not getattr(args, path_name).is_file():
            parser.error(f"missing --{path_name.replace('_', '-')}: {getattr(args, path_name)}")
    if args.mode != "check" and not args.can_port:
        parser.error(f"{args.mode} requires --can-port")
    if args.mode == "execute" and not args.allow_motion:
        parser.error("execute mode requires --allow-motion")
    if args.mode != "execute" and args.allow_motion:
        parser.error("--allow-motion is valid only in execute mode")
    if args.mode == "execute" and args.pause_before_each and not sys.stdin.isatty():
        parser.error("execute with pauses requires an interactive terminal")
    for name in (
        "time_scale",
        "max_joint_step_deg",
        "max_gripper_step_percent",
        "max_start_joint_error_deg",
        "max_trajectory_joint_limit_overshoot_deg",
        "max_tracking_error_deg",
        "max_relative_target",
        "max_gripper_relative_target",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not 1 <= args.speed_percent <= 100:
        parser.error("--speed-percent must be in [1,100]")
    if not 0 <= args.gripper_effort <= 5000:
        parser.error("--gripper-effort must be in [0,5000]")
    if not 0.0 <= args.gripper_binary_threshold_percent <= 100.0:
        parser.error("--gripper-binary-threshold-percent must be in [0,100]")
    if args.max_tracking_error_cycles < 1:
        parser.error("--max-tracking-error-cycles must be positive")
    if (
        not math.isfinite(args.j2_feedback_offset_deg)
        or abs(args.j2_feedback_offset_deg) > 15.0
    ):
        parser.error("--j2-feedback-offset-deg must be finite and within [-15,15]")
    if (
        not math.isfinite(args.j2_command_offset_deg)
        or abs(args.j2_command_offset_deg) > 15.0
    ):
        parser.error("--j2-command-offset-deg must be finite and within [-15,15]")
    if (
        not math.isfinite(args.j3_command_offset_deg)
        or abs(args.j3_command_offset_deg) > 15.0
    ):
        parser.error("--j3-command-offset-deg must be finite and within [-15,15]")
    if (
        not math.isfinite(args.j5_command_offset_deg)
        or abs(args.j5_command_offset_deg) > 15.0
    ):
        parser.error("--j5-command-offset-deg must be finite and within [-15,15]")
    if args.j2_feedback_offset_deg != 0.0 and args.j2_command_offset_deg != 0.0:
        parser.error(
            "do not combine J2 feedback and command offsets in one replay; "
            "this experiment requires raw feedback"
        )
    return args


def main() -> None:
    args = parse_args()
    preflight = offline_preflight(args)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.log_dir / f"selected_episode_batch_{args.mode}_{timestamp}.jsonl"
    summary = {
        "event": "batch_preflight",
        "mode": args.mode,
        "entries": args.entries,
        "episode_indices": [entry - 1 for entry in args.entries],
        "trajectories": preflight,
        "model_or_checkpoint_loaded": False,
        "camera_opened": False,
        "motion_command_sent": False,
    }
    append_log(log_path, summary)
    print(json.dumps(summary, indent=2))
    if args.mode == "check":
        print(f"BATCH CHECK PASSED: offline only; no CAN opened. log={log_path}")
        return

    for sequence_index, entry in enumerate(args.entries, start=1):
        if args.mode == "execute" and args.pause_before_each:
            print(
                "\n"
                f"Ready for dataset entry {entry} "
                f"(episode_index={entry - 1}, {sequence_index}/{len(args.entries)}).\n"
                "Reset all workpieces to the recorded initial layout, confirm "
                "the arm is at this episode's start pose, clear the sweep area, "
                "and keep the emergency stop accessible."
            )
            response = input("Press Enter to replay, or type q then Enter to stop: ")
            if response.strip().lower() in {"q", "quit", "stop"}:
                append_log(
                    log_path,
                    {
                        "event": "batch_stopped_by_operator",
                        "before_entry": entry,
                        "motion_command_sent": False,
                    },
                )
                print(f"BATCH STOPPED before entry {entry}. log={log_path}")
                return
        command = child_command(args, entry=entry, mode=args.mode)
        append_log(
            log_path,
            {
                "event": "episode_start",
                "entry_number": entry,
                "episode_index": entry - 1,
                "command": command,
                "motion_command_sent": args.mode == "execute",
            },
        )
        completed = subprocess.run(command, check=False)
        append_log(
            log_path,
            {
                "event": "episode_exit",
                "entry_number": entry,
                "episode_index": entry - 1,
                "returncode": int(completed.returncode),
                "motion_command_sent": args.mode == "execute",
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"entry {entry} replay failed with return code "
                f"{completed.returncode}; batch stopped"
            )
    append_log(
        log_path,
        {
            "event": "batch_complete",
            "entries": args.entries,
            "motion_command_sent": args.mode == "execute",
        },
    )
    print(f"BATCH COMPLETE: {len(args.entries)} episodes. log={log_path}")


if __name__ == "__main__":
    main()
