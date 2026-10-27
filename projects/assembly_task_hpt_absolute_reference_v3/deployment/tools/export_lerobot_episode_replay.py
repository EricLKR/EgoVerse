#!/usr/bin/env python3
"""Export one LeRobot episode into an audited, model-free replay trajectory.

The dataset contains both the teleoperator ``action`` and the follower's raw
``observation.state`` feedback.  Exact physical replay must use the latter:
the action percentiles are driver-normalised commands and re-decoding them
through another arm's calibration changes the physical end-effector pose.

The action is still retained for auditing and is decoded through the recorded
follower calibration to document the checkpoint's target space.  The replay
trajectory itself uses the measured six-axis follower angles and measured
gripper aperture.

This tool never imports a robot driver and never accesses CAN or cameras.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLLOWER_CALIBRATION = BUNDLE_ROOT / "assets/piper_follower_calibration.json"
DEFAULT_LEADER_CALIBRATION = BUNDLE_ROOT / "assets/piper_original_leader_calibration.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_limits(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    low = np.asarray(
        [payload[f"joint_{index}"]["range_min"] / 1000.0 for index in range(6)],
        dtype=np.float64,
    )
    high = np.asarray(
        [payload[f"joint_{index}"]["range_max"] / 1000.0 for index in range(6)],
        dtype=np.float64,
    )
    stroke = float(payload["gripper"]["range_max"]) / 1000.0
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)) or np.any(high <= low):
        raise ValueError(f"invalid joint calibration: {path}")
    if not np.isfinite(stroke) or stroke <= 0.0:
        raise ValueError(f"invalid gripper calibration: {path}")
    return low, high, stroke


def decode_percent(action_percent: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    values = np.asarray(action_percent, dtype=np.float64)
    return low[None, :] + (values + 100.0) / 200.0 * (high - low)[None, :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--follower-calibration", type=Path, default=DEFAULT_FOLLOWER_CALIBRATION
    )
    parser.add_argument(
        "--leader-calibration", type=Path, default=DEFAULT_LEADER_CALIBRATION
    )
    args = parser.parse_args()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required only for the offline export step") from exc

    files = sorted((args.dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no LeRobot parquet files under {args.dataset_root}")
    table = pa.concat_tables(
        [
            pq.read_table(
                path,
                columns=[
                    "episode_index",
                    "frame_index",
                    "timestamp",
                    "action",
                    "observation.state",
                ],
            )
            for path in files
        ]
    )
    episode = np.asarray(table["episode_index"], dtype=np.int64)
    mask = episode == args.episode_index
    if not np.any(mask):
        available = sorted(np.unique(episode).tolist())
        raise ValueError(
            f"episode {args.episode_index} is absent; available range={available[:3]}...{available[-3:]}"
        )
    frame = np.asarray(table["frame_index"], dtype=np.int64)[mask]
    timestamp = np.asarray(table["timestamp"], dtype=np.float64)[mask]
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)[mask]
    state = np.asarray(
        table["observation.state"].to_pylist(), dtype=np.float64
    )[mask]
    order = np.argsort(frame)
    frame = frame[order]
    timestamp = timestamp[order]
    action = action[order]
    state = state[order]
    if action.ndim != 2 or action.shape[1] != 7 or not np.isfinite(action).all():
        raise ValueError(f"invalid action array: {action.shape}")
    if state.ndim != 2 or state.shape[1] != 7 or not np.isfinite(state).all():
        raise ValueError(f"invalid observation.state array: {state.shape}")
    joint_action_overshoot = np.maximum(
        np.maximum(-100.0 - action[:, :6], 0.0),
        np.maximum(action[:, :6] - 100.0, 0.0),
    )
    gripper_action_overshoot = np.maximum(
        np.maximum(-action[:, 6], 0.0),
        np.maximum(action[:, 6] - 100.0, 0.0),
    )
    # These actions are retained only for model-target auditing; replay uses
    # measured follower observation.state. Admit tiny float/quantisation
    # overshoot, record it, and clip the audit copy exactly to its schema.
    if float(joint_action_overshoot.max(initial=0.0)) > 0.05:
        raise ValueError("joint action percentiles materially exceed [-100,100]")
    if float(gripper_action_overshoot.max(initial=0.0)) > 0.05:
        raise ValueError("gripper action percentiles materially exceed [0,100]")
    action_was_clipped = bool(
        np.any(joint_action_overshoot > 0.0)
        or np.any(gripper_action_overshoot > 0.0)
    )
    action = action.copy()
    action[:, :6] = np.clip(action[:, :6], -100.0, 100.0)
    action[:, 6] = np.clip(action[:, 6], 0.0, 100.0)
    if not np.array_equal(frame, np.arange(len(frame), dtype=np.int64)):
        raise ValueError("episode frame indices are not contiguous from zero")

    follower_low, follower_high, follower_stroke = load_limits(args.follower_calibration)
    leader_low, leader_high, leader_stroke = load_limits(args.leader_calibration)
    follower_deg = decode_percent(action[:, :6], follower_low, follower_high)
    recorded_follower_deg = state[:, :6] / 1000.0
    recorded_follower_gripper_mm = state[:, 6] / 1000.0
    if np.any(recorded_follower_gripper_mm < -0.5) or np.any(
        recorded_follower_gripper_mm > follower_stroke + 2.0
    ):
        raise ValueError(
            "recorded follower gripper feedback exceeds the audited physical range"
        )
    hardware_gripper_percent = np.clip(
        recorded_follower_gripper_mm / leader_stroke * 100.0, 0.0, 100.0
    )
    delta = np.abs(np.diff(recorded_follower_deg, axis=0))
    action_state_error = np.abs(follower_deg - recorded_follower_deg)
    info_path = args.dataset_root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    fps = float(info.get("fps", 30.0))
    metadata = {
        "schema": "assembly_new1_physical_joint_episode_replay_v2",
        "source_dataset": str(args.dataset_root.resolve()),
        "episode_index": int(args.episode_index),
        "frame_count": int(len(frame)),
        "fps": fps,
        "action_semantics": "recorded_leader_driver_percentiles_for_model_target_audit_only",
        "action_audit_clipped_to_schema": action_was_clipped,
        "action_audit_clipped_value_count": int(
            np.count_nonzero(joint_action_overshoot)
            + np.count_nonzero(gripper_action_overshoot)
        ),
        "action_audit_max_joint_overshoot_percent": float(
            joint_action_overshoot.max(initial=0.0)
        ),
        "action_audit_max_gripper_overshoot_percent": float(
            gripper_action_overshoot.max(initial=0.0)
        ),
        "model_semantics": "action_percentiles_decoded_through_recorded_follower_limits",
        "hardware_semantics": "recorded_follower_observation_state_physical_joint_degrees",
        "gripper_semantics": "recorded_follower_feedback_mm_encoded_for_current_hardware_stroke",
        "follower_calibration": str(args.follower_calibration.resolve()),
        "follower_calibration_sha256": sha256_file(args.follower_calibration),
        "leader_calibration": str(args.leader_calibration.resolve()),
        "leader_calibration_sha256": sha256_file(args.leader_calibration),
        "follower_gripper_stroke_mm": follower_stroke,
        "leader_gripper_stroke_mm": leader_stroke,
        "action_vs_state_joint_mae_deg": action_state_error.mean(axis=0).tolist(),
        "action_vs_state_joint_max_deg": action_state_error.max(axis=0).tolist(),
        "max_adjacent_hardware_joint_delta_deg": (
            float(delta.max()) if len(delta) else 0.0
        ),
        "max_adjacent_hardware_joint_delta_per_axis_deg": (
            delta.max(axis=0).tolist() if len(delta) else [0.0] * 6
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        frame_index=frame,
        timestamp=timestamp,
        action_percent=action,
        model_follower_joint_deg=follower_deg,
        recorded_follower_joint_deg=recorded_follower_deg,
        hardware_joint_deg=recorded_follower_deg,
        recorded_follower_gripper_mm=recorded_follower_gripper_mm,
        gripper_percent=hardware_gripper_percent,
    )
    print(json.dumps({**metadata, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
