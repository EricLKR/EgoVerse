"""Add calibrated joint-space signals to existing PIPER Zarr episodes.

This migration is additive and idempotent.  It preserves the existing images,
Cartesian arrays, metadata, and binary gripper labels.  The source LeRobot
recordings contain measured joints in raw 0.001-degree units and commanded
joints as percentages of the calibrated follower range.  We recover:

* ``obs_joint``: six measured absolute joints in radians;
* ``cmd_joint``: six commanded absolute joints in radians;
* ``cmd_ee_pose``: FK(cmd_joint), for metrics/plots only.

No camera extrinsics or inverse kinematics are involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from egomimic.robot.piper.piper_kinematics import PiperKinematicsSolver, raw_joint_to_rad


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    payload = path.read_bytes()
    calibration = json.loads(payload)
    keys = [f"joint_{index}" for index in range(6)]
    missing = [key for key in keys if key not in calibration]
    if missing:
        raise ValueError(f"Calibration {path} is missing keys {missing}")
    low = np.deg2rad(
        np.asarray([calibration[key]["range_min"] for key in keys], dtype=np.float64) / 1000.0
    )
    high = np.deg2rad(
        np.asarray([calibration[key]["range_max"] for key in keys], dtype=np.float64) / 1000.0
    )
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)) or np.any(high <= low):
        raise ValueError(f"Invalid joint ranges in {path}")
    return low, high, hashlib.sha256(payload).hexdigest()


def action_percent_to_joint_rad(
    action_percent: np.ndarray, low_rad: np.ndarray, high_rad: np.ndarray
) -> np.ndarray:
    action_percent = np.asarray(action_percent, dtype=np.float64)
    if action_percent.ndim != 2 or action_percent.shape[1] != 6:
        raise ValueError(f"Expected action joint shape (T, 6), got {action_percent.shape}")
    return low_rad[None] + (high_rad - low_rad)[None] * (action_percent + 100.0) / 200.0


def load_session(session_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    files = sorted((session_dir / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {session_dir}")
    table = pa.concat_tables(
        [
            pq.read_table(
                path,
                columns=["observation.state", "action", "episode_index", "frame_index"],
            )
            for path in files
        ]
    )
    episode_indices = np.asarray(table["episode_index"])
    frame_indices = np.asarray(table["frame_index"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for episode_index in sorted(np.unique(episode_indices).tolist()):
        mask = episode_indices == episode_index
        order = np.argsort(frame_indices[mask])
        result[int(episode_index)] = (states[mask][order], actions[mask][order])
    return result


def array_equal(existing: np.ndarray, expected: np.ndarray) -> bool:
    return existing.shape == expected.shape and np.allclose(
        existing, expected, rtol=0.0, atol=1e-10, equal_nan=False
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--zarr-dir", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    low_rad, high_rad, calibration_sha256 = load_calibration(args.calibration_json)
    fk = PiperKinematicsSolver()
    planned: list[tuple[Path, dict[str, np.ndarray], str]] = []
    report: dict = {
        "representation": "six absolute commanded joint radians + independent binary gripper",
        "cmd_ee_pose_role": "FK audit/evaluation only; not a model input or output",
        "calibration_path": str(args.calibration_json),
        "calibration_sha256": calibration_sha256,
        "joint_min_rad": low_rad.tolist(),
        "joint_max_rad": high_rad.tolist(),
        "sessions": {},
    }

    # Validate every source/target pair before the first write.
    for session_index in range(1, 8):
        session = f"teddy_0912_{session_index}"
        source = load_session(args.raw_root / session)
        if sorted(source) != list(range(10)):
            raise ValueError(f"{session}: expected episodes 0..9, got {sorted(source)}")
        session_stats = {
            "episodes": 0,
            "frames": 0,
            "obs_joint_min_rad": np.full(6, np.inf),
            "obs_joint_max_rad": np.full(6, -np.inf),
            "cmd_joint_min_rad": np.full(6, np.inf),
            "cmd_joint_max_rad": np.full(6, -np.inf),
            "max_abs_fk_recompute_error": 0.0,
        }
        for episode_index, (state, action) in source.items():
            target = args.zarr_dir / f"piper-{session}-ep{episode_index:04d}.zarr"
            if not target.is_dir():
                raise FileNotFoundError(f"Missing target {target}")
            group = zarr.open_group(str(target), mode="r")
            frame_count = int(group.attrs["total_frames"])
            if len(state) != frame_count or len(action) != frame_count:
                raise ValueError(
                    f"{target.name}: state={len(state)}, action={len(action)}, zarr={frame_count}"
                )
            if state.shape[1] < 6 or action.shape[1] < 6:
                raise ValueError(f"{target.name}: invalid state/action dimensions")
            obs_joint = raw_joint_to_rad(state[:, :6]).astype(np.float64)
            cmd_joint = action_percent_to_joint_rad(action[:, :6], low_rad, high_rad)
            cmd_ee_pose = fk.fk(cmd_joint).astype(np.float64)
            arrays = {
                "obs_joint": obs_joint,
                "cmd_joint": cmd_joint,
                "cmd_ee_pose": cmd_ee_pose,
            }
            for name, expected in arrays.items():
                if not np.isfinite(expected).all():
                    raise ValueError(f"{target.name}: non-finite {name}")
                if name in group:
                    existing = np.asarray(group[name][:frame_count], dtype=np.float64)
                    if not array_equal(existing, expected):
                        raise ValueError(
                            f"{target.name}: existing {name} differs; refusing to overwrite"
                        )
            tolerance = 1e-8
            if np.any(cmd_joint < low_rad[None] - tolerance) or np.any(
                cmd_joint > high_rad[None] + tolerance
            ):
                raise ValueError(f"{target.name}: recovered command exceeds calibrated range")
            planned.append((target, arrays, session))
            session_stats["episodes"] += 1
            session_stats["frames"] += frame_count
            session_stats["obs_joint_min_rad"] = np.minimum(
                session_stats["obs_joint_min_rad"], obs_joint.min(axis=0)
            )
            session_stats["obs_joint_max_rad"] = np.maximum(
                session_stats["obs_joint_max_rad"], obs_joint.max(axis=0)
            )
            session_stats["cmd_joint_min_rad"] = np.minimum(
                session_stats["cmd_joint_min_rad"], cmd_joint.min(axis=0)
            )
            session_stats["cmd_joint_max_rad"] = np.maximum(
                session_stats["cmd_joint_max_rad"], cmd_joint.max(axis=0)
            )
        report["sessions"][session] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in session_stats.items()
        }

    written = skipped = 0
    metadata = {
        "obs_joint": "measured absolute joint angles, radians, joints 0..5",
        "cmd_joint": "commanded absolute joint angles, radians, joints 0..5",
        "cmd_ee_pose": "base-frame FK(cmd_joint), xyz + quaternion xyzw, metrics only",
        "calibration_sha256": calibration_sha256,
        "joint_min_rad": low_rad.tolist(),
        "joint_max_rad": high_rad.tolist(),
    }
    for target, arrays, _ in planned:
        group = zarr.open_group(str(target), mode="a")
        for name, expected in arrays.items():
            if name in group:
                skipped += 1
            else:
                group[name] = expected
                written += 1
            stored = np.asarray(group[name][:], dtype=np.float64)
            if not array_equal(stored, expected):
                raise RuntimeError(f"Post-write verification failed: {target.name}/{name}")
        features = dict(group.attrs.get("features", {}))
        features.update(
            {
                "obs_joint": {
                    "dtype": "float64",
                    "shape": [6],
                    "names": ["dim_0"],
                },
                "cmd_joint": {
                    "dtype": "float64",
                    "shape": [6],
                    "names": ["dim_0"],
                },
                "cmd_ee_pose": {
                    "dtype": "float64",
                    "shape": [7],
                    "names": ["dim_0"],
                },
            }
        )
        group.attrs["features"] = features
        group.attrs["joint_space_encoding"] = metadata

    report["episode_count"] = len(planned)
    report["total_frames"] = sum(item["frames"] for item in report["sessions"].values())
    report["arrays_written"] = written
    report["arrays_already_present_and_equal"] = skipped
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
