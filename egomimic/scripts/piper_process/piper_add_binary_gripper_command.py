"""Add binary commanded-gripper intent to existing PIPER Zarr episodes.

This is an idempotent, additive migration: it leaves every existing array
untouched and writes only ``cmd_gripper``.  Source labels come from each raw
LeRobot episode's recorded ``action[:, 6]`` command, not from the measured
``obs_gripper`` aperture.  A 0.2/0.8 hysteresis converts safety-clamped command
ramps into stable closed/open intent (0=closed, 1=open).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from egomimic.robot.piper.gripper_utils import binarize_gripper_commands


def load_session_actions(session_dir: Path) -> dict[int, np.ndarray]:
    files = sorted((session_dir / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet data found under {session_dir}")
    table = pa.concat_tables(
        [pq.read_table(path, columns=["action", "episode_index", "frame_index"]) for path in files]
    )
    episode_indices = np.asarray(table["episode_index"])
    frame_indices = np.asarray(table["frame_index"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)

    result: dict[int, np.ndarray] = {}
    for episode_index in sorted(np.unique(episode_indices).tolist()):
        mask = episode_indices == episode_index
        order = np.argsort(frame_indices[mask])
        result[int(episode_index)] = actions[mask][order]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--zarr-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--close-threshold", type=float, default=0.2)
    parser.add_argument("--open-threshold", type=float, default=0.8)
    args = parser.parse_args()

    planned: list[tuple[Path, np.ndarray, str, int]] = []
    report = {
        "encoding": "binary command intent: 0=closed, 1=open; in-band values retain prior state",
        "source": "raw LeRobot action[:, 6] / 100",
        "close_threshold": args.close_threshold,
        "open_threshold": args.open_threshold,
        "sessions": {},
    }

    # Preflight all 70 source/target pairs before writing anything.
    for session_index in range(1, 8):
        session = f"teddy_0912_{session_index}"
        source = load_session_actions(args.raw_root / session)
        if sorted(source) != list(range(10)):
            raise ValueError(f"{session}: expected episode indices 0..9, got {sorted(source)}")

        closed = open_ = transitions = frames = 0
        for episode_index, action in source.items():
            target = args.zarr_dir / f"piper-{session}-ep{episode_index:04d}.zarr"
            if not target.is_dir():
                raise FileNotFoundError(f"Missing target episode {target}")
            group = zarr.open_group(str(target), mode="r")
            frame_count = int(group.attrs["total_frames"])
            if len(action) != frame_count:
                raise ValueError(
                    f"{target.name}: raw actions={len(action)}, zarr frames={frame_count}"
                )
            command = binarize_gripper_commands(
                action[:, 6:7] / 100.0,
                close_threshold=args.close_threshold,
                open_threshold=args.open_threshold,
            )
            if "cmd_gripper" in group:
                existing = np.asarray(group["cmd_gripper"][:frame_count], dtype=np.float32)
                if existing.shape != command.shape or not np.array_equal(existing, command):
                    raise ValueError(
                        f"{target.name}: existing cmd_gripper differs; refusing to overwrite"
                    )
            planned.append((target, command, session, episode_index))
            closed += int(np.sum(command == 0.0))
            open_ += int(np.sum(command == 1.0))
            transitions += int(np.sum(command[1:, 0] != command[:-1, 0]))
            frames += len(command)

        report["sessions"][session] = {
            "episodes": len(source),
            "frames": frames,
            "closed_frames": closed,
            "open_frames": open_,
            "transitions": transitions,
        }

    written = skipped = 0
    for target, command, _, _ in planned:
        group = zarr.open_group(str(target), mode="a")
        if "cmd_gripper" in group:
            skipped += 1
            continue
        group["cmd_gripper"] = command
        group.attrs["cmd_gripper_encoding"] = {
            "source": "raw LeRobot action[:, 6] / 100",
            "closed": 0,
            "open": 1,
            "close_threshold": args.close_threshold,
            "open_threshold": args.open_threshold,
            "in_band": "retain_previous_state",
        }
        stored = np.asarray(group["cmd_gripper"][:], dtype=np.float32)
        if stored.shape != command.shape or not np.array_equal(stored, command):
            raise RuntimeError(f"Post-write verification failed for {target}")
        written += 1

    report["episode_count"] = len(planned)
    report["written"] = written
    report["already_present_and_equal"] = skipped
    report["total_frames"] = sum(item["frames"] for item in report["sessions"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
