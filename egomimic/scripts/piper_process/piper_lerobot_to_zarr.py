"""Converts the raw LeRobot-format PIPER-L recordings (yukk1/pick_and_place_1,
yukk1/pick_and_place_2 -- or any dataset merged/split from them with lerobot's
`lerobot_edit_dataset`) into EgoVerse Zarr v3 episodes.

Deliberately reads the dataset's parquet/mp4 files directly with pyarrow + PyAV
rather than importing `lerobot`: this script runs in the `emimic` conda env
(needs egomimic's ZarrWriter + FK), which doesn't have `lerobot` installed, and
the `lerobot` env doesn't have `egomimic` installed. Neither env has both, and
the LeRobot v3.0 on-disk layout (data/chunk-*/file-*.parquet, videos/<key>/
chunk-*/file-*.mp4, meta/episodes/*.parquet with per-episode row/timestamp
ranges) is simple enough to read directly rather than stand up a shared env
for it.

What this does NOT yet have real values for -- see --help / the flags below:
  - Camera intrinsics (`images.front_1` = 435if/D435i @ 1280x720, `images.wrist_1`
    = 405/D405 @ 848x480). ZarrWriter *requires* intrinsics; nothing in the
    LeRobot dataset carries this rig's actual factory/checkerboard calibration.
    --intrinsics-json expects a {"front_1": [[fx,0,cx,0],[0,fy,cy,0],[0,0,1,0]],
    "wrist_1": [...]} file with this rig's real, per-camera-calibrated values.
    --allow-approximate-intrinsics falls back to RealSense nominal/typical
    values -- clearly NOT this unit's calibration, fine for a structural
    pipeline smoke test, not for anything that will be used for real training,
    trajectory-overlay viz, or upload.
  - Extrinsics (camera<-base transform): left as None (Piper.EXTRINSICS is also
    None), which restricts training to `cartesian_base` mode (ee_pose in the
    arm's own base frame) -- see egomimic/rldb/embodiment/piper.py.
  - Which physical camera is "front" vs "wrist": assumed 435if=front, 405=wrist
    per Piper.get_keymap's docstring; unverified against the physical rig.

Usage:
    conda run -n emimic python piper_lerobot_to_zarr.py \
        --dataset-root ~/.cache/huggingface/lerobot/yukk1/pick_and_place_all \
        --out-dir /path/to/local/piper_zarr \
        --calibration-json ~/.cache/huggingface/lerobot/calibration/robots/piper/my_piper_follower.json \
        --intrinsics-json my_calibrated_intrinsics.json

With ``--calibration-json`` the converter also writes the joint-space training
signals ``obs_joint`` and ``cmd_joint`` (radians), plus ``cmd_ee_pose`` obtained
by applying the same DH forward kinematics to ``cmd_joint``.  The calibration
is mandatory for command recovery: LeRobot stores joint commands as percentages
of this particular follower's calibrated range, not as angles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from egomimic.robot.piper.piper_kinematics import (
    PiperKinematicsSolver,
    raw_gripper_to_frac,
    raw_joint_to_rad,
)
from egomimic.robot.piper.gripper_utils import binarize_gripper_commands
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

CAMERA_ZARR_KEYS = {
    "observation.images.435if": "images.front_1",  # D435i, 1280x720 -- see module docstring
    "observation.images.405": "images.wrist_1",  # D405, 848x480 -- see module docstring
}

# Nominal/typical RealSense color-stream intrinsics -- NOT any specific unit's
# factory calibration. Only for --allow-approximate-intrinsics smoke-testing.
_APPROX_INTRINSICS = {
    "front_1": [[920.0, 0.0, 640.0, 0.0], [0.0, 920.0, 360.0, 0.0], [0.0, 0.0, 1.0, 0.0]],  # D435i @ 1280x720
    "wrist_1": [[430.0, 0.0, 424.0, 0.0], [0.0, 430.0, 240.0, 0.0], [0.0, 0.0, 1.0, 0.0]],  # D405 @ 848x480
}


def _load_episode_table(dataset_root: Path):
    ep_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    return pq.read_table(ep_files[0]).to_pandas() if len(ep_files) == 1 else pa.concat_tables(
        [pq.read_table(f) for f in ep_files]
    ).to_pandas()


def _decode_video_range(video_path: Path, from_s: float, to_s: float, fps: int) -> np.ndarray:
    """Decode frames [round(from_s*fps), round(to_s*fps)) from an mp4, RGB HWC uint8."""
    start_idx = round(from_s * fps)
    end_idx = round(to_s * fps)
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i < start_idx:
                continue
            if i >= end_idx:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames, axis=0)


def load_joint_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load this follower's six calibrated joint ranges in radians."""
    calibration = json.loads(path.expanduser().read_text())
    joint_keys = [f"joint_{index}" for index in range(6)]
    missing = [key for key in joint_keys if key not in calibration]
    if missing:
        raise ValueError(f"Calibration {path} is missing keys: {missing}")
    low_deg = np.asarray(
        [calibration[key]["range_min"] / 1000.0 for key in joint_keys], dtype=np.float64
    )
    high_deg = np.asarray(
        [calibration[key]["range_max"] / 1000.0 for key in joint_keys], dtype=np.float64
    )
    if np.any(high_deg <= low_deg):
        raise ValueError(f"Invalid calibrated joint ranges in {path}")
    return np.deg2rad(low_deg), np.deg2rad(high_deg)


def action_percent_to_joint_rad(
    action_percent: np.ndarray,
    low_rad: np.ndarray,
    high_rad: np.ndarray,
) -> np.ndarray:
    """Invert LeRobot PIPER's per-joint ``[-100, 100]`` command scaling."""
    action_percent = np.asarray(action_percent, dtype=np.float64)
    if action_percent.ndim != 2 or action_percent.shape[1] != 6:
        raise ValueError(f"Expected joint commands shaped (T, 6), got {action_percent.shape}")
    return low_rad[None] + (high_rad - low_rad)[None] * (action_percent + 100.0) / 200.0


def convert_episode(
    dataset_root: Path,
    episode_row,
    data_df,
    out_dir: Path,
    fk: PiperKinematicsSolver,
    intrinsics: dict,
    task_name: str,
    joint_calibration_rad: tuple[np.ndarray, np.ndarray] | None = None,
    verbose: bool = True,
) -> Path:
    ep_idx = int(episode_row["episode_index"])
    frm, to = int(episode_row["dataset_from_index"]), int(episode_row["dataset_to_index"])
    ep_rows = data_df.iloc[frm:to]

    state = np.stack(ep_rows["observation.state"].to_numpy())  # (T, 7) raw units
    action = np.stack(ep_rows["action"].to_numpy())  # joints [-100,100], gripper [0,100]
    assert state.shape[0] == episode_row["length"]
    assert action.shape == state.shape

    obs_joint = raw_joint_to_rad(state[:, :6]).astype(np.float64)
    obs_ee_pose = fk.fk(obs_joint).astype(np.float64)  # (T, 7) XYZWXYZ, base_link frame
    obs_gripper = raw_gripper_to_frac(state[:, 6:7]).astype(np.float64)  # (T, 1)
    cmd_gripper = binarize_gripper_commands(action[:, 6:7] / 100.0)  # (T, 1), 0=closed/1=open
    cmd_joint = None
    cmd_ee_pose = None
    if joint_calibration_rad is not None:
        low_rad, high_rad = joint_calibration_rad
        cmd_joint = action_percent_to_joint_rad(action[:, :6], low_rad, high_rad)
        cmd_ee_pose = fk.fk(cmd_joint).astype(np.float64)

    image_data = {}
    for lerobot_key, zarr_key in CAMERA_ZARR_KEYS.items():
        chunk_idx = int(episode_row[f"videos/{lerobot_key}/chunk_index"])
        file_idx = int(episode_row[f"videos/{lerobot_key}/file_index"])
        from_ts = float(episode_row[f"videos/{lerobot_key}/from_timestamp"])
        to_ts = float(episode_row[f"videos/{lerobot_key}/to_timestamp"])
        video_path = dataset_root / "videos" / lerobot_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        frames = _decode_video_range(video_path, from_ts, to_ts, fps=30)
        if frames.shape[0] != state.shape[0]:
            # Off-by-a-frame-or-two from float timestamp rounding is common; trim to the
            # shorter of the two rather than fail the whole episode.
            n = min(frames.shape[0], state.shape[0])
            frames = frames[:n]
        image_data[zarr_key] = frames

    # Trim numeric arrays to match, in case images.shape[0] < state.shape[0] above.
    n = min(obs_ee_pose.shape[0], next(iter(image_data.values())).shape[0])
    obs_joint, obs_ee_pose, obs_gripper, cmd_gripper = (
        obs_joint[:n],
        obs_ee_pose[:n],
        obs_gripper[:n],
        cmd_gripper[:n],
    )
    if cmd_joint is not None:
        cmd_joint = cmd_joint[:n]
        cmd_ee_pose = cmd_ee_pose[:n]
    image_data = {k: v[:n] for k, v in image_data.items()}

    episode_hash = f"piper-{task_name}-ep{ep_idx:04d}"  # placeholder id, not a UTC-timestamp hash
    episode_path = out_dir / f"{episode_hash}.zarr"

    if verbose:
        print(f"episode {ep_idx}: {n} frames -> {episode_path}")

    numeric_data = {
        "obs_joint": obs_joint,
        "obs_ee_pose": obs_ee_pose,
        "obs_gripper": obs_gripper,
        "cmd_gripper": cmd_gripper,
    }
    if cmd_joint is not None:
        numeric_data.update(
            {
                "cmd_joint": cmd_joint,
                "cmd_ee_pose": cmd_ee_pose,
            }
        )

    ZarrWriter.create_and_write(
        episode_path=episode_path,
        numeric_data=numeric_data,
        image_data=image_data,
        embodiment="piper_left_arm",
        fps=30,
        task_name=task_name,
        task_description="; ".join(episode_row["tasks"]),
        annotations=[],
        intrinsics=intrinsics,
        extrinsics=None,
    )
    return episode_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--task-name", default="pick_and_place")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="Subset of episode indices; default all.")
    ap.add_argument("--intrinsics-json", type=Path, default=None)
    ap.add_argument(
        "--calibration-json",
        type=Path,
        default=None,
        help=(
            "Follower calibration JSON used to recover cmd_joint radians and cmd_ee_pose. "
            "Required for the joint-space training configuration."
        ),
    )
    ap.add_argument("--allow-approximate-intrinsics", action="store_true")
    args = ap.parse_args()

    if args.intrinsics_json is not None:
        intrinsics = json.loads(args.intrinsics_json.read_text())
    elif args.allow_approximate_intrinsics:
        print("WARNING: using nominal RealSense intrinsics, NOT this rig's calibration.")
        intrinsics = _APPROX_INTRINSICS
    else:
        raise SystemExit(
            "No intrinsics given. Pass --intrinsics-json <file> with this rig's real "
            "per-camera calibration, or --allow-approximate-intrinsics for a structural "
            "smoke test only (see this script's module docstring)."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root.expanduser()

    data_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    data_df = pa.concat_tables([pq.read_table(f) for f in data_files]).to_pandas()

    ep_table = _load_episode_table(dataset_root)
    if args.episodes is not None:
        ep_table = ep_table[ep_table["episode_index"].isin(args.episodes)]

    fk = PiperKinematicsSolver()
    joint_calibration_rad = (
        load_joint_calibration(args.calibration_json)
        if args.calibration_json is not None
        else None
    )
    if joint_calibration_rad is None:
        print(
            "WARNING: --calibration-json was not supplied; cmd_joint/cmd_ee_pose will not "
            "be written and the output cannot be used by the joint-space training config."
        )
    for _, row in ep_table.iterrows():
        convert_episode(
            dataset_root,
            row,
            data_df,
            args.out_dir,
            fk,
            intrinsics,
            args.task_name,
            joint_calibration_rad=joint_calibration_rad,
        )


if __name__ == "__main__":
    main()
