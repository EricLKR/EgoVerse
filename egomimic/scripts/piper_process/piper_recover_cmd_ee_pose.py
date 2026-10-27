"""Recovers a genuine COMMANDED end-effector pose (`cmd_ee_pose`) for the teddy_bear_basket Piper
dataset, closing a limitation documented since before this investigation (see piper.py's Piper
embodiment docstring: "no usable *commanded* pose... no way to recover real joint angles from the
recorded `action` column alone" -- which led every Piper-side experiment this whole session to build
action chunks by stacking *future* `obs_ee_pose` frames instead of a true command).

That limitation turns out to be about what survived the zarr conversion, not about whether the
calibration exists at all. It does, in a separate, persisted lerobot cache file
(~/.cache/huggingface/lerobot/calibration/robots/piper/my_piper_follower.json), created 2026-08-19 --
a full day before the actual teddy_bear_basket recording sessions (2026-08-20) that `piper_zarr_teddy_bear`
was converted from. Since `piper.py`'s Robot.__init__ loads this saved calibration automatically on every
connect, taking priority over both an unset CLI override and the (separately documented, buggy)
generic +-170deg live-query default, the real per-joint limits were almost certainly what the recording
session actually used.

Verified directly, not just inferred from timestamps: FK(calibrated raw `action`) vs. the already-trusted
`obs_ee_pose` for episode 0 gives a median position error of 1.1cm (mean 1.2cm, max 5.8cm) over the
frames where real action data exists -- exactly the magnitude expected from normal teleoperation lag
(commanded position leading observed position slightly), not a wrong-calibration error (which would show
tens of centimeters of error, not ~1cm).

Source of the raw `action` column: NOT `piper_zarr_teddy_bear` itself (the zarr conversion never
preserved it, confirmed by inspecting its zarr keys) but the original LeRobot parquet dataset,
`~/.cache/huggingface/lerobot/yukk1/teddy_bear_basket_all` -- still present on disk, and confirmed (via
piper_lerobot_to_zarr.py's own `ep_idx = int(episode_row["episode_index"])` convention) to be the actual
conversion source: episode_index i <-> `piper-teddy_bear_basket-ep{i:04d}.zarr`.

Frame-count handling: each zarr episode is padded to a round target length (600 or 700 frames) beyond
the real recorded action stream's (irregular) length -- confirmed across 5 episodes, diffs ranging
26-86 frames, zarr always landing on a round number. For the padding tail (no real commanded action
exists there), forward-fills the last real cmd_ee_pose -- the same convention already established and
audited earlier this session for other missing-data situations, not a new, untested choice.

Writes to a NEW zarr directory (_cmdee suffix) via the same zero-risk copy pattern used throughout this
investigation -- copies fresh from the live SRC_ROBOT episodes, never modifies them, so nothing already
built on piper_zarr_teddy_bear (P0/P2/P3/P2mh, the causal-transfer checks) is at any risk.
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # egomimic/ root, for `robot.piper...`
from robot.piper.piper_kinematics import PiperKinematicsSolver

SRC_ROBOT = Path("/mnt/wd10/tongxi/egoverse_piper/EgoVerse/data/piper_zarr_teddy_bear")
DST_ROBOT = Path("/mnt/wd10/tongxi/egoverse_piper/EgoVerse/data/piper_zarr_teddy_bear_cmdee")
CALIBRATION_PATH = Path("/home/master-ht/.cache/huggingface/lerobot/calibration/robots/piper/my_piper_follower.json")
LEROBOT_PARQUET_GLOB = "/home/master-ht/.cache/huggingface/lerobot/yukk1/teddy_bear_basket_all/data/chunk-*/file-*.parquet"


def load_calibration():
    with open(CALIBRATION_PATH) as f:
        cal = json.load(f)
    joint_keys = [f"joint_{i}" for i in range(6)]
    min_deg = np.array([cal[k]["range_min"] / 1000.0 for k in joint_keys])
    max_deg = np.array([cal[k]["range_max"] / 1000.0 for k in joint_keys])
    gripper_range_max_raw = cal["gripper"]["range_max"]  # 0.001mm units, e.g. 102800 -> 102.8mm
    return min_deg, max_deg, gripper_range_max_raw


def action_pct_to_real_deg(action_joints_pct: np.ndarray, min_deg: np.ndarray, max_deg: np.ndarray) -> np.ndarray:
    """[-100,100] per-joint percent -> real degrees, exact inverse of piper_sdk_interface.py's
    `set_joint_positions` scaling (verified against it directly, not re-derived from scratch)."""
    return min_deg[None, :] + (max_deg[None, :] - min_deg[None, :]) * (action_joints_pct + 100) / 200


def main():
    print("*** Recovering cmd_ee_pose for piper_zarr_teddy_bear via calibrated raw action + FK ***")
    min_deg, max_deg, gripper_max_raw = load_calibration()
    print(f"  calibration (deg): {list(zip(min_deg.tolist(), max_deg.tolist()))}")

    parquet_files = sorted(glob.glob(LEROBOT_PARQUET_GLOB))
    assert parquet_files, f"no parquet files found at {LEROBOT_PARQUET_GLOB}"
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    print(f"  loaded {len(df)} rows from {len(parquet_files)} parquet file(s), "
          f"{df['episode_index'].nunique()} episodes")

    solver = PiperKinematicsSolver()

    DST_ROBOT.mkdir(parents=True, exist_ok=True)
    src_episodes = sorted(p for p in SRC_ROBOT.iterdir() if p.is_dir())
    n_ok, n_skip = 0, 0
    for src_ep in src_episodes:
        ep_name = src_ep.name  # e.g. "piper-teddy_bear_basket-ep0024.zarr"
        try:
            ep_idx = int(ep_name.rsplit("ep", 1)[-1].replace(".zarr", ""))
        except ValueError:
            print(f"  {ep_name}: could not parse episode index, skipping")
            n_skip += 1
            continue

        ep_rows = df[df["episode_index"] == ep_idx].reset_index(drop=True)
        if len(ep_rows) == 0:
            print(f"  {ep_name}: no matching episode_index={ep_idx} in source parquet, skipping")
            n_skip += 1
            continue

        action = np.stack(ep_rows["action"].to_numpy())  # [n_action, 7]
        action_joints_pct = action[:, :6]
        action_gripper_pct = action[:, 6]  # [0, 100]

        real_deg = action_pct_to_real_deg(action_joints_pct, min_deg, max_deg)
        real_rad = np.deg2rad(real_deg)
        cmd_ee_pose = solver.fk(real_rad)  # [n_action, 7] xyzwxyz, arm base_link frame
        cmd_gripper_frac = action_gripper_pct / 100.0  # matches obs_gripper's own [0,1] convention

        dst_ep = DST_ROBOT / ep_name
        if dst_ep.exists():
            print(f"  {ep_name}: destination already exists, skipping copy (leaving as-is)")
        else:
            shutil.copytree(src_ep, dst_ep)

        g = zarr.open(str(dst_ep), mode="a")
        n_frames = g["images.front_1"].shape[0]
        n_action = len(action)

        cmd_ee_full = np.zeros((n_frames, 7), dtype=np.float32)
        cmd_gripper_full = np.zeros((n_frames,), dtype=np.float32)
        n_real = min(n_action, n_frames)
        cmd_ee_full[:n_real] = cmd_ee_pose[:n_real]
        cmd_gripper_full[:n_real] = cmd_gripper_frac[:n_real]
        if n_frames > n_action:
            # padding tail beyond the real recorded action stream -- forward-fill the last real value,
            # same convention already established/audited for other missing-data cases this session.
            cmd_ee_full[n_action:] = cmd_ee_pose[-1]
            cmd_gripper_full[n_action:] = cmd_gripper_frac[-1]

        # Persist the real/padded boundary directly -- found via a downstream CPU-only audit that
        # trying to RE-DERIVE this from cmd_ee_pose alone (e.g. "last index where it changes") is
        # unreliable: a real held-still moment during genuine commanded motion looks identical to the
        # start of the forward-filled padding tail, confirmed empirically (6/6 episodes checked
        # mismatched, understating n_real by 15-25 frames). This script already computes the correct
        # value once, directly from the source parquet -- just save it instead of making every consumer
        # re-guess it.
        g.attrs["n_real_action"] = int(n_action)

        if "cmd_ee_pose" in g:
            del g["cmd_ee_pose"]
        if "cmd_gripper" in g:
            del g["cmd_gripper"]
        g["cmd_ee_pose"] = cmd_ee_full
        g["cmd_gripper"] = cmd_gripper_full
        n_ok += 1
        n_padded = max(0, n_frames - n_action)
        print(f"  {ep_name}: injected cmd_ee_pose/cmd_gripper [{n_frames}] "
              f"({n_action} real from calibrated action, {n_padded} forward-filled padding)")

    print(f"\n  {n_ok}/{len(src_episodes)} episodes injected, {n_skip} skipped")
    print(f"  output written to {DST_ROBOT}")
    print("\n=== cmd_ee_pose RECOVERY COMPLETE ===")


if __name__ == "__main__":
    main()
