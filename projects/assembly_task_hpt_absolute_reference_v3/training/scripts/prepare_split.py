#!/usr/bin/env python3
"""Create a deterministic, collection-order-balanced 90/5/5 split manifest.

The raw LeRobot dataset is never modified.  The split is consumed by the Zarr
converter and stored again in every converted episode's attributes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


VALID = [9, 29, 49, 69, 89]
TEST = [19, 39, 59, 79, 99]
TRAIN = sorted(set(range(100)) - set(VALID) - set(TEST))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    info_path = root / "meta" / "info.json"
    scene_path = root / "meta" / "scene.json"
    reset_path = root / "meta" / "reset_control.json"
    extrinsics_path = root / "meta" / "head_cam_to_follower_base.calib"
    info = json.loads(info_path.read_text())
    if int(info.get("total_episodes", -1)) != 100:
        raise ValueError(f"expected 100 episodes, got {info.get('total_episodes')}")

    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_files:
        raise FileNotFoundError("no episode parquet files")
    table = pa.concat_tables([pq.read_table(path) for path in episode_files]).to_pandas()
    indices = sorted(int(value) for value in table["episode_index"].tolist())
    if indices != list(range(100)):
        raise ValueError("episode indices are not exactly 0..99")

    split_by_episode = {str(index): "train" for index in TRAIN}
    split_by_episode.update({str(index): "valid" for index in VALID})
    split_by_episode.update({str(index): "test" for index in TEST})
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(root),
        "strategy": "systematic_holdout_across_collection_order",
        "seed": None,
        "counts": {"train": 90, "valid": 5, "test": 5},
        "episodes": {"train": TRAIN, "valid": VALID, "test": TEST},
        "split_by_episode": split_by_episode,
        "source_sha256": {
            "meta/info.json": sha256(info_path),
            "meta/scene.json": sha256(scene_path),
            "meta/reset_control.json": sha256(reset_path),
            "meta/head_cam_to_follower_base.calib": sha256(extrinsics_path),
        },
        "camera_policy": {
            "included": ["head", "wrist", "side"],
            "excluded": ["aria"],
        },
        "phase_sampling": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
