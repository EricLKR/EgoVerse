#!/usr/bin/env python3
"""Verify all Gen2 conversion shards and write one canonical status file."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import zarr


CONTRACT = "aria_assembly_new1_right_gen2_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    args = parser.parse_args()

    rows = {
        int(row["episode"]): row
        for row in csv.DictReader(args.manifest.open(newline=""))
    }
    split_by_episode = json.loads(args.split_json.read_text())["split_by_episode"]
    if sorted(rows) != list(range(1, 61)):
        raise ValueError("source manifest does not contain exactly episodes 1..60")

    episodes: dict[str, dict] = {}
    for episode in range(1, 61):
        path = args.output_root / f"aria-assembly-new1-right-ep{episode:04d}.zarr"
        if not path.is_dir():
            raise FileNotFoundError(path)
        group = zarr.open_group(str(path), mode="r")
        expected = rows[episode]
        split = split_by_episode[str(episode)]
        checks = {
            "source_episode_index": int(group.attrs.get("source_episode_index", -1)) == episode,
            "source_vrs_sha256": group.attrs.get("source_vrs_sha256") == expected["vrs_sha256"],
            "dataset_split": group.attrs.get("dataset_split") == split,
            "data_contract": group.attrs.get("data_contract") == CONTRACT,
            "embodiment": group.attrs.get("embodiment") == "human_right_arm",
            "trajectory_source": group.attrs.get("trajectory_source")
            == "vrs_stream_371-4_vio_high_frequency",
            "trajectory_reference_frame": group.attrs.get("trajectory_reference_frame")
            == "per_episode_odometry",
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"episode {episode:03d} failed provenance checks: {failed}")
        frames = int(group.attrs.get("total_frames", 0))
        if frames <= 30:
            raise ValueError(f"episode {episode:03d} has only {frames} valid frames")
        episodes[str(episode)] = {
            "state": "complete",
            "output": str(path.resolve()),
            "split": split,
            "total_frames": frames,
            "stats": dict(group.attrs.get("conversion_stats", {})),
        }

    status = {
        "schema_version": 2,
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "data_contract": CONTRACT,
        "episodes": episodes,
        "counts": {
            "total": len(episodes),
            "train": sum(item["split"] == "train" for item in episodes.values()),
            "valid": sum(item["split"] == "valid" for item in episodes.values()),
            "test": sum(item["split"] == "test" for item in episodes.values()),
        },
    }
    output = args.output_root / "conversion_status.json"
    output.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
