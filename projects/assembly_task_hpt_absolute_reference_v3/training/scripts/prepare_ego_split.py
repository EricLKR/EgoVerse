#!/usr/bin/env python3
"""Create the deterministic 54/3/3 split for the 60 new Aria episodes."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


VALID_EPISODES = {10, 30, 50}
TEST_EPISODES = {20, 40, 60}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open(newline="")))
    indices = [int(row["episode"]) for row in rows]
    if indices != list(range(1, 61)):
        raise ValueError(f"expected ordered episodes 1..60, got {indices}")

    split_by_episode: dict[str, str] = {}
    groups = {"train": [], "valid": [], "test": []}
    for index in indices:
        split = "valid" if index in VALID_EPISODES else "test" if index in TEST_EPISODES else "train"
        split_by_episode[str(index)] = split
        groups[split].append(index)

    assert {key: len(value) for key, value in groups.items()} == {
        "train": 54,
        "valid": 3,
        "test": 3,
    }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest.resolve()),
        "strategy": "chronology-balanced fixed holdout: valid={10,30,50}, test={20,40,60}",
        "episodes": groups,
        "split_by_episode": split_by_episode,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "counts": {k: len(v) for k, v in groups.items()}}, indent=2))


if __name__ == "__main__":
    main()
