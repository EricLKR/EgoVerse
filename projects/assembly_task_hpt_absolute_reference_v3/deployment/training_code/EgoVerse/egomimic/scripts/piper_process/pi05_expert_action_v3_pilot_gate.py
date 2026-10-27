#!/usr/bin/env python3
"""Technical quality gate between the 3k pilot and 20k resume."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


def _finite_csv_values(paths: list[Path]) -> tuple[int, list[str]]:
    finite = 0
    columns = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for key, raw in row.items():
                    if raw in (None, ""):
                        continue
                    try:
                        value = float(raw)
                    except ValueError:
                        continue
                    if not math.isfinite(value):
                        raise ValueError(f"non-finite metric {key}={raw} in {path}")
                    finite += 1
                    columns.add(key)
    return finite, sorted(columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.pilot_run / "train" / "checkpoints" / "last.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (args.pilot_run / "PILOT_TRAINING_COMPLETED").is_file():
        raise RuntimeError("pilot training completion marker is missing")
    if not (args.pilot_run / "PILOT_EVAL_COMPLETED").is_file():
        raise RuntimeError("pilot evaluation completion marker is missing")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    global_step = int(payload.get("global_step", -1))
    if global_step != 3000:
        raise ValueError(f"pilot global_step must be 3000, got {global_step}")
    if not payload.get("optimizer_states"):
        raise ValueError("pilot checkpoint lacks optimizer state; cannot resume")
    if not payload.get("lr_schedulers"):
        raise ValueError("pilot checkpoint lacks LR scheduler state; cannot resume")
    state_keys = list(payload.get("state_dict", {}))
    required_fragments = (
        "gemma_expert",
        "action_out_proj",
        "egoverse_camera_slot_embeddings",
        "auxiliary.gripper_head",
    )
    missing = [
        fragment
        for fragment in required_fragments
        if not any(fragment in key for key in state_keys)
    ]
    if missing:
        raise ValueError(f"pilot checkpoint lacks required V3 parameters: {missing}")

    metric_paths = sorted(args.pilot_run.rglob("metrics.csv"))
    if not metric_paths:
        raise FileNotFoundError("pilot produced no CSV metrics")
    finite_values, columns = _finite_csv_values(metric_paths)
    if finite_values == 0:
        raise ValueError("pilot CSV logs contain no numeric metrics")
    validation_columns = [column for column in columns if "Valid/" in column]
    if not validation_columns:
        raise ValueError("pilot produced no validation metrics")
    selection_columns = [
        column for column in columns if "physical_selection_score" in column
    ]
    if not selection_columns:
        raise ValueError("pilot produced no PIPER physical selection score")

    report = {
        "pilot_run": str(args.pilot_run),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "global_step": global_step,
        "optimizer_states": len(payload["optimizer_states"]),
        "lr_schedulers": len(payload["lr_schedulers"]),
        "state_dict_keys": len(state_keys),
        "metric_files": [str(path) for path in metric_paths],
        "finite_numeric_metric_values": finite_values,
        "validation_columns": validation_columns,
        "selection_columns": selection_columns,
        "resume_ready": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("EXPERT_ACTION_V3_PILOT_GATE_OK")


if __name__ == "__main__":
    main()
