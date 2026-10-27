#!/usr/bin/env python3
"""Fail-fast contract audit for V5 compact checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _has(keys: list[str], token: str) -> bool:
    return any(token in key for key in keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") != "egomimic_trainable_state_and_buffers_v5":
        raise ValueError(f"unexpected checkpoint format: {payload.get('checkpoint_format')}")
    if int(payload.get("global_step", -1)) != args.expected_step:
        raise ValueError(
            f"expected global_step={args.expected_step}, got {payload.get('global_step')}"
        )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint has no state_dict")
    keys = sorted(state)
    required_tokens = [
        "egoverse_v5_conditioning.state_adapters.0",
        "egoverse_v5_conditioning.state_adapters.1",
        "egoverse_v5_conditioning.embodiment_embeddings",
        "egoverse_v5_training_step",
        "egoverse_camera_slot_embeddings",
        "joint_residual_scale_rad_vector",
        "gripper_head",
        "egoverse_lora_a",
        "egoverse_lora_b",
    ]
    missing = [token for token in required_tokens if not _has(keys, token)]
    if missing:
        raise ValueError(f"checkpoint contract keys are missing: {missing}")
    step_keys = [key for key in keys if key.endswith("egoverse_v5_training_step")]
    if len(step_keys) != 1:
        raise ValueError(f"expected one V5 schedule buffer, got {step_keys}")
    stored_step = int(state[step_keys[0]].item())
    if stored_step != args.expected_step:
        raise ValueError(
            f"V5 schedule buffer is {stored_step}, expected {args.expected_step}"
        )
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("checkpoint contains NaN/Inf")

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_format": payload["checkpoint_format"],
        "global_step": args.expected_step,
        "v5_schedule_step": stored_step,
        "state_key_count": len(keys),
        "stored_tensor_element_count": sum(value.numel() for value in state.values()),
        "required_contract_tokens": required_tokens,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PI05_MIXED_CAMFRAME_V5_CHECKPOINT_GATE_OK")


if __name__ == "__main__":
    main()
