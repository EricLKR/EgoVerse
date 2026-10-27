#!/usr/bin/env python3
"""Create a hardware-free deployment NPZ from one Assembly New1 Zarr frame.

The deployment wrappers deliberately consume the same raw PIPER feedback units
as the live SDK (0.001 degree and 0.001 mm).  The converted training Zarr stores
physical joint radians and a normalized gripper aperture, so this utility makes
that unit conversion explicit and records its provenance alongside the sample.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import zarr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--calibration-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def decode_rgb(value: object, key: str) -> np.ndarray:
    # zarr-python 3.1 may expose vlen-byte scalars as nested zero-dimensional
    # object arrays. Unwrap only scalar containers; never flatten image data.
    while isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    payload = bytes(value)
    with Image.open(io.BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (480, 640, 3):
        raise RuntimeError(f"{key} decoded to {rgb.shape}, expected (480,640,3)")
    return rgb


def main() -> None:
    args = parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    root = zarr.open_group(str(args.episode), mode="r")
    total_frames = int(root.attrs["total_frames"])
    if not 0 <= args.frame < total_frames:
        raise ValueError(f"frame {args.frame} is outside [0,{total_frames})")
    with args.calibration_json.open() as stream:
        calibration = json.load(stream)
    gripper_stroke_mm = float(calibration["gripper"]["range_max"]) * 1e-3

    joint_rad = np.asarray(root["obs_joint"][args.frame], dtype=np.float64)
    gripper_fraction = float(np.asarray(root["obs_gripper"][args.frame]).reshape(-1)[0])
    if joint_rad.shape != (6,) or not np.isfinite(joint_rad).all():
        raise RuntimeError(f"invalid obs_joint at frame {args.frame}: {joint_rad}")
    if not np.isfinite(gripper_fraction) or not 0.0 <= gripper_fraction <= 1.0:
        raise RuntimeError(f"invalid obs_gripper at frame {args.frame}: {gripper_fraction}")

    raw_state = np.concatenate(
        (
            np.rad2deg(joint_rad) * 1000.0,
            np.asarray([gripper_fraction * gripper_stroke_mm * 1000.0]),
        )
    ).astype(np.float32)
    arrays = {
        "observation.state": np.repeat(raw_state[None, :], args.repeat, axis=0),
        "observation.images.435if": np.repeat(
            decode_rgb(root["images.front_1"][args.frame], "images.front_1")[None, ...],
            args.repeat,
            axis=0,
        ),
        "observation.images.405": np.repeat(
            decode_rgb(root["images.wrist_1"][args.frame], "images.wrist_1")[None, ...],
            args.repeat,
            axis=0,
        ),
        "observation.images.side": np.repeat(
            decode_rgb(root["images.side_1"][args.frame], "images.side_1")[None, ...],
            args.repeat,
            axis=0,
        ),
        "source_episode": np.asarray([args.episode.name]),
        "source_frame": np.asarray([args.frame], dtype=np.int64),
        "state_units": np.asarray(["joint_0.001deg_gripper_0.001mm"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episode": args.episode.name,
                "frame": args.frame,
                "repeat": args.repeat,
                "raw_state": raw_state.tolist(),
                "image_shapes": {
                    key: list(value.shape) for key, value in arrays.items() if "images" in key
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
