#!/usr/bin/env python3
"""Capture the three deployment RGB streams without opening the robot or CAN."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
import pyrealsense2 as rs


CAMERAS = {
    "front": "337322071261",
    "wrist": "427622273943",
    "side": "427622272585",
}


def image_metrics(rgb: np.ndarray) -> dict[str, object]:
    rgb_f = rgb.astype(np.float32)
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32)
    lab = np.asarray(Image.fromarray(rgb).convert("LAB"), dtype=np.float32)
    return {
        "rgb_mean": rgb_f.mean(axis=(0, 1)).tolist(),
        "rgb_std": rgb_f.std(axis=(0, 1)).tolist(),
        "hsv_saturation_mean": float(hsv[..., 1].mean()),
        "hsv_saturation_median": float(np.median(hsv[..., 1])),
        "hsv_saturation_p10": float(np.percentile(hsv[..., 1], 10)),
        "hsv_saturation_p90": float(np.percentile(hsv[..., 1], 90)),
        "hsv_value_mean": float(hsv[..., 2].mean()),
        "lab_mean": lab.mean(axis=(0, 1)).tolist(),
        "red_blue_ratio": float((rgb_f[..., 0].mean() + 1.0) / (rgb_f[..., 2].mean() + 1.0)),
        "dark_fraction": float((hsv[..., 2] < 25).mean()),
        "highlight_fraction": float((hsv[..., 2] > 250).mean()),
    }


def sensor_options(profile: rs.pipeline_profile) -> dict[str, float | bool | None]:
    device = profile.get_device()
    color_sensor = None
    for sensor in device.query_sensors():
        has_color_profile = any(
            profile.stream_type() == rs.stream.color
            for profile in sensor.get_stream_profiles()
        )
        if sensor.is_color_sensor() or has_color_profile:
            color_sensor = sensor
            break
    if color_sensor is None:
        return {}
    names = {
        "enable_auto_exposure": rs.option.enable_auto_exposure,
        "exposure": rs.option.exposure,
        "gain": rs.option.gain,
        "enable_auto_white_balance": rs.option.enable_auto_white_balance,
        "white_balance": rs.option.white_balance,
        "brightness": rs.option.brightness,
        "contrast": rs.option.contrast,
        "saturation": rs.option.saturation,
        "sharpness": rs.option.sharpness,
        "gamma": rs.option.gamma,
        "hue": rs.option.hue,
    }
    output: dict[str, float | bool | None] = {}
    for name, option in names.items():
        if not color_sensor.supports(option):
            output[name] = None
            continue
        value = float(color_sensor.get_option(option))
        output[name] = bool(round(value)) if name.startswith("enable_") else value
    return output


def aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key in records[0]:
        values = [record[key] for record in records]
        array = np.asarray(values, dtype=np.float64)
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        output[key] = mean.tolist() if mean.ndim else float(mean)
        output[f"{key}_frame_std"] = std.tolist() if std.ndim else float(std)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--sample-frames", type=int, default=30)
    args = parser.parse_args()
    if args.warmup_frames < 1 or args.sample_frames < 1:
        raise ValueError("frame counts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipelines: dict[str, rs.pipeline] = {}
    profiles: dict[str, rs.pipeline_profile] = {}
    try:
        for name, serial in CAMERAS.items():
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            profiles[name] = pipeline.start(config)
            pipelines[name] = pipeline

        for _ in range(args.warmup_frames):
            for pipeline in pipelines.values():
                pipeline.wait_for_frames(5000).get_color_frame()

        report: dict[str, object] = {
            "captured_unix_s": time.time(),
            "resolution": [640, 480],
            "fps": 30,
            "warmup_frames": args.warmup_frames,
            "sample_frames": args.sample_frames,
            "cameras": {},
        }
        for name, pipeline in pipelines.items():
            records: list[dict[str, object]] = []
            last_rgb = None
            for _ in range(args.sample_frames):
                frame = pipeline.wait_for_frames(5000).get_color_frame()
                if not frame:
                    raise RuntimeError(f"{name}: missing RGB frame")
                last_rgb = np.asanyarray(frame.get_data()).copy()
                records.append(image_metrics(last_rgb))
            if last_rgb is None:
                raise RuntimeError(f"{name}: no captured frames")
            Image.fromarray(last_rgb).save(args.output_dir / f"current_{name}.png")
            report["cameras"][name] = {
                "serial": CAMERAS[name],
                "sensor_options": sensor_options(profiles[name]),
                "metrics": aggregate(records),
            }

        report_path = args.output_dir / "current_camera_metrics.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report_path)
    finally:
        for pipeline in pipelines.values():
            try:
                pipeline.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
