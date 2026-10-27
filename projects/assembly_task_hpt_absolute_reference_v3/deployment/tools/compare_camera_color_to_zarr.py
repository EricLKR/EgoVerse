#!/usr/bin/env python3
"""Compare current deployment RGB frames with sampled training RGB frames."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import zarr


CAMERA_KEYS = {
    "front": "images.front_1",
    "wrist": "images.wrist_1",
    "side": "images.side_1",
}


def metrics(rgb: np.ndarray) -> dict[str, float | list[float]]:
    rgb_f = rgb.astype(np.float32)
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32)
    return {
        "rgb_mean": rgb_f.mean(axis=(0, 1)).tolist(),
        "rgb_std": rgb_f.std(axis=(0, 1)).tolist(),
        "hsv_saturation_mean": float(hsv[..., 1].mean()),
        "hsv_saturation_median": float(np.median(hsv[..., 1])),
        "hsv_saturation_p90": float(np.percentile(hsv[..., 1], 90)),
        "hsv_value_mean": float(hsv[..., 2].mean()),
        "red_blue_ratio": float((rgb_f[..., 0].mean() + 1.0) / (rgb_f[..., 2].mean() + 1.0)),
        "dark_fraction": float((hsv[..., 2] < 25).mean()),
        "highlight_fraction": float((hsv[..., 2] > 250).mean()),
    }


def decode_jpeg(value: object) -> np.ndarray:
    while isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if not isinstance(value, (bytes, bytearray, np.bytes_)):
        raise TypeError(f"unsupported encoded image value: {type(value)!r}")
    data = bytes(value)
    return np.asarray(Image.open(BytesIO(data)).convert("RGB"))


def aggregate(records: list[dict[str, float | list[float]]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key in records[0]:
        array = np.asarray([record[key] for record in records], dtype=np.float64)
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        p05 = np.percentile(array, 5, axis=0)
        p95 = np.percentile(array, 95, axis=0)
        output[key] = {
            "mean": mean.tolist() if mean.ndim else float(mean),
            "std": std.tolist() if std.ndim else float(std),
            "p05": p05.tolist() if p05.ndim else float(p05),
            "p95": p95.tolist() if p95.ndim else float(p95),
        }
    return output


def compare_scalar(current: float, distribution: dict[str, float]) -> dict[str, float | bool]:
    mean = float(distribution["mean"])
    std = float(distribution["std"])
    return {
        "current": current,
        "training_mean": mean,
        "training_std": std,
        "difference": current - mean,
        "relative_difference": (current - mean) / max(abs(mean), 1e-9),
        "z_score": (current - mean) / max(std, 1e-9),
        "inside_training_p05_p95": bool(
            float(distribution["p05"]) <= current <= float(distribution["p95"])
        ),
    }


def labeled_tile(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    tile = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size[0], size[1] + 28), "white")
    canvas.paste(tile, (0, 28))
    ImageDraw.Draw(canvas).text((6, 7), label, fill="black", font=ImageFont.load_default())
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-episodes", type=int, default=30)
    parser.add_argument("--frames-per-episode", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = sorted(args.dataset.glob("*train*.zarr"))
    if not episodes:
        raise FileNotFoundError(f"no training episodes in {args.dataset}")
    episode_indices = np.linspace(
        0, len(episodes) - 1, min(args.sample_episodes, len(episodes)), dtype=int
    )
    sampled_episodes = [episodes[index] for index in episode_indices]

    current_report = json.loads(
        (args.current_dir / "current_camera_metrics.json").read_text(encoding="utf-8")
    )
    output: dict[str, object] = {
        "dataset": str(args.dataset),
        "training_episode_count": len(episodes),
        "sampled_episode_count": len(sampled_episodes),
        "frames_per_episode": args.frames_per_episode,
        "cameras": {},
        "caveat": (
            "Whole-frame color statistics include scene geometry and object placement. "
            "They detect major input-domain shifts but are not a camera-only calibration measurement."
        ),
    }
    montage_rows: list[Image.Image] = []
    for camera, key in CAMERA_KEYS.items():
        records: list[dict[str, float | list[float]]] = []
        representative: list[tuple[Image.Image, str]] = []
        for episode_position, episode in enumerate(sampled_episodes):
            array = zarr.open_array(str(episode / key), mode="r")
            frame_count = int(array.shape[0])
            frame_indices = np.linspace(
                0,
                frame_count - 1,
                min(args.frames_per_episode, frame_count),
                dtype=int,
            )
            for frame_index in frame_indices:
                rgb = decode_jpeg(array[int(frame_index)])
                records.append(metrics(rgb))
                if len(representative) < 4 and episode_position in {
                    0,
                    len(sampled_episodes) // 3,
                    2 * len(sampled_episodes) // 3,
                    len(sampled_episodes) - 1,
                }:
                    representative.append(
                        (Image.fromarray(rgb), f"train {episode.name[-11:-5]} f{int(frame_index)}")
                    )

        training = aggregate(records)
        current_metrics = current_report["cameras"][camera]["metrics"]
        comparisons = {
            metric: compare_scalar(float(current_metrics[metric]), training[metric])
            for metric in (
                "hsv_saturation_mean",
                "hsv_saturation_median",
                "hsv_saturation_p90",
                "hsv_value_mean",
                "red_blue_ratio",
                "dark_fraction",
                "highlight_fraction",
            )
        }
        output["cameras"][camera] = {
            "training_sample_count": len(records),
            "training_distribution": training,
            "current_metrics": current_metrics,
            "comparison": comparisons,
        }

        current_image = Image.open(args.current_dir / f"current_{camera}.png").convert("RGB")
        tiles = [labeled_tile(current_image, f"CURRENT {camera}", (320, 240))]
        tiles.extend(labeled_tile(image, label, (320, 240)) for image, label in representative[:4])
        row = Image.new("RGB", (sum(tile.width for tile in tiles), max(tile.height for tile in tiles)), "white")
        x = 0
        for tile in tiles:
            row.paste(tile, (x, 0))
            x += tile.width
        montage_rows.append(row)

    montage = Image.new(
        "RGB",
        (max(row.width for row in montage_rows), sum(row.height for row in montage_rows)),
        "white",
    )
    y = 0
    for row in montage_rows:
        montage.paste(row, (0, y))
        y += row.height
    montage.save(args.output_dir / "current_vs_training_montage.jpg", quality=94)
    report_path = args.output_dir / "camera_color_comparison.json"
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
