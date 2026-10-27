#!/usr/bin/env python3
"""Summarize and visualize held-out 25k versus 30k V3 evaluations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=(25000, 30000))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def episode_reports(directory: Path) -> dict[str, dict]:
    reports = {}
    for path in sorted(directory.glob("*_metrics.json")):
        if path.name == "summary_metrics.json":
            continue
        report = load_json(path)
        reports[str(report["episode"])] = report
    return reports


def safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def shared_representative(
    roots: dict[int, Path], embodiment: str, metric: str
) -> str:
    per_step = {
        step: episode_reports(root / embodiment) for step, root in roots.items()
    }
    names = set.intersection(*(set(items) for items in per_step.values()))
    if not names:
        raise RuntimeError(f"No shared {embodiment} episodes across checkpoints")
    scores = {
        name: float(np.mean([per_step[step][name][metric] for step in roots]))
        for name in names
    }
    median = float(np.median(list(scores.values())))
    return min(scores, key=lambda name: abs(scores[name] - median))


def make_metric_plot(roots: dict[int, Path], destination: Path) -> None:
    steps = list(roots)
    piper = [load_json(roots[step] / "piper/summary_metrics.json") for step in steps]
    human = [load_json(roots[step] / "human/summary_metrics.json") for step in steps]
    panels = [
        ("PIPER joint MAE", [x["joint_mae_deg"] for x in piper], "deg", False),
        (
            "PIPER FK position MAE",
            [100 * x["cmd_ee_pose_position_mae_m"] for x in piper],
            "cm",
            False,
        ),
        (
            "PIPER FK rotation MAE",
            [x["cmd_ee_pose_rotation_mae_deg"] for x in piper],
            "deg",
            False,
        ),
        (
            "PIPER gripper accuracy",
            [100 * x["gripper_accuracy"] for x in piper],
            "%",
            True,
        ),
        (
            "Human position MAE",
            [100 * x["position_mae_m"] for x in human],
            "cm",
            False,
        ),
        (
            "Human rotation MAE",
            [x["rotation_geodesic_mae_deg"] for x in human],
            "deg",
            False,
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    colors = ("#4c78a8", "#f58518")
    for axis, (title, values, unit, higher_better) in zip(axes.flat, panels):
        bars = axis.bar(
            [str(step // 1000) + "k" for step in steps], values, color=colors
        )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        direction = "higher is better" if higher_better else "lower is better"
        axis.text(
            0.5,
            -0.17,
            direction,
            transform=axis.transAxes,
            ha="center",
            fontsize=8,
        )
    fig.suptitle("Assembly New1 HPT Absolute-Reference V3 — held-out test qf=30")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def label_image(image: Image.Image, label: str) -> Image.Image:
    result = Image.new("RGB", (image.width, image.height + 44), "white")
    result.paste(image.convert("RGB"), (0, 44))
    draw = ImageDraw.Draw(result)
    draw.text((12, 12), label, fill="black", font=ImageFont.load_default())
    return result


def side_by_side(paths: list[tuple[str, Path]], destination: Path) -> None:
    images = [label_image(Image.open(path), label) for label, path in paths]
    target_height = max(image.height for image in images)
    resized = []
    for image in images:
        if image.height != target_height:
            width = round(image.width * target_height / image.height)
            image = image.resize((width, target_height), Image.Resampling.LANCZOS)
        resized.append(image)
    canvas = Image.new(
        "RGB", (sum(image.width for image in resized), target_height), "white"
    )
    x = 0
    for image in resized:
        canvas.paste(image, (x, 0))
        x += image.width
    canvas.save(destination, quality=95)


def main() -> int:
    args = parse_args()
    roots = {step: args.output_root / f"step{step}" for step in args.steps}
    for root in roots.values():
        for required in (
            root / "piper/summary_metrics.json",
            root / "human/summary_metrics.json",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)

    piper_episode = shared_representative(roots, "piper", "joint_mae_deg")
    human_episode = shared_representative(roots, "human", "position_mae_m")
    piper_stem = safe_stem(piper_episode)
    human_stem = safe_stem(human_episode)

    make_metric_plot(roots, args.output_root / "checkpoint_metric_comparison.png")
    side_by_side(
        [
            (
                f"{step // 1000}k — {piper_episode}",
                root / "piper" / f"{piper_stem}_joint_eval.png",
            )
            for step, root in roots.items()
        ],
        args.output_root / "piper_same_episode_comparison.png",
    )
    side_by_side(
        [
            (
                f"{step // 1000}k — {human_episode}",
                root / "human" / f"{human_stem}_human_episode_eval.png",
            )
            for step, root in roots.items()
        ],
        args.output_root / "human_same_episode_comparison.png",
    )

    report = {
        "split": "test",
        "query_frequency": 30,
        "action_horizon_frames": 30,
        "num_inference_steps": 10,
        "piper_representative_episode": piper_episode,
        "human_representative_episode": human_episode,
        "checkpoints": {
            str(step): {
                "piper": load_json(root / "piper/summary_metrics.json"),
                "human": load_json(root / "human/summary_metrics.json"),
            }
            for step, root in roots.items()
        },
    }
    (args.output_root / "checkpoint_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
