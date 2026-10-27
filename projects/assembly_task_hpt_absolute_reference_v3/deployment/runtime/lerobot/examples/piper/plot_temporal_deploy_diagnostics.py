#!/usr/bin/env python
"""Render saved Piper inference diagnostics after a deployment run.

This script is intentionally post-run only. It never imports robot drivers,
opens CAN, reads cameras, or sends commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def format_phase(manifest: dict) -> str:
    phase = manifest.get("temporal_alignment") or manifest.get("temporal_alignment_rejection")
    if not isinstance(phase, dict):
        return "No temporal alignment metadata"
    offset = phase.get("aligned_local_offset")
    if offset is None:
        best = phase.get("best_candidate") or {}
        offset = best.get("local_offset")
    xyz = phase.get("xyz_distance_m")
    if xyz is None:
        best = phase.get("best_candidate") or {}
        xyz = best.get("xyz_distance_m")
    joint = phase.get("joint_rms")
    if joint is None:
        best = phase.get("best_candidate") or {}
        joint = best.get("joint_rms")
    parts = [
        f"event={manifest.get('event')}",
        f"absolute_step={phase.get('absolute_index')}",
        f"aligned_offset={offset}",
    ]
    if xyz is not None:
        parts.append(f"xyz_distance={float(xyz) * 1000:.1f} mm")
    if joint is not None:
        parts.append(f"joint_rms={float(joint):.2f}")
    if phase.get("output_joint_rms") is not None:
        parts.append(f"blended_output_rms={float(phase['output_joint_rms']):.2f}")
    if phase.get("reason") is not None:
        parts.append(f"reason={phase['reason']}")
    return " | ".join(parts)


def render_manifest(manifest_path: Path, output_dir: Path) -> Path:
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    base = manifest_path.parent
    front = np.asarray(Image.open(base / manifest["front_png"]).convert("RGB"))
    wrist = np.asarray(Image.open(base / manifest["wrist_png"]).convert("RGB"))
    with np.load(base / manifest["predictions_npz"], allow_pickle=False) as archive:
        full = archive["full_model_chunk_unnormalized"]
        query_indices = archive["query_indices"]
        selected_xyz = archive["selected_xyz_base_m"]
        selected_gripper_raw = archive["selected_gripper_raw"]
        selected_gripper_binary = archive["selected_gripper_binary"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].imshow(front)
    axes[0, 0].set_title("Front camera — exact inference frame")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(wrist)
    axes[0, 1].set_title("Wrist camera — exact inference frame")
    axes[0, 1].axis("off")

    xyz_axis = axes[1, 0]
    if full.ndim == 2 and full.shape[1] >= 3:
        model_steps = np.arange(len(full))
        for dim, label, color in zip(range(3), ("x", "y", "z"), ("tab:red", "tab:green", "tab:blue")):
            xyz_axis.plot(model_steps, full[:, dim] * 1000.0, color=color, label=label)
            if len(query_indices) == len(selected_xyz):
                xyz_axis.scatter(
                    query_indices,
                    selected_xyz[:, dim] * 1000.0,
                    color=color,
                    s=18,
                    marker="o",
                )
    xyz_axis.set_title("Predicted Cartesian trajectory (lines=100 model steps, dots=selected IK steps)")
    xyz_axis.set_xlabel("model step")
    xyz_axis.set_ylabel("base-frame position [mm]")
    xyz_axis.grid(True, alpha=0.3)
    xyz_axis.legend(loc="best")

    gripper_axis = axes[1, 1]
    if full.ndim == 2 and full.shape[1] >= 10:
        gripper_axis.plot(full[:, 9], color="tab:purple", label="raw 100-step prediction")
    if len(query_indices) == len(selected_gripper_raw):
        gripper_axis.scatter(
            query_indices,
            selected_gripper_raw,
            color="tab:red",
            s=22,
            label="selected raw prediction",
        )
        gripper_axis.step(
            query_indices,
            selected_gripper_binary,
            where="mid",
            color="black",
            alpha=0.7,
            label="binary command",
        )
    gripper_axis.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="threshold")
    gripper_axis.set_title("Gripper prediction")
    gripper_axis.set_xlabel("model step")
    gripper_axis.set_ylabel("open fraction / binary command")
    gripper_axis.grid(True, alpha=0.3)
    gripper_axis.legend(loc="best")

    fig.suptitle(
        f"Piper inference diagnostic — frame {manifest.get('frame')}, prediction "
        f"{manifest.get('prediction_id')}\n{format_phase(manifest)}",
        fontsize=13,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{manifest_path.stem}_summary.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    output_dir = (args.output_dir or (run_dir / "plots")).resolve()
    manifests = sorted(run_dir.glob("frame_*.json"))
    if not manifests:
        raise FileNotFoundError(f"no frame_*.json diagnostic manifests in {run_dir}")
    for manifest in manifests:
        print(render_manifest(manifest, output_dir))


if __name__ == "__main__":
    main()
