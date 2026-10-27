#!/usr/bin/env python3
"""Measure wrist-camera alignment against a training Home reference.

This diagnostic opens only the D405 RGB stream.  It never imports the PIPER
driver, opens CAN, enables motors, or sends a robot command.  The reported
affine transform maps features in the training reference into the current
frame.  Perfect alignment is therefore tx=0 px, ty=0 px, rotation=0 deg, and
scale=1.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    BUNDLE_ROOT / "assets/wrist_alignment_reference_train_ep0000_frame0.png"
)
WRIST_SERIAL = "427622273943"


def capture_wrist(*, warmup_frames: int, sample_frames: int) -> np.ndarray:
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(WRIST_SERIAL)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(config)
    try:
        for _ in range(warmup_frames):
            frame = pipeline.wait_for_frames(5000).get_color_frame()
            if not frame:
                raise RuntimeError("D405 returned no RGB frame during warmup")
        frames: list[np.ndarray] = []
        for _ in range(sample_frames):
            frame = pipeline.wait_for_frames(5000).get_color_frame()
            if not frame:
                raise RuntimeError("D405 returned no RGB frame during sampling")
            frames.append(np.asanyarray(frame.get_data()).copy())
        # A temporal median suppresses sensor noise without changing geometry.
        return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    finally:
        pipeline.stop()


def estimate_alignment(reference_rgb: np.ndarray, current_rgb: np.ndarray) -> dict:
    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=7)
    reference_keypoints, reference_desc = orb.detectAndCompute(reference_gray, None)
    current_keypoints, current_desc = orb.detectAndCompute(current_gray, None)
    if reference_desc is None or current_desc is None:
        raise RuntimeError("insufficient wrist-image features for visual alignment")

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        reference_desc, current_desc, k=2
    )
    good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    if len(good) < 30:
        raise RuntimeError(f"too few wrist-image matches: {len(good)}")
    reference_points = np.float32(
        [reference_keypoints[match.queryIdx].pt for match in good]
    )
    current_points = np.float32(
        [current_keypoints[match.trainIdx].pt for match in good]
    )
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        reference_points,
        current_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=10000,
        confidence=0.999,
    )
    if affine is None or inlier_mask is None:
        raise RuntimeError("wrist-image affine alignment did not converge")
    inlier_count = int(inlier_mask.sum())
    if inlier_count < 25:
        raise RuntimeError(f"too few robust wrist-image inliers: {inlier_count}")

    scale = float(np.hypot(affine[0, 0], affine[0, 1]))
    rotation_deg = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
    tx_px = float(affine[0, 2])
    ty_px = float(affine[1, 2])
    return {
        "transform_direction": "training_reference_to_current",
        "tx_px": tx_px,
        "ty_px": ty_px,
        "translation_norm_px": float(np.hypot(tx_px, ty_px)),
        "rotation_deg": rotation_deg,
        "scale": scale,
        "good_match_count": len(good),
        "inlier_count": inlier_count,
        "affine_2x3": affine.tolist(),
        "target": {
            "tx_px": 0.0,
            "ty_px": 0.0,
            "rotation_deg": 0.0,
            "scale": 1.0,
        },
        "robot_or_can_opened": False,
        "motion_command_sent": False,
    }


def save_montage(
    reference_rgb: np.ndarray,
    current_rgb: np.ndarray,
    report: dict,
    output_path: Path,
) -> None:
    height, width = reference_rgb.shape[:2]
    canvas = Image.new("RGB", (width * 2, height + 42), "white")
    canvas.paste(Image.fromarray(reference_rgb), (0, 42))
    canvas.paste(Image.fromarray(current_rgb), (width, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), "TRAINING wrist reference", fill="black")
    draw.text(
        (width + 6, 8),
        (
            "CURRENT wrist  "
            f"tx={report['tx_px']:+.1f}px  ty={report['ty_px']:+.1f}px  "
            f"rot={report['rotation_deg']:+.2f}deg"
        ),
        fill="black",
    )
    canvas.save(output_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--current-image", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--sample-frames", type=int, default=15)
    args = parser.parse_args()
    if args.warmup_frames < 1 or args.sample_frames < 1:
        raise ValueError("frame counts must be positive")
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_rgb = np.asarray(Image.open(args.reference).convert("RGB"))
    if args.current_image is None:
        current_rgb = capture_wrist(
            warmup_frames=args.warmup_frames,
            sample_frames=args.sample_frames,
        )
        source = "live_D405_RGB"
    else:
        current_rgb = np.asarray(Image.open(args.current_image).convert("RGB"))
        source = str(args.current_image)
    if current_rgb.shape != reference_rgb.shape:
        raise RuntimeError(
            f"image shape mismatch: reference={reference_rgb.shape}, current={current_rgb.shape}"
        )

    report = estimate_alignment(reference_rgb, current_rgb)
    report.update(
        {
            "captured_unix_s": time.time(),
            "wrist_serial": WRIST_SERIAL,
            "reference": str(args.reference),
            "current_source": source,
        }
    )
    Image.fromarray(current_rgb).save(args.output_dir / "current_wrist.png")
    save_montage(
        reference_rgb,
        current_rgb,
        report,
        args.output_dir / "wrist_alignment_montage.jpg",
    )
    report_path = args.output_dir / "wrist_alignment.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
