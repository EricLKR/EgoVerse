#!/usr/bin/env python
"""HPT-style decoded-joint RTC deployment for Mixed CamFrame V5.

The model adapter is the exact V5 Lightning ``ModelWrapper`` and its native
``forward_joint_deploy`` path.  The control layer is the separately audited
HPT-style RTC implementation: background inference, successful-command tick
time, intra-chunk arm smoothing, absolute-time cosine overlap, causal
velocity/acceleration/jerk limiting, and an independent binary gripper state
machine.  Earlier deployment files are imported read-only and remain intact.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import torch

import deploy_egoverse_pi05_mixed_camframe_v5_joint as v5_base
import deploy_egoverse_pi05_piper_v3_v4_hptstyle_rtc_joint as rtc


# The RTC module resolves ``base`` dynamically in parse_args/run_live.  Bind it
# to the additive V5 adapter without altering the historical V3/V4 source.
rtc.base = v5_base
rtc.legacy = v5_base.legacy
legacy = v5_base.legacy
_required_rtc_exports = (
    "EgoVersePI05PiperV3V4JointPolicy",
    "adjacent_joint_rms",
    "smooth_joint_action_chunk",
    "legacy",
    "parse_args",
)
_missing_rtc_exports = [
    name for name in _required_rtc_exports if not hasattr(v5_base, name)
]
if _missing_rtc_exports:
    raise ImportError(
        "Mixed CamFrame V5 adapter is missing RTC compatibility exports: "
        f"{_missing_rtc_exports}"
    )
logger = logging.getLogger(
    "deploy_egoverse_pi05_mixed_camframe_v5_hptstyle_rtc_joint"
)


def parse_args():
    return rtc.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    if args.list_checkpoint_profiles:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    actual_sha256 = legacy.sha256_file(args.checkpoint)
    if actual_sha256 != args.checkpoint_sha256:
        raise RuntimeError(
            "checkpoint SHA-256 mismatch: "
            f"expected {args.checkpoint_sha256}, got {actual_sha256}"
        )
    joint_limits, gripper_stroke_mm = legacy.load_calibration(
        args.calibration_json
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    policy = v5_base.EgoVersePI05MixedCamFrameV5JointPolicy(
        args, joint_limits, gripper_stroke_mm
    )
    if args.mode == "checkpoint":
        print(
            "PI05_MIXED_CAMFRAME_V5_HPTSTYLE_RTC_CHECKPOINT_OK "
            f"profile={args.checkpoint_profile} sha256={actual_sha256} "
            "robot_constructed=false camera_opened=false can_opened=false"
        )
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = legacy.JsonlLogger(
        args.log_dir
        / f"pi05_mixed_camframe_v5_hptstyle_rtc_{args.mode}_{timestamp}.jsonl"
    )
    try:
        log.write(
            {
                "event": "start",
                "mode": args.mode,
                "checkpoint_profile": args.checkpoint_profile,
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": actual_sha256,
                "checkpoint_contract": policy.checkpoint_report,
                "rtc": True,
                "rtc_mode": "post_decode_absolute_time_cosine_overlap",
                "rtc_space": "safe_physical_6d_joint_radians",
                "flow_sampler_rtc_guidance": False,
                "rtc_replan_interval": args.rtc_replan_interval,
                "rtc_blend_actions": args.rtc_blend_actions,
                "rtc_min_suffix_actions": args.rtc_min_suffix_actions,
                "rtc_max_result_age_ms": args.rtc_max_result_age_ms,
                "rtc_intrachunk_smoothing": args.rtc_intrachunk_smoothing,
                "rtc_intrachunk_window": args.rtc_intrachunk_window,
                "rtc_intrachunk_sigma": args.rtc_intrachunk_sigma,
                "rtc_causal_filter_space": "piper_driver_coordinates",
                "rtc_max_joint_command_step": args.max_joint_command_step,
                "rtc_max_joint_command_accel": args.max_joint_command_accel,
                "rtc_max_joint_command_jerk": args.rtc_max_joint_command_jerk,
                "rtc_command_deadband": args.rtc_command_deadband,
                "successful_action_tick_clock": True,
                "inference_thread": True,
                "wrist_color_normalization": args.wrist_color_normalization,
                "gripper_absolute_time_binary": True,
                "gripper_contact_detection": args.gripper_contact_detection,
                "gripper_min_close_hold_frames": (
                    args.gripper_min_close_hold_frames
                ),
                "gripper_min_open_hold_frames": args.gripper_min_open_hold_frames,
                "command_sent": False,
            }
        )
        if args.mode == "offline":
            rtc.run_offline_pair(args, policy, log, device)
        else:
            rtc.run_live(
                args,
                policy,
                log,
                device,
                joint_limits,
                gripper_stroke_mm,
                timestamp,
            )
    finally:
        log.close()
        logger.info("wrote Mixed CamFrame V5 RTC audit log to %s", log.path)


if __name__ == "__main__":
    main()
