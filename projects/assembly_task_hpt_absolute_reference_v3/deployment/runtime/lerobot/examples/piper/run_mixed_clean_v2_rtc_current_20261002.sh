#!/usr/bin/env bash
set -euo pipefail

# User-confirmed Mixed Clean V2 decoded-joint RTC command, recorded 2026-10-02.
#
# This is a high-dynamic real-robot profile:
#   speed_percent=25
#   max_relative_target=5.5
#   max_joint_command_step=5.5
#   max_joint_command_accel=3.5
#
# Before running, activate the existing environment, set PYTHONPATH, stop all
# other robot-control processes, verify can_follower and both cameras, and keep
# the emergency stop accessible. Running this file sends real motion commands.

python examples/piper/deploy_egoverse_hpt_joint_rtc.py \
  --mode execute \
  --checkpoint-profile mixed-clean-v2 \
  --can-port can_follower \
  --seed 42 \
  --query-frequency 30 \
  --inference-steps 10 \
  --max-steps 10000 \
  --speed-percent 25 \
  --max-relative-target 5.5 \
  --max-gripper-relative-target 100 \
  --max-joint-command-step 5.5 \
  --max-joint-command-accel 3.5 \
  --max-initial-joint-delta-deg 5 \
  --gripper-threshold 0.5 \
  --gripper-prediction-source full-horizon \
  --gripper-control-mode latched \
  --gripper-close-confirm-frames 3 \
  --gripper-open-confirm-frames 3 \
  --gripper-min-close-hold-frames 60 \
  --gripper-min-open-hold-frames 60 \
  --gripper-close-feedback-threshold 10 \
  --gripper-open-feedback-threshold 90 \
  --no-gripper-terminal-open-after-release \
  --gripper-contact-detection \
  --gripper-contact-max-feedback 70 \
  --gripper-contact-stall-frames 15 \
  --gripper-contact-motion-tolerance 0.5 \
  --gripper-contact-min-close-frames 10 \
  --gripper-contact-min-drop 10 \
  --rtc-replan-interval 10 \
  --rtc-blend-actions 8 \
  --rtc-min-suffix-actions 8 \
  --rtc-max-result-age-ms 500 \
  --no-feedback-tracking \
  --wrist-color-normalization startup \
  --wrist-color-calibration-frames 30 \
  --wrist-color-gain-min 0.85 \
  --wrist-color-gain-max 1.20 \
  --device cuda \
  --allow-motion
