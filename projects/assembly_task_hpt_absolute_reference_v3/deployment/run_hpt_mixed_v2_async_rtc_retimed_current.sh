#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/wd10/tongxi/egoverse_piper/EgoVerse/emimic/bin/python}"
MODE="${1:-execute}"
if [[ $# -gt 0 ]]; then shift; fi

exec env PYTHON_BIN="$PYTHON_BIN" \
  "$BUNDLE_ROOT/run_hpt_mixed_v2_async_rtc_retimed.sh" "$MODE" \
  --checkpoint-profile assembly-new1-hpt-absolute-reference-v3-30k \
  --can-port can_leader \
  --seed 42 \
  --query-frequency 30 \
  --inference-steps 10 \
  --max-steps 5000 \
  --speed-percent 15 \
  --max-relative-target 5.0 \
  --max-gripper-relative-target 100 \
  --max-joint-command-step 1.0 \
  --max-joint-command-accel 0.30 \
  --max-initial-joint-delta-deg 5 \
  --joint-mapping-mode physical-identity \
  --j2-command-offset-deg 6.0 \
  --j3-command-offset-deg -1.0 \
  --j5-command-offset-deg -1.0 \
  --home-at-start \
  --home-at-end \
  --end-effector-z-bias-mm 0 \
  --gripper-threshold 0.5 \
  --gripper-prediction-source full-horizon \
  --gripper-control-mode latched \
  --gripper-effort 5000 \
  --gripper-close-confirm-frames 1 \
  --gripper-open-confirm-frames 1 \
  --gripper-min-close-hold-frames 60 \
  --gripper-min-open-hold-frames 60 \
  --gripper-open-reached-confirm-frames 3 \
  --gripper-reclose-consensus-horizons 1 \
  --gripper-close-feedback-threshold 10 \
  --gripper-open-feedback-threshold 90 \
  --no-gripper-terminal-open-after-release \
  --no-gripper-initial-close-guard \
  --no-gripper-initial-close-deadline-latch \
  --gripper-contact-detection \
  --gripper-contact-max-feedback 70 \
  --gripper-contact-stall-frames 15 \
  --gripper-contact-motion-tolerance 0.5 \
  --gripper-contact-min-close-frames 10 \
  --gripper-contact-min-drop 10 \
  --gripper-contact-preload-percent 100 \
  --gripper-release-on-any-open-in-active-horizon \
  --rtc-finish-active-plan-before-install \
  --no-rtc-replan-after-contact \
  --rtc-replan-interval 28 \
  --rtc-blend-actions 8 \
  --rtc-min-suffix-actions 8 \
  --rtc-max-result-age-ms 1000 \
  --retime-max-joint-step-deg 0.75 \
  --retime-max-feedback-lead-deg 1.50 \
  --retime-max-substeps-per-action 64 \
  --rtc-max-switch-joint-rms-deg 5.0 \
  --rtc-max-bridge-correction-deg 10.0 \
  --model-limit-soft-clamp-tolerance-deg 0.25 \
  --record-full-run-images \
  --device cuda \
  "$@"
