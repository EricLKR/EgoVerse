#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$BUNDLE_ROOT/tools/runtime_env.sh"
export PYTHONPATH="$BUNDLE_ROOT/training_code/EgoVerse:$PYTHONPATH"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-checkpoint}"
if [[ $# -gt 0 ]]; then shift; fi
exec "$PYTHON_BIN" "$BUNDLE_ROOT/robot/deploy_assembly_new1_hpt_joint_async_rtc.py" \
  --mode "$MODE" \
  --checkpoint-profile assembly-new1-hpt-absolute-reference-v3-30k \
  --query-frequency 30 \
  --gripper-prediction-source full-horizon \
  --gripper-initial-close-guard \
  --gripper-initial-close-consensus-horizons 2 \
  --gripper-initial-close-deadline-latch \
  --gripper-initial-close-deadline-max-wait-frames 10 \
  --gripper-first-close-pose-profile "$BUNDLE_ROOT/assets/first_grasp_pose_profile.json" \
  --gripper-first-close-max-nearest-rms-deg 3.0 \
  --gripper-first-close-envelope-margin-deg 1.0 \
  "$@"
