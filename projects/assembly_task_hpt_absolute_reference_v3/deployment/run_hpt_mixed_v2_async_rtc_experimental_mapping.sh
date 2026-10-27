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
  --checkpoint-profile assembly-new1-hpt-mixed-v2-30k \
  --query-frequency 30 \
  --gripper-prediction-source full-horizon \
  --joint-zero-offset-json "$BUNDLE_ROOT/assets/joint_zero_offset_home_experimental.json" \
  --allow-experimental-joint-mapping \
  "$@"
