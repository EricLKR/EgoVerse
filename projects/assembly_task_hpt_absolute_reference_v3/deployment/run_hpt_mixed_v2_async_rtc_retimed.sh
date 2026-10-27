#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$BUNDLE_ROOT/tools/runtime_env.sh"
export PYTHONPATH="$BUNDLE_ROOT/training_code/EgoVerse:$PYTHONPATH"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-checkpoint}"
if [[ $# -gt 0 ]]; then shift; fi
exec "$PYTHON_BIN" \
  "$BUNDLE_ROOT/robot/deploy_assembly_new1_hpt_joint_async_rtc_retimed.py" \
  --mode "$MODE" \
  "$@"
