#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$BUNDLE_ROOT/tools/runtime_env.sh"
# This checkpoint must be reconstructed from the exact HPT source snapshot,
# not from the independent pi0.5 tree which also owns an ``egomimic`` package.
export PYTHONPATH="$BUNDLE_ROOT/training_code/EgoVerse:$PYTHONPATH"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-checkpoint}"
if [[ $# -gt 0 ]]; then shift; fi
exec "$PYTHON_BIN" "$BUNDLE_ROOT/robot/deploy_assembly_new1_hpt_joint.py" \
  --mode "$MODE" \
  --checkpoint-profile assembly-new1-hpt-expert-action-v3-30k \
  "$@"
