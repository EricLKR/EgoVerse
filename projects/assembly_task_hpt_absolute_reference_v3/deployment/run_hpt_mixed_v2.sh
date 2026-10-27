#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$BUNDLE_ROOT/tools/runtime_env.sh"
# Keep the HPT Lightning classes ahead of the separate pi0.5 source snapshot.
# Both trees expose an ``egomimic`` package, so relying on one shared ordering
# can silently reconstruct a checkpoint with the wrong source tree.
export PYTHONPATH="$BUNDLE_ROOT/training_code/EgoVerse:$PYTHONPATH"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-checkpoint}"
if [[ $# -gt 0 ]]; then shift; fi
exec "$PYTHON_BIN" "$BUNDLE_ROOT/robot/deploy_assembly_new1_hpt_joint.py" \
  --mode "$MODE" \
  --checkpoint-profile assembly-new1-hpt-mixed-v2-30k \
  "$@"
