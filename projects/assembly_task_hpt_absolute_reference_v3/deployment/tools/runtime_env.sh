#!/usr/bin/env bash
# Source this file from a run script. It only prepares Python import paths.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source runtime_env.sh instead of executing it" >&2
  exit 2
fi

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_LEROBOT="$BUNDLE_ROOT/runtime/lerobot"
if [[ ! -d "$RUNTIME_LEROBOT/src/lerobot" ]]; then
  echo "Missing isolated runtime. Run tools/prepare_runtime.sh first." >&2
  return 3
fi

export PYTHONDONTWRITEBYTECODE=1
export PIPER_RECORDING_EXTRINSICS_JSON="$BUNDLE_ROOT/assets/piper_front_extrinsics_recording_20261016.json"
unset PIPER_EXTRINSICS_JSON || true
export PYTHONPATH="$BUNDLE_ROOT/pi05_runtime/python_overlay:$BUNDLE_ROOT/training_code/EgoVerse_pi05/external/openpi/packages/openpi-client/src:$BUNDLE_ROOT/training_code/EgoVerse_pi05/external/openpi/src:$BUNDLE_ROOT/training_code/EgoVerse_pi05:$BUNDLE_ROOT/training_code/EgoVerse:$RUNTIME_LEROBOT/src:$RUNTIME_LEROBOT/examples/piper${PYTHONPATH:+:$PYTHONPATH}"
