#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model-free Home paired with the recorded-percentile-affine deployment mode.
# This restores the collection leader pose stored in reset_control.json rather
# than replaying the follower's physical-angle Home unchanged.
exec "$BUNDLE_ROOT/run_home_only.sh" \
  "$@" \
  --initial-pose-json "$BUNDLE_ROOT/assets/initial_pose_original_leader.json"
