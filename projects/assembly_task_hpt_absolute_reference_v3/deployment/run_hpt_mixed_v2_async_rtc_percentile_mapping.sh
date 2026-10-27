#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-checkpoint}"
if [[ $# -gt 0 ]]; then shift; fi

# Experimental A/B deployment profile for using the collection-time
# follower-percentile -> leader-percentile conversion.  The stable launcher
# remains unchanged and continues to use physical-identity mapping.
exec "$BUNDLE_ROOT/run_hpt_mixed_v2_async_rtc.sh" "$MODE" \
  "$@" \
  --joint-mapping-mode recorded-percentile-affine
