#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-execute}"
if [[ $# -gt 0 ]]; then shift; fi

exec "$BUNDLE_ROOT/run_hpt_mixed_v2_async_rtc_retimed_current.sh" "$MODE" \
  --retime-advance-mode control-frame \
  "$@"
