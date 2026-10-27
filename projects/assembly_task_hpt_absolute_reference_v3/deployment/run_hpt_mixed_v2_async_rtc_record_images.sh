#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add only the non-blocking, per-control-frame three-camera recorder. All
# policy, RTC, calibration and motion arguments remain owned by the audited
# Mixed V2 launcher and by explicit command-line options.
exec "$BUNDLE_ROOT/run_hpt_mixed_v2_async_rtc.sh" "$@" \
  --record-full-run-images \
  --full-run-image-queue-size 64 \
  --full-run-image-jpeg-quality 90
