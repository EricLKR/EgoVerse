#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$BUNDLE_ROOT/tools/runtime_env.sh"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  : # Respect an explicit caller override.
elif [[ -x /mnt/wd10/tongxi/egoverse_piper/EgoVerse/emimic/bin/python ]]; then
  PYTHON_BIN=/mnt/wd10/tongxi/egoverse_piper/EgoVerse/emimic/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "No usable Python interpreter found; set PYTHON_BIN explicitly" >&2
  exit 4
fi
exec "$PYTHON_BIN" "$BUNDLE_ROOT/robot/home_assembly_new1.py" "$@"
