#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Prepare an isolated runtime without opening cameras, CAN, or the robot.

Usage:
  prepare_runtime.sh \
    --lerobot-root /path/to/lerobot \
    [--pi05-base-weights /path/to/pi05_base_pytorch] \
    [--pi05-tokenizer /path/to/paligemma_tokenizer.model] \
    [--pi05-python-overlay /path/to/python_overlay]

The script refuses to overwrite an existing runtime/lerobot directory.  It
copies the supplied LeRobot tree, overlays only the audited Assembly New1
deployment files, and creates symlinks to large pi0.5 assets when provided.
EOF
}

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT=""
PI05_BASE_WEIGHTS=""
PI05_TOKENIZER=""
PI05_PYTHON_OVERLAY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lerobot-root)
      LEROBOT_ROOT="$2"
      shift 2
      ;;
    --pi05-base-weights)
      PI05_BASE_WEIGHTS="$2"
      shift 2
      ;;
    --pi05-tokenizer)
      PI05_TOKENIZER="$2"
      shift 2
      ;;
    --pi05-python-overlay)
      PI05_PYTHON_OVERLAY="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$LEROBOT_ROOT" || ! -d "$LEROBOT_ROOT/src/lerobot" ]]; then
  echo "--lerobot-root must contain src/lerobot" >&2
  exit 2
fi

RUNTIME_ROOT="$BUNDLE_ROOT/runtime/lerobot"
if [[ -e "$RUNTIME_ROOT" ]]; then
  echo "Refusing to overwrite existing runtime: $RUNTIME_ROOT" >&2
  exit 3
fi

mkdir -p "$BUNDLE_ROOT/runtime"
cp -a "$LEROBOT_ROOT" "$RUNTIME_ROOT"

POLICY_DST="$RUNTIME_ROOT/src/lerobot/policies/egoverse_hpt"
mkdir -p "$POLICY_DST" "$RUNTIME_ROOT/examples/piper"
cp "$BUNDLE_ROOT/src/lerobot/policies/egoverse_hpt/configuration_egoverse_hpt_joint.py" "$POLICY_DST/"
cp "$BUNDLE_ROOT/src/lerobot/policies/egoverse_hpt/modeling_egoverse_hpt_joint.py" "$POLICY_DST/"
cp "$BUNDLE_ROOT/robot/deploy_assembly_new1_hpt_joint.py" "$RUNTIME_ROOT/examples/piper/"
cp "$BUNDLE_ROOT/robot/deploy_assembly_new1_pi05_mixed_v2_joint.py" "$RUNTIME_ROOT/examples/piper/"
cp "$BUNDLE_ROOT/robot/deploy_egoverse_hpt_joint.py" "$RUNTIME_ROOT/examples/piper/"
cp "$BUNDLE_ROOT/robot/deploy_egoverse_pi05_joint.py" "$RUNTIME_ROOT/examples/piper/"
cp "$BUNDLE_ROOT/robot/deploy_egoverse_pi05_piper_v3_v4_joint.py" "$RUNTIME_ROOT/examples/piper/"

link_asset() {
  local source_path="$1"
  local destination_path="$2"
  local label="$3"
  if [[ ! -e "$source_path" ]]; then
    echo "$label does not exist: $source_path" >&2
    exit 4
  fi
  if [[ -e "$destination_path" || -L "$destination_path" ]]; then
    echo "Refusing to replace existing $label destination: $destination_path" >&2
    exit 5
  fi
  ln -s "$source_path" "$destination_path"
}

if [[ -n "$PI05_BASE_WEIGHTS" ]]; then
  link_asset "$PI05_BASE_WEIGHTS" "$BUNDLE_ROOT/pi05_runtime/pi05_base_pytorch" "pi0.5 base weights"
fi
if [[ -n "$PI05_TOKENIZER" ]]; then
  link_asset "$PI05_TOKENIZER" "$BUNDLE_ROOT/pi05_runtime/paligemma_tokenizer.model" "pi0.5 tokenizer"
fi
if [[ -n "$PI05_PYTHON_OVERLAY" ]]; then
  link_asset "$PI05_PYTHON_OVERLAY" "$BUNDLE_ROOT/pi05_runtime/python_overlay" "pi0.5 Python overlay"
fi

cat <<EOF
RUNTIME_PREPARED
bundle=$BUNDLE_ROOT
lerobot_runtime=$RUNTIME_ROOT
robot_opened=false
camera_opened=false
can_opened=false
EOF
