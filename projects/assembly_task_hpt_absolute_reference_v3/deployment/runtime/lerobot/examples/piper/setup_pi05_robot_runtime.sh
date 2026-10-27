#!/usr/bin/env bash
set -euo pipefail

# Build a private pi0.5 inference overlay owned by kerun.  The shared emimic
# environment under /mnt/wd10/tongxi is read but never modified.
RUNTIME_DIR="${PI05_RUNTIME_DIR:-/mnt/wd10/kerun/pi05_runtime}"
SHARED_PYTHON="/mnt/wd10/tongxi/egoverse_piper/EgoVerse/emimic/bin/python"
PATCH_ROOT="/mnt/wd10/kerun/egoverse_piper/EgoVerse/external/openpi/src/openpi/models_pytorch/transformers_replace/models"
TOKENIZER_URL="https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"

if [[ "$RUNTIME_DIR" != /mnt/wd10/kerun/* ]]; then
  echo "Refusing runtime outside /mnt/wd10/kerun: $RUNTIME_DIR" >&2
  exit 2
fi
if [[ ! -x "$SHARED_PYTHON" ]]; then
  echo "Shared Python is missing: $SHARED_PYTHON" >&2
  exit 2
fi
for required in \
  "$PATCH_ROOT/gemma/configuration_gemma.py" \
  "$PATCH_ROOT/gemma/modeling_gemma.py" \
  "$PATCH_ROOT/paligemma/modeling_paligemma.py" \
  "$PATCH_ROOT/siglip/modeling_siglip.py" \
  "$PATCH_ROOT/siglip/check.py"; do
  [[ -f "$required" ]] || { echo "Missing OpenPI transformers patch: $required" >&2; exit 2; }
done

mkdir -p "$RUNTIME_DIR"
TRANSFORMERS_SOURCE="$($SHARED_PYTHON -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')"
if [[ ! -d "$RUNTIME_DIR/transformers" ]]; then
  cp -a "$TRANSFORMERS_SOURCE" "$RUNTIME_DIR/transformers"
fi

cp "$PATCH_ROOT/gemma/configuration_gemma.py" \
  "$RUNTIME_DIR/transformers/models/gemma/configuration_gemma.py"
cp "$PATCH_ROOT/gemma/modeling_gemma.py" \
  "$RUNTIME_DIR/transformers/models/gemma/modeling_gemma.py"
cp "$PATCH_ROOT/paligemma/modeling_paligemma.py" \
  "$RUNTIME_DIR/transformers/models/paligemma/modeling_paligemma.py"
cp "$PATCH_ROOT/siglip/modeling_siglip.py" \
  "$RUNTIME_DIR/transformers/models/siglip/modeling_siglip.py"
cp "$PATCH_ROOT/siglip/check.py" \
  "$RUNTIME_DIR/transformers/models/siglip/check.py"

if ! PYTHONPATH="$RUNTIME_DIR" "$SHARED_PYTHON" -c 'import sentencepiece' 2>/dev/null; then
  WHEEL_DIR="$RUNTIME_DIR/.wheels"
  mkdir -p "$WHEEL_DIR"
  pip3 download \
    --disable-pip-version-check \
    --no-deps \
    --only-binary=:all: \
    --implementation cp \
    --python-version 311 \
    --abi cp311 \
    --platform manylinux_2_17_x86_64 \
    --dest "$WHEEL_DIR" \
    'sentencepiece==0.2.0'
  SENTENCEPIECE_WHEEL="$(find "$WHEEL_DIR" -maxdepth 1 -type f -name 'sentencepiece-0.2.0-*.whl' -print -quit)"
  [[ -n "$SENTENCEPIECE_WHEEL" ]] || { echo "sentencepiece cp311 wheel download failed" >&2; exit 2; }
  "$SHARED_PYTHON" -c \
    'import pathlib, sys, zipfile; wheel=pathlib.Path(sys.argv[1]); target=pathlib.Path(sys.argv[2]); zipfile.ZipFile(wheel).extractall(target)' \
    "$SENTENCEPIECE_WHEEL" "$RUNTIME_DIR"
fi

if [[ ! -s "$RUNTIME_DIR/paligemma_tokenizer.model" ]]; then
  curl --fail --location --retry 3 \
    "$TOKENIZER_URL" \
    --output "$RUNTIME_DIR/paligemma_tokenizer.model"
fi

PYTHONPATH="$RUNTIME_DIR:/mnt/wd10/kerun/lerobot/src" \
  "$SHARED_PYTHON" -c \
  'import sentencepiece, transformers; from transformers.models.siglip import check; assert check.check_whether_transformers_replace_is_installed_correctly(); print("PI05_RUNTIME_OK", transformers.__version__)'

echo "pi0.5 private runtime ready: $RUNTIME_DIR"
