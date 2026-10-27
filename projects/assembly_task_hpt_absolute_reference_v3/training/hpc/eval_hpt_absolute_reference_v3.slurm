#!/bin/bash -l
#SBATCH --account=naiss2026-3-376-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --job-name=asm-new1-hpt-abs-v3-eval
#SBATCH --output=/nobackup/proj/disk/naiss2024-5-164/personal/kerun/egoverse_piper_run/outputs/logs/%x-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-5-164/personal/kerun/egoverse_piper_run/outputs/logs/%x-%j.err

set -euo pipefail
umask 077

TRAIN_JOB_ID=${TRAIN_JOB_ID:-4283532}
ROOT=/nobackup/proj/disk/naiss2024-5-164/personal/kerun/egoverse_piper_run
EXPERIMENT_BUNDLE="$ROOT/code/assembly_task_new1_hpt_absolute_reference_v3"
EVAL_BUNDLE="$ROOT/code/assembly_task_new1_eval_v2"
BASE_BUNDLE="$ROOT/code/assembly_task_new1_training_v2"
CODE="$ROOT/code/egoverse_piper_assembly_hpt_new1_v2/EgoVerse"
IMAGE="$ROOT/env/pytorch_25.06-py3.sif"
PYTHON="$ROOT/env/emimic/bin/python"
TEST_CONFIG="$CODE/egomimic/hydra_configs/data/assembly_task_new1_mixed_90_5_5_test.yaml"
TRAIN_RUN="$ROOT/outputs/runs/assembly_new1_hpt_absolute_reference_v3/formal_${TRAIN_JOB_ID}"
OUTPUT_ROOT="$ROOT/outputs/evaluations/assembly_new1_hpt_absolute_reference_v3/compare_25k_30k_${SLURM_JOB_ID}"

export PIPER_HPC="$ROOT"
export PIPER_RECORDING_EXTRINSICS_JSON="$BASE_BUNDLE/assets/piper_front_extrinsics_recording_20261016.json"
unset PIPER_EXTRINSICS_JSON || true
export HF_HOME="$ROOT/cache/huggingface"
export TORCH_HOME="$ROOT/cache/torch"
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export ASSEMBLY_NEW1_EVAL_SCRIPT_DIR="$EVAL_BUNDLE/scripts"

test -s "$TRAIN_RUN/FORMAL_COMPLETED"
grep -qx "job_id=${TRAIN_JOB_ID}" "$TRAIN_RUN/FORMAL_COMPLETED"
grep -qx "final_step=30000" "$TRAIN_RUN/FORMAL_COMPLETED"
test -s "$TRAIN_RUN/checkpoints.sha256"
(cd "$TRAIN_RUN" && sha256sum -c checkpoints.sha256)
test -s "$TEST_CONFIG"
test -s "$EXPERIMENT_BUNDLE/eval/piper_episode_eval_v3.py"
test -s "$EXPERIMENT_BUNDLE/eval/human_episode_eval_v3.py"
test -s "$EXPERIMENT_BUNDLE/eval/compare_checkpoints.py"
mkdir -p "$OUTPUT_ROOT" "$ROOT/outputs/logs" "$TMPDIR"

run_container() {
  apptainer exec --nv --bind "$ROOT:$ROOT" "$IMAGE" "$@"
}

cp "$0" "$OUTPUT_ROOT/eval_hpt_absolute_reference_v3.slurm"
sha256sum "$TEST_CONFIG" \
  "$EVAL_BUNDLE/scripts/piper_episode_eval.py" \
  "$EVAL_BUNDLE/scripts/human_episode_eval.py" \
  "$EXPERIMENT_BUNDLE/eval/piper_episode_eval_v3.py" \
  "$EXPERIMENT_BUNDLE/eval/human_episode_eval_v3.py" \
  "$EXPERIMENT_BUNDLE/eval/compare_checkpoints.py" \
  > "$OUTPUT_ROOT/evaluation_code.sha256"

cd "$CODE"
run_container "$PYTHON" -m py_compile \
  "$EXPERIMENT_BUNDLE/eval/piper_episode_eval_v3.py" \
  "$EXPERIMENT_BUNDLE/eval/human_episode_eval_v3.py" \
  "$EXPERIMENT_BUNDLE/eval/compare_checkpoints.py"

for STEP in 25000 30000; do
  CHECKPOINT="$TRAIN_RUN/train/checkpoints/step_step=${STEP}.ckpt"
  test -s "$CHECKPOINT"
  mkdir -p "$OUTPUT_ROOT/step${STEP}/piper" "$OUTPUT_ROOT/step${STEP}/human"
  sha256sum "$CHECKPOINT" > "$OUTPUT_ROOT/step${STEP}/checkpoint.sha256"

  run_container "$PYTHON" "$EXPERIMENT_BUNDLE/eval/piper_episode_eval_v3.py" \
    --checkpoint "$CHECKPOINT" --data-config "$TEST_CONFIG" \
    --output-dir "$OUTPUT_ROOT/step${STEP}/piper" --split test --allow-test \
    --query-frequency 30 --action-horizon-frames 30 --batch-size 8 \
    --num-inference-steps 10 --device cuda --seed 42

  run_container "$PYTHON" "$EXPERIMENT_BUNDLE/eval/human_episode_eval_v3.py" \
    --checkpoint "$CHECKPOINT" --data-config "$TEST_CONFIG" \
    --output-dir "$OUTPUT_ROOT/step${STEP}/human" --split test --allow-test \
    --query-frequency 30 --action-horizon-frames 30 --batch-size 8 \
    --num-inference-steps 10 --device cuda --seed 42

  test -s "$OUTPUT_ROOT/step${STEP}/piper/summary_metrics.json"
  test -s "$OUTPUT_ROOT/step${STEP}/human/summary_metrics.json"
  test -s "$OUTPUT_ROOT/step${STEP}/piper/piper_representative_joint_eval.png"
  test -s "$OUTPUT_ROOT/step${STEP}/human/human_representative_episode_eval.png"
done

run_container "$PYTHON" "$EXPERIMENT_BUNDLE/eval/compare_checkpoints.py" \
  --output-root "$OUTPUT_ROOT" --steps 25000 30000
test -s "$OUTPUT_ROOT/checkpoint_comparison.json"
test -s "$OUTPUT_ROOT/checkpoint_metric_comparison.png"
test -s "$OUTPUT_ROOT/piper_same_episode_comparison.png"
test -s "$OUTPUT_ROOT/human_same_episode_comparison.png"

printf 'completed_utc=%s\njob_id=%s\ntrain_job_id=%s\ncheckpoints=25000,30000\nsplit=test\nquery_frequency=30\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SLURM_JOB_ID" "$TRAIN_JOB_ID" \
  > "$OUTPUT_ROOT/EVALUATION_COMPLETED"

echo "Absolute-Reference V3 held-out comparison complete: $OUTPUT_ROOT"
