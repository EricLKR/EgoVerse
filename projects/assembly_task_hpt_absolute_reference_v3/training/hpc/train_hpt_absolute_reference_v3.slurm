#!/bin/bash -l
#SBATCH --account=naiss2026-3-376-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --job-name=asm-new1-hpt-abs-v3
#SBATCH --output=/nobackup/proj/disk/naiss2024-5-164/personal/kerun/egoverse_piper_run/outputs/logs/%x-%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-5-164/personal/kerun/egoverse_piper_run/outputs/logs/%x-%j.err

set -euo pipefail
umask 077

RUN_MODE=${RUN_MODE:-smoke}
MAX_STEPS=${MAX_STEPS:-30000}
DATA_ROOT=/nobackup/proj/disk/naiss2024-5-164/personal/kerun/egoverse_piper_run
BASE_BUNDLE="$DATA_ROOT/code/assembly_task_new1_training_v2"
EXPERIMENT_BUNDLE="$DATA_ROOT/code/assembly_task_new1_hpt_absolute_reference_v3"
CODE="$DATA_ROOT/code/egoverse_piper_assembly_hpt_new1_v2/EgoVerse"
IMAGE="$DATA_ROOT/env/pytorch_25.06-py3.sif"
PYTHON="$DATA_ROOT/env/emimic/bin/python"
PIPER_DATA="$DATA_ROOT/data/piper_assembly_new1_3cam_90_5_5_zarr"
HUMAN_DATA="$DATA_ROOT/data/aria_assembly_new1_right_54_3_3_zarr"
EXTRINSICS="$BASE_BUNDLE/assets/piper_front_extrinsics_recording_20261016.json"
ACCEPTANCE="$BASE_BUNDLE/assets/formal_extrinsics_acceptance_PENDING.json"
RUN_ROOT="$DATA_ROOT/outputs/runs/assembly_new1_hpt_absolute_reference_v3"
RUN_DIR="$RUN_ROOT/${RUN_MODE}_${SLURM_JOB_ID}"

export PIPER_HPC="$DATA_ROOT"
export PIPER_RECORDING_EXTRINSICS_JSON="$EXTRINSICS"
unset PIPER_EXTRINSICS_JSON || true
export HF_HOME="$DATA_ROOT/cache/huggingface"
export TORCH_HOME="$DATA_ROOT/cache/torch"
export WANDB_DIR="$DATA_ROOT/cache/wandb"
export PIP_CACHE_DIR="$DATA_ROOT/cache/pip"
export TMPDIR="$DATA_ROOT/tmp"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

test -f "$DATA_ROOT/env/ENVIRONMENT_VALIDATED"
test -s "$IMAGE"; test -x "$PYTHON"
test -d "$PIPER_DATA"; test -d "$HUMAN_DATA"; test -s "$EXTRINSICS"
test -s "$CODE/egomimic/models/absolute_reference_joint_policy_v3.py"
test -s "$CODE/egomimic/algo/hpt_absolute_reference_v3.py"
test -s "$CODE/egomimic/hydra_configs/model/hpt_cotrain_piper_human_absolute_reference_v3_assembly_new1_3cam.yaml"
mkdir -p "$RUN_DIR" "$DATA_ROOT/outputs/logs" "$TMPDIR"

run_container() {
  apptainer exec --nv --bind "$DATA_ROOT:$DATA_ROOT" "$IMAGE" "$@"
}

run_container "$PYTHON" "$BASE_BUNDLE/scripts/preflight_robot_3cam.py" \
  --robot-root "$PIPER_DATA" --human-root "$HUMAN_DATA" \
  --split-json "$BASE_BUNDLE/assets/split_90_5_5.json" \
  --human-split-json "$BASE_BUNDLE/assets/ego_split_54_3_3.json" \
  --calibration-json "$BASE_BUNDLE/assets/piper_follower_calibration.json" \
  --camera-contract "$BASE_BUNDLE/assets/camera_contract.json" \
  --extrinsics-json "$EXTRINSICS" --formal-acceptance "$ACCEPTANCE" \
  --output "$RUN_DIR/robot_preflight.json"

run_container "$PYTHON" -c \
  'from egomimic.models.absolute_reference_joint_policy_v3 import AbsoluteReferenceJointPolicyV3; assert AbsoluteReferenceJointPolicyV3.contract_version == 3'

cd "$CODE"
sha256sum \
  egomimic/models/absolute_reference_joint_policy_v3.py \
  egomimic/algo/hpt_absolute_reference_v3.py \
  egomimic/hydra_configs/train_zarr_piper_human_absolute_reference_v3.yaml \
  egomimic/hydra_configs/model/hpt_cotrain_piper_human_absolute_reference_v3_assembly_new1_3cam.yaml \
  egomimic/hydra_configs/data/assembly_task_new1_mixed_90_5_5.yaml \
  "$PIPER_DATA/conversion_manifest.json" "$HUMAN_DATA/conversion_status.json" \
  "$BASE_BUNDLE/assets/ego_split_54_3_3.json" "$EXTRINSICS" \
  > "$RUN_DIR/source_checksums.sha256"

run_container "$PYTHON" -m pytest -q \
  egomimic/models/test_absolute_reference_joint_policy_v3.py \
  egomimic/models/test_residual_joint_policy_mixed_clean_v2.py \
  egomimic/algo/test_hpt_multihead_loss.py \
  egomimic/rldb/embodiment/test_piper_human_joint_three_camera.py

case "$RUN_MODE" in
  smoke)
    STEPS=3; NORM_PATH=null; CHECKPOINT_INTERVAL=3; VAL_INTERVAL=3
    ;;
  formal)
    SMOKE_JOB_ID=${SMOKE_JOB_ID:?Set SMOKE_JOB_ID to a successful V3 smoke job}
    SMOKE_RUN="$RUN_ROOT/smoke_${SMOKE_JOB_ID}"
    test -s "$SMOKE_RUN/SMOKE_COMPLETED"
    test -s "$SMOKE_RUN/train/norm_stats/norm_stats.json"
    run_container "$PYTHON" - "$ACCEPTANCE" "$EXTRINSICS" <<'PY'
import hashlib, json, pathlib, sys
a=json.loads(pathlib.Path(sys.argv[1]).read_text()); e=pathlib.Path(sys.argv[2])
assert a.get('accepted') is True, 'formal training blocked: extrinsics not accepted'
assert a.get('extrinsics_sha256') == hashlib.sha256(e.read_bytes()).hexdigest()
PY
    sha256sum -c "$SMOKE_RUN/source_checksums.sha256"
    STEPS="$MAX_STEPS"; NORM_PATH="$SMOKE_RUN/train/norm_stats"; CHECKPOINT_INTERVAL=5000; VAL_INTERVAL=5000
    ;;
  *) echo "Unknown RUN_MODE=$RUN_MODE" >&2; exit 2 ;;
esac

cd "$CODE/egomimic"
run_container "$PYTHON" trainHydra.py \
  --config-name=train_zarr_piper_human_absolute_reference_v3 \
  data=assembly_task_new1_mixed_90_5_5 \
  model=hpt_cotrain_piper_human_absolute_reference_v3_assembly_new1_3cam \
  evaluator=eval_hpt_piper_human_residual_clean_v2 logger=csv \
  name=assembly_new1_hpt_absolute_reference_v3 description="${RUN_MODE}_${SLURM_JOB_ID}" \
  "hydra.run.dir=$RUN_DIR/train" init_weights_path=null ckpt_path=null \
  "norm_stats.precomputed_norm_path=$NORM_PATH" norm_stats.sample_frac=1.0 \
  +trainer.max_steps="$STEPS" trainer.max_epochs=-1 trainer.min_epochs=null \
  trainer.limit_train_batches=1.0 trainer.limit_val_batches=2 \
  trainer.num_sanity_val_steps=0 trainer.check_val_every_n_epoch=null \
  +trainer.val_check_interval="$VAL_INTERVAL" \
  +trainer.gradient_clip_val=10.0 +trainer.gradient_clip_algorithm=norm \
  model.scheduler.T_max="$STEPS" callbacks.model_checkpoint.every_n_epochs=null \
  +callbacks.model_checkpoint.every_n_train_steps="$CHECKPOINT_INTERVAL" \
  callbacks.model_checkpoint.save_top_k=-1 \
  +callbacks.model_checkpoint.save_on_train_epoch_end=false \
  "callbacks.model_checkpoint.filename='step_{step}'"

if [[ "$RUN_MODE" == smoke ]]; then
  test -s "$RUN_DIR/train/norm_stats/norm_stats.json"
  test -n "$(find "$RUN_DIR/train/checkpoints" -maxdepth 1 -name '*.ckpt' -print -quit)"
  printf 'completed_utc=%s\njob_id=%s\ncontract=piper_absolute_reference_joint_v3\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SLURM_JOB_ID" > "$RUN_DIR/SMOKE_COMPLETED"
else
  test -n "$(find "$RUN_DIR/train/checkpoints" -maxdepth 1 -name "*${STEPS}*.ckpt" -print -quit)"
  sha256sum "$RUN_DIR"/train/checkpoints/*.ckpt > "$RUN_DIR/checkpoints.sha256"
  printf 'completed_utc=%s\njob_id=%s\nfinal_step=%s\ncontract=piper_absolute_reference_joint_v3\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SLURM_JOB_ID" "$STEPS" > "$RUN_DIR/FORMAL_COMPLETED"
fi
