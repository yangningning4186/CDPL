#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data4/ynz/CDPL-src}
RUN_ROOT=${RUN_ROOT:-/data4/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib}
CONDA_SH=${CONDA_SH:-/data4/ynz/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-ubt}

RUN_MODE=${RUN_MODE:-late_weak}
SEED=${SEED:-0}
NUM_GPUS=${NUM_GPUS:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
RUN_NAME=${RUN_NAME:-${RUN_MODE}_${NUM_GPUS}gpu_seed${SEED}}

MODEL_WEIGHTS=${MODEL_WEIGHTS:-detectron2://ImageNetPretrained/MSRA/R-50.pkl}
MAX_ITER=${MAX_ITER:-24000}
BURN_UP_STEP=${BURN_UP_STEP:-9000}
EVAL_PERIOD=${EVAL_PERIOD:-3000}
CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-3000}
BASE_LR=${BASE_LR:-0.005}
IMS_PER_BATCH=${IMS_PER_BATCH:-8}
IMG_PER_BATCH_LABEL=${IMG_PER_BATCH_LABEL:-4}
IMG_PER_BATCH_UNLABEL=${IMG_PER_BATCH_UNLABEL:-4}
NUM_WORKERS=${NUM_WORKERS:-2}
DIAG_LIMIT=${DIAG_LIMIT:-0}

export OCT_SS_TRAIN_JSON=${OCT_SS_TRAIN_JSON:-/data4/ynz/OCT_SS-main/datasets/annotations/train_labeled.json}
export OCT_SS_UNLABEL_JSON=${OCT_SS_UNLABEL_JSON:-/data4/ynz/OCT_SS-main/datasets/annotations/train_unlabeled_v3_1.json}
export OCT_SS_TEST_JSON=${OCT_SS_TEST_JSON:-/data4/ynz/OCT_SS-main/datasets/annotations/test_v3_1.json}
export OCT_SS_TRAIN_IMAGE_ROOT=${OCT_SS_TRAIN_IMAGE_ROOT:-/data4/ynz/YOLO-SS/data/split_data/images/train}
export OCT_SS_UNLABEL_IMAGE_ROOT=${OCT_SS_UNLABEL_IMAGE_ROOT:-/data4/ynz/YOLO-SS/data/unlabeled_data}
export OCT_SS_IMAGE_ROOT=${OCT_SS_IMAGE_ROOT:-/data4/ynz/YOLO-SS/data/split_data/images/test}
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

case "$RUN_MODE" in
  ubt_baseline)
    TRAINER=ubteacher
    CDPL_ENABLED=False
    CDPL_ALPHA_TAIL=${CDPL_ALPHA_TAIL:-0.15}
    CDPL_MIN_CLS_THRESHOLD=${CDPL_MIN_CLS_THRESHOLD:-0.35}
    CDPL_MAX_THRESHOLD_OFFSET=${CDPL_MAX_THRESHOLD_OFFSET:--1.0}
    CDPL_START_ITER=${CDPL_START_ITER:--1}
    CDPL_RAMP_ITERS=${CDPL_RAMP_ITERS:-0}
    ;;
  late_weak)
    TRAINER=cdpl
    CDPL_ENABLED=True
    CDPL_ALPHA_TAIL=${CDPL_ALPHA_TAIL:-0.05}
    CDPL_MIN_CLS_THRESHOLD=${CDPL_MIN_CLS_THRESHOLD:-0.35}
    CDPL_MAX_THRESHOLD_OFFSET=${CDPL_MAX_THRESHOLD_OFFSET:--1.0}
    CDPL_START_ITER=${CDPL_START_ITER:-18000}
    CDPL_RAMP_ITERS=${CDPL_RAMP_ITERS:-6000}
    ;;
  floor_clamp)
    TRAINER=cdpl
    CDPL_ENABLED=True
    CDPL_ALPHA_TAIL=${CDPL_ALPHA_TAIL:-0.15}
    CDPL_MIN_CLS_THRESHOLD=${CDPL_MIN_CLS_THRESHOLD:-0.45}
    CDPL_MAX_THRESHOLD_OFFSET=${CDPL_MAX_THRESHOLD_OFFSET:-0.05}
    CDPL_START_ITER=${CDPL_START_ITER:--1}
    CDPL_RAMP_ITERS=${CDPL_RAMP_ITERS:-0}
    ;;
  *)
    echo "Unknown RUN_MODE=$RUN_MODE" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT"
cd "$REPO_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

RUN_DIR="$RUN_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"

COMMIT=$(git rev-parse HEAD)
BRANCH=$(git branch --show-current)

cmd=(
  python train_net.py
  --num-gpus "$NUM_GPUS"
  --resume
  --config-file configs/oct_ss_cdpl.yaml
  MODEL.WEIGHTS "$MODEL_WEIGHTS"
  OUTPUT_DIR "$RUN_DIR"
  SEED "$SEED"
  SOLVER.BASE_LR "$BASE_LR"
  SOLVER.MAX_ITER "$MAX_ITER"
  SOLVER.IMS_PER_BATCH "$IMS_PER_BATCH"
  SOLVER.IMG_PER_BATCH_LABEL "$IMG_PER_BATCH_LABEL"
  SOLVER.IMG_PER_BATCH_UNLABEL "$IMG_PER_BATCH_UNLABEL"
  SOLVER.CHECKPOINT_PERIOD "$CHECKPOINT_PERIOD"
  DATALOADER.NUM_WORKERS "$NUM_WORKERS"
  TEST.EVAL_PERIOD "$EVAL_PERIOD"
  SEMISUPNET.BURN_UP_STEP "$BURN_UP_STEP"
  SEMISUPNET.Trainer "$TRAINER"
  SEMISUPNET.CDPL_ENABLED "$CDPL_ENABLED"
  SEMISUPNET.CDPL_ALPHA_TAIL "$CDPL_ALPHA_TAIL"
  SEMISUPNET.CDPL_MIN_CLS_THRESHOLD "$CDPL_MIN_CLS_THRESHOLD"
  SEMISUPNET.CDPL_MAX_THRESHOLD_OFFSET "$CDPL_MAX_THRESHOLD_OFFSET"
  SEMISUPNET.CDPL_START_ITER "$CDPL_START_ITER"
  SEMISUPNET.CDPL_RAMP_ITERS "$CDPL_RAMP_ITERS"
)

cp configs/oct_ss_cdpl.yaml "$RUN_DIR/config_input.yaml"
git status --short --branch > "$RUN_DIR/git_status.txt"
echo "$COMMIT" > "$RUN_DIR/git_commit.txt"
printf '%q ' "${cmd[@]}" > "$RUN_DIR/launch_command.sh"
printf '\n' >> "$RUN_DIR/launch_command.sh"
chmod +x "$RUN_DIR/launch_command.sh"
{
  echo "run_name: $RUN_NAME"
  echo "run_mode: $RUN_MODE"
  echo "branch: $BRANCH"
  echo "commit: $COMMIT"
  echo "repo_dir: $REPO_DIR"
  echo "run_root: $RUN_ROOT"
  echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
  echo "num_gpus: $NUM_GPUS"
  echo "max_iter: $MAX_ITER"
  echo "burn_up_step: $BURN_UP_STEP"
  echo "eval_period: $EVAL_PERIOD"
  echo "checkpoint_period: $CHECKPOINT_PERIOD"
  echo "base_lr: $BASE_LR"
  echo "trainer: $TRAINER"
  echo "cdpl_enabled: $CDPL_ENABLED"
  echo "cdpl_alpha_tail: $CDPL_ALPHA_TAIL"
  echo "cdpl_min_cls_threshold: $CDPL_MIN_CLS_THRESHOLD"
  echo "cdpl_max_threshold_offset: $CDPL_MAX_THRESHOLD_OFFSET"
  echo "cdpl_start_iter: $CDPL_START_ITER"
  echo "cdpl_ramp_iters: $CDPL_RAMP_ITERS"
  echo "oct_ss_train_json: $OCT_SS_TRAIN_JSON"
  echo "oct_ss_unlabel_json: $OCT_SS_UNLABEL_JSON"
  echo "oct_ss_test_json: $OCT_SS_TEST_JSON"
} > "$RUN_DIR/run_metadata.txt"

nvidia-smi > "$RUN_DIR/nvidia_smi_start.txt" 2>&1 || true
set +e
"${cmd[@]}" 2>&1 | tee "$RUN_DIR/train.log"
train_code=${PIPESTATUS[0]}
set -e
nvidia-smi > "$RUN_DIR/nvidia_smi_end.txt" 2>&1 || true
echo "$train_code" > "$RUN_DIR/exit_code.txt"

if [[ "$train_code" -ne 0 ]]; then
  echo "[$(date '+%F %T')] $RUN_NAME failed with exit code $train_code"
  exit "$train_code"
fi

test -f "$RUN_DIR/model_final.pth"
test -f "$RUN_DIR/inference/coco_instances_results.json"

python scripts/experiments/diagnose_step2_pseudo_labels.py \
  --config-file configs/oct_ss_cdpl.yaml \
  --checkpoint "$RUN_DIR/model_final.pth" \
  --output "$RUN_DIR/pseudo_label_diagnostics_final.json" \
  --limit "$DIAG_LIMIT" \
  MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.05 \
  SEMISUPNET.BBOX_THRESHOLD 0.5 \
  SEMISUPNET.CDPL_ALPHA_TAIL "$CDPL_ALPHA_TAIL" \
  SEMISUPNET.CDPL_MIN_CLS_THRESHOLD "$CDPL_MIN_CLS_THRESHOLD" \
  SEMISUPNET.CDPL_MAX_THRESHOLD_OFFSET "$CDPL_MAX_THRESHOLD_OFFSET"

date '+%F %T' > "$RUN_DIR/complete.ok"
echo "[$(date '+%F %T')] Completed $RUN_NAME"
