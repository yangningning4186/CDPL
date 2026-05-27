#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data2/ynz/CDPL-src}
RUN_ROOT=${RUN_ROOT:-/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/oct_ss_small_cutout_ubt_vs_cdpl_24k}
CONDA_SH=${CONDA_SH:-/data2/ynz/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-oct_ss}
CONFIG_FILE=${CONFIG_FILE:-configs/oct_ss_ubt_repro.yaml}

RUN_MODE=${RUN_MODE:-ubt_repro}
SEED=${SEED:-0}
NUM_GPUS=${NUM_GPUS:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MAX_ITER=${MAX_ITER:-24000}
RUN_NAME=${RUN_NAME:-${RUN_MODE}_${MAX_ITER}iter_1gpu_seed${SEED}}

export OCT_SS_TRAIN_JSON=${OCT_SS_TRAIN_JSON:-/data2/ynz/cdpl_inputs/annotations/train_labeled.json}
export OCT_SS_UNLABEL_JSON=${OCT_SS_UNLABEL_JSON:-/data2/ynz/OCT_SS/datasets/annotations/train_unlabeled_v3_1.json}
export OCT_SS_TEST_JSON=${OCT_SS_TEST_JSON:-/data2/ynz/OCT_SS/datasets/annotations/test_v3_1.json}
export OCT_SS_TRAIN_IMAGE_ROOT=${OCT_SS_TRAIN_IMAGE_ROOT:-/data2/ynz/OCT_SS/datasets/images}
export OCT_SS_UNLABEL_IMAGE_ROOT=${OCT_SS_UNLABEL_IMAGE_ROOT:-/data2/ynz/OCT_SS/datasets/images}
export OCT_SS_IMAGE_ROOT=${OCT_SS_IMAGE_ROOT:-/data2/ynz/OCT_SS/datasets/images}
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export RUN_MODE SEED NUM_GPUS MAX_ITER RUN_NAME

case "$RUN_MODE" in
  ubt_repro)
    TRAINER=ubteacher
    CDPL_ENABLED=False
    ;;
  cdpl_calib_repro)
    TRAINER=cdpl
    CDPL_ENABLED=True
    ;;
  cdpl_sweep|cdpl_delayed_weak)
    TRAINER=cdpl
    CDPL_ENABLED=True
    ;;
  *)
    echo "Unknown RUN_MODE=$RUN_MODE; expected ubt_repro, cdpl_calib_repro, cdpl_sweep, or cdpl_delayed_weak" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT"
cd "$REPO_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

RUN_DIR="$RUN_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"
if [[ -f "$RUN_DIR/complete.ok" ]]; then
  echo "skip $RUN_NAME: already complete"
  exit 0
fi

COMMIT=$(git rev-parse HEAD)
BRANCH=$(git branch --show-current)
cmd=(
  python train_net.py
  --num-gpus "$NUM_GPUS"
  --resume
  --config-file "$CONFIG_FILE"
  OUTPUT_DIR "$RUN_DIR"
  SEED "$SEED"
  SOLVER.MAX_ITER "$MAX_ITER"
  SEMISUPNET.Trainer "$TRAINER"
  SEMISUPNET.CDPL_ENABLED "$CDPL_ENABLED"
)

append_cfg_override_if_set() {
  local env_name="$1"
  local cfg_key="$2"
  local value="${!env_name:-}"
  if [[ -n "$value" ]]; then
    cmd+=("$cfg_key" "$value")
  fi
}

append_cfg_override_if_set BBOX_THRESHOLD SEMISUPNET.BBOX_THRESHOLD
append_cfg_override_if_set BURN_UP_STEP SEMISUPNET.BURN_UP_STEP
append_cfg_override_if_set CDPL_TAU_BASE SEMISUPNET.CDPL_TAU_BASE
append_cfg_override_if_set CDPL_ALPHA_TAIL SEMISUPNET.CDPL_ALPHA_TAIL
append_cfg_override_if_set CDPL_MIN_CLS_THRESHOLD SEMISUPNET.CDPL_MIN_CLS_THRESHOLD
append_cfg_override_if_set CDPL_MAX_THRESHOLD_OFFSET SEMISUPNET.CDPL_MAX_THRESHOLD_OFFSET
append_cfg_override_if_set CDPL_START_ITER SEMISUPNET.CDPL_START_ITER
append_cfg_override_if_set CDPL_RAMP_ITERS SEMISUPNET.CDPL_RAMP_ITERS

cp "$CONFIG_FILE" "$RUN_DIR/config_input.yaml"
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
  echo "config_file: $CONFIG_FILE"
  echo "conda_env: $CONDA_ENV"
  echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
  echo "num_gpus: $NUM_GPUS"
  echo "max_iter: $MAX_ITER"
  echo "trainer: $TRAINER"
  echo "cdpl_enabled: $CDPL_ENABLED"
  [[ -n "${BBOX_THRESHOLD:-}" ]] && echo "bbox_threshold: $BBOX_THRESHOLD"
  [[ -n "${BURN_UP_STEP:-}" ]] && echo "burn_up_step: $BURN_UP_STEP"
  [[ -n "${CDPL_TAU_BASE:-}" ]] && echo "cdpl_tau_base: $CDPL_TAU_BASE"
  [[ -n "${CDPL_ALPHA_TAIL:-}" ]] && echo "cdpl_alpha_tail: $CDPL_ALPHA_TAIL"
  [[ -n "${CDPL_MIN_CLS_THRESHOLD:-}" ]] && echo "cdpl_min_cls_threshold: $CDPL_MIN_CLS_THRESHOLD"
  [[ -n "${CDPL_MAX_THRESHOLD_OFFSET:-}" ]] && echo "cdpl_max_threshold_offset: $CDPL_MAX_THRESHOLD_OFFSET"
  [[ -n "${CDPL_START_ITER:-}" ]] && echo "cdpl_start_iter: $CDPL_START_ITER"
  [[ -n "${CDPL_RAMP_ITERS:-}" ]] && echo "cdpl_ramp_iters: $CDPL_RAMP_ITERS"
  echo "oct_ss_train_json: $OCT_SS_TRAIN_JSON"
  echo "oct_ss_unlabel_json: $OCT_SS_UNLABEL_JSON"
  echo "oct_ss_test_json: $OCT_SS_TEST_JSON"
  echo "oct_ss_train_image_root: $OCT_SS_TRAIN_IMAGE_ROOT"
  echo "oct_ss_unlabel_image_root: $OCT_SS_UNLABEL_IMAGE_ROOT"
  echo "oct_ss_image_root: $OCT_SS_IMAGE_ROOT"
} > "$RUN_DIR/run_metadata.txt"

python - "$RUN_DIR/config_overrides.json" <<'PY'
import json
import os
import sys

keys = [
    "RUN_NAME",
    "RUN_MODE",
    "SEED",
    "NUM_GPUS",
    "CUDA_VISIBLE_DEVICES",
    "MAX_ITER",
    "BBOX_THRESHOLD",
    "BURN_UP_STEP",
    "CDPL_TAU_BASE",
    "CDPL_ALPHA_TAIL",
    "CDPL_MIN_CLS_THRESHOLD",
    "CDPL_MAX_THRESHOLD_OFFSET",
    "CDPL_START_ITER",
    "CDPL_RAMP_ITERS",
]
payload = {key.lower(): os.environ[key] for key in keys if os.environ.get(key) not in (None, "")}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY

nvidia-smi > "$RUN_DIR/nvidia_smi_start.txt" 2>&1 || true
set +e
"${cmd[@]}" 2>&1 | tee "$RUN_DIR/train.log"
train_code=${PIPESTATUS[0]}
set -e
nvidia-smi > "$RUN_DIR/nvidia_smi_end.txt" 2>&1 || true
echo "$train_code" > "$RUN_DIR/exit_code.txt"
if [[ "$train_code" -ne 0 ]]; then
  exit "$train_code"
fi

test -f "$RUN_DIR/model_final.pth"
date '+%F %T' > "$RUN_DIR/complete.ok"
