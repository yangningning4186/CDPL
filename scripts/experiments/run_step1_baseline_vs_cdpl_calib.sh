#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data4/ynz/CDPL-src}
RUN_ROOT=${RUN_ROOT:-/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib}
DOC_PATH=${DOC_PATH:-docs/superpowers/experiments/step1-baseline-vs-cdpl-calibration.md}
CONDA_SH=${CONDA_SH:-/data4/ynz/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-ubt}

NUM_GPUS=${NUM_GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
MAX_ITER=${MAX_ITER:-360000}
EVAL_PERIOD=${EVAL_PERIOD:-10000}
CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-10000}
IMG_PER_BATCH_LABEL=${IMG_PER_BATCH_LABEL:-4}
IMG_PER_BATCH_UNLABEL=${IMG_PER_BATCH_UNLABEL:-4}
IMS_PER_BATCH=${IMS_PER_BATCH:-4}
NUM_WORKERS=${NUM_WORKERS:-2}
SEED=${SEED:-0}

export OCT_SS_TRAIN_JSON=${OCT_SS_TRAIN_JSON:-/data4/ynz/OCT_SS-main/datasets/annotations/train_labeled.json}
export OCT_SS_UNLABEL_JSON=${OCT_SS_UNLABEL_JSON:-/data4/ynz/OCT_SS-main/datasets/annotations/train_unlabeled_v3_1.json}
export OCT_SS_TEST_JSON=${OCT_SS_TEST_JSON:-/data4/ynz/OCT_SS-main/datasets/annotations/test_v3_1.json}
export OCT_SS_TRAIN_IMAGE_ROOT=${OCT_SS_TRAIN_IMAGE_ROOT:-/data4/ynz/YOLO-SS/data/split_data/images/train}
export OCT_SS_UNLABEL_IMAGE_ROOT=${OCT_SS_UNLABEL_IMAGE_ROOT:-/data4/ynz/YOLO-SS/data/unlabeled_data}
export OCT_SS_IMAGE_ROOT=${OCT_SS_IMAGE_ROOT:-/data4/ynz/YOLO-SS/data/split_data/images/test}
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

mkdir -p "$RUN_ROOT"
cd "$REPO_DIR"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

COMMIT=$(git rev-parse HEAD)
BRANCH=$(git branch --show-current)

write_run_metadata() {
  local run_dir=$1
  local run_name=$2
  shift 2

  mkdir -p "$run_dir"
  cp configs/oct_ss_cdpl.yaml "$run_dir/config_input.yaml"
  git status --short --branch > "$run_dir/git_status.txt"
  conda env export --no-builds > "$run_dir/conda_env.yaml"
  python - <<'PY' > "$run_dir/python_env.txt"
import platform
print("python:", platform.python_version())
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_device_count:", torch.cuda.device_count())
except Exception as exc:
    print("torch_import_error:", repr(exc))
try:
    import detectron2
    print("detectron2:", getattr(detectron2, "__version__", "unknown"))
except Exception as exc:
    print("detectron2_import_error:", repr(exc))
PY
  {
    echo "run_name: $run_name"
    echo "branch: $BRANCH"
    echo "commit: $COMMIT"
    echo "repo_dir: $REPO_DIR"
    echo "run_root: $RUN_ROOT"
    echo "num_gpus: $NUM_GPUS"
    echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
    echo "max_iter: $MAX_ITER"
    echo "eval_period: $EVAL_PERIOD"
    echo "checkpoint_period: $CHECKPOINT_PERIOD"
    echo "img_per_batch_label: $IMG_PER_BATCH_LABEL"
    echo "img_per_batch_unlabel: $IMG_PER_BATCH_UNLABEL"
    echo "seed: $SEED"
    echo "oct_ss_train_json: $OCT_SS_TRAIN_JSON"
    echo "oct_ss_unlabel_json: $OCT_SS_UNLABEL_JSON"
    echo "oct_ss_test_json: $OCT_SS_TEST_JSON"
    echo "oct_ss_train_image_root: $OCT_SS_TRAIN_IMAGE_ROOT"
    echo "oct_ss_unlabel_image_root: $OCT_SS_UNLABEL_IMAGE_ROOT"
    echo "oct_ss_image_root: $OCT_SS_IMAGE_ROOT"
  } > "$run_dir/run_metadata.txt"

  printf '%q ' "$@" > "$run_dir/launch_command.sh"
  printf '\n' >> "$run_dir/launch_command.sh"
  chmod +x "$run_dir/launch_command.sh"
  echo "$COMMIT" > "$run_dir/git_commit.txt"
}

run_one() {
  local run_name=$1
  local trainer=$2
  local cdpl_enabled=$3
  local run_dir="$RUN_ROOT/$run_name"

  if [[ -f "$run_dir/complete.ok" ]]; then
    echo "[$(date '+%F %T')] $run_name already complete; skipping."
    return 0
  fi

  mkdir -p "$run_dir"

  local cmd=(
    python train_net.py
    --num-gpus "$NUM_GPUS"
    --resume
    --config-file configs/oct_ss_cdpl.yaml
    MODEL.WEIGHTS ""
    OUTPUT_DIR "$run_dir"
    SEED "$SEED"
    SOLVER.MAX_ITER "$MAX_ITER"
    SOLVER.IMS_PER_BATCH "$IMS_PER_BATCH"
    SOLVER.IMG_PER_BATCH_LABEL "$IMG_PER_BATCH_LABEL"
    SOLVER.IMG_PER_BATCH_UNLABEL "$IMG_PER_BATCH_UNLABEL"
    SOLVER.CHECKPOINT_PERIOD "$CHECKPOINT_PERIOD"
    DATALOADER.NUM_WORKERS "$NUM_WORKERS"
    TEST.EVAL_PERIOD "$EVAL_PERIOD"
    SEMISUPNET.Trainer "$trainer"
    SEMISUPNET.CDPL_ENABLED "$cdpl_enabled"
  )

  write_run_metadata "$run_dir" "$run_name" "${cmd[@]}"

  echo "[$(date '+%F %T')] Starting $run_name on GPUs $CUDA_VISIBLE_DEVICES"
  nvidia-smi > "$run_dir/nvidia_smi_start.txt"
  set +e
  "${cmd[@]}" 2>&1 | tee "$run_dir/train.log"
  local train_code=${PIPESTATUS[0]}
  set -e
  nvidia-smi > "$run_dir/nvidia_smi_end.txt" || true
  echo "$train_code" > "$run_dir/exit_code.txt"

  if [[ "$train_code" -ne 0 ]]; then
    echo "[$(date '+%F %T')] $run_name failed with exit code $train_code"
    return "$train_code"
  fi

  test -f "$run_dir/model_final.pth"
  test -f "$run_dir/inference/coco_instances_results.json"
  date '+%F %T' > "$run_dir/complete.ok"
  echo "[$(date '+%F %T')] Completed $run_name"
}

run_one ubt_baseline_4gpu_seed0 ubteacher False
run_one cdpl_calib_4gpu_seed0 cdpl True

python scripts/experiments/summarize_step1_results.py \
  --run-root "$RUN_ROOT" \
  --annotations "$OCT_SS_TEST_JSON" \
  --doc "$DOC_PATH"
