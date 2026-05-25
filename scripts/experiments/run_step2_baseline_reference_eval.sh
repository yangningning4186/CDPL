#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data4/ynz/CDPL-src}
STEP1_ROOT=${STEP1_ROOT:-/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib}
STEP2_ROOT=${STEP2_ROOT:-/data4/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib}
CONDA_SH=${CONDA_SH:-/data4/ynz/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-ubt}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_WORKERS=${NUM_WORKERS:-2}

export OCT_SS_TRAIN_JSON=${OCT_SS_TRAIN_JSON:-/data4/ynz/OCT_SS-main/datasets/annotations/train_labeled.json}
export OCT_SS_UNLABEL_JSON=${OCT_SS_UNLABEL_JSON:-/data4/ynz/OCT_SS-main/datasets/annotations/train_unlabeled_v3_1.json}
export OCT_SS_TEST_JSON=${OCT_SS_TEST_JSON:-/data4/ynz/OCT_SS-main/datasets/annotations/test_v3_1.json}
export OCT_SS_TRAIN_IMAGE_ROOT=${OCT_SS_TRAIN_IMAGE_ROOT:-/data4/ynz/YOLO-SS/data/split_data/images/train}
export OCT_SS_UNLABEL_IMAGE_ROOT=${OCT_SS_UNLABEL_IMAGE_ROOT:-/data4/ynz/YOLO-SS/data/unlabeled_data}
export OCT_SS_IMAGE_ROOT=${OCT_SS_IMAGE_ROOT:-/data4/ynz/YOLO-SS/data/split_data/images/test}
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

cd "$REPO_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

REF_ROOT="$STEP2_ROOT/baseline_reference"
mkdir -p "$REF_ROOT"
SOURCE_GIT_COMMIT=$(cat "$STEP1_ROOT/ubt_baseline_1gpu_seed0/git_commit.txt" 2>/dev/null || true)

eval_one() {
  local label="$1"
  local checkpoint="$2"
  local output_dir="$REF_ROOT/$label"
  local log="$output_dir/eval.log"

  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    return 1
  fi

  mkdir -p "$output_dir"
  local cmd=(
    python train_net.py
    --num-gpus 1
    --eval-only
    --config-file configs/oct_ss_cdpl.yaml
    MODEL.WEIGHTS "$checkpoint"
    OUTPUT_DIR "$output_dir"
    SEMISUPNET.Trainer ubteacher
    SEMISUPNET.CDPL_ENABLED False
    SOLVER.IMS_PER_BATCH 8
    SOLVER.IMG_PER_BATCH_LABEL 4
    SOLVER.IMG_PER_BATCH_UNLABEL 4
    DATALOADER.NUM_WORKERS "$NUM_WORKERS"
  )

  printf '%q ' "${cmd[@]}" > "$output_dir/eval_command.sh"
  printf '\n' >> "$output_dir/eval_command.sh"
  chmod +x "$output_dir/eval_command.sh"

  if [[ -f "$output_dir/inference/coco_instances_results.json" && "${FORCE_EVAL:-0}" != "1" ]]; then
    echo "skip $label: predictions already exist"
  else
    "${cmd[@]}" 2>&1 | tee "$log"
  fi

  test -f "$output_dir/inference/coco_instances_results.json"
  date '+%F %T' > "$output_dir/eval.ok"
}

eval_one \
  ubt_baseline_seed0_24k_final_iter23999 \
  "$STEP1_ROOT/ubt_baseline_1gpu_seed0/model_0023999.pth"

eval_one \
  ubt_baseline_seed0_24k_best_iter17999 \
  "$STEP1_ROOT/ubt_baseline_1gpu_seed0/model_0017999.pth"

cat > "$REF_ROOT/reference_metadata.json" <<EOF
{
  "source_step1_root": "$STEP1_ROOT",
  "source_git_commit": "$SOURCE_GIT_COMMIT",
  "purpose": "UBT baseline reference for Step2 MAX_ITER=24000 fair comparison",
  "final_checkpoint": "$STEP1_ROOT/ubt_baseline_1gpu_seed0/model_0023999.pth",
  "best_test_checkpoint": "$STEP1_ROOT/ubt_baseline_1gpu_seed0/model_0017999.pth",
  "best_test_selection": "best periodic test bbox/AP among checkpoints up to iter 23999",
  "output_root": "$REF_ROOT",
  "created_at": "$(date '+%F %T')"
}
EOF

echo "Wrote baseline reference evals to $REF_ROOT"
