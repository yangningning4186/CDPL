#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data2/ynz/CDPL-src}
RUN_ROOT=${RUN_ROOT:-/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/oct_ss_small_cutout_ubt_vs_cdpl_24k}
CONDA_SH=${CONDA_SH:-/data2/ynz/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-oct_ss}
CONFIG_FILE=${CONFIG_FILE:-configs/oct_ss_ubt_repro.yaml}
TRAINING_COMMIT=${TRAINING_COMMIT:-3555268ad84e2489eeef9ca6513a99d255d2ef32}
UBT_RUN=${UBT_RUN:-ubt_repro_24000iter_1gpu_seed0}
CDPL_RUN=${CDPL_RUN:-cdpl_calib_repro_24000iter_1gpu_seed0}
POSTPROCESS_GPU=${POSTPROCESS_GPU:-3}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-12000}
ALLOW_LOW_FREE_GPU=${ALLOW_LOW_FREE_GPU:-0}
BURN_UP_STEP=${BURN_UP_STEP:-5000}

export OCT_SS_TRAIN_JSON=${OCT_SS_TRAIN_JSON:-/data2/ynz/cdpl_inputs/annotations/train_labeled.json}
export OCT_SS_UNLABEL_JSON=${OCT_SS_UNLABEL_JSON:-/data2/ynz/OCT_SS/datasets/annotations/train_unlabeled_v3_1.json}
export OCT_SS_TEST_JSON=${OCT_SS_TEST_JSON:-/data2/ynz/OCT_SS/datasets/annotations/test_v3_1.json}
export OCT_SS_TRAIN_IMAGE_ROOT=${OCT_SS_TRAIN_IMAGE_ROOT:-/data2/ynz/OCT_SS/datasets/images}
export OCT_SS_UNLABEL_IMAGE_ROOT=${OCT_SS_UNLABEL_IMAGE_ROOT:-/data2/ynz/OCT_SS/datasets/images}
export OCT_SS_IMAGE_ROOT=${OCT_SS_IMAGE_ROOT:-/data2/ynz/OCT_SS/datasets/images}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

SUMMARY_JSON="$RUN_ROOT/small_cutout_eval_summary.json"
GENERATED_DOC="$RUN_ROOT/small_cutout_results_generated.md"
AUDIT_JSON="$RUN_ROOT/small_cutout_result_artifact_audit.json"
LOG_PATH="$RUN_ROOT/postprocess_oct_ss_repro_pair_24k.log"
OK_PATH="$RUN_ROOT/postprocess.ok"

require_completed_run() {
  local run_name="$1"
  local run_dir="$RUN_ROOT/$run_name"
  test -f "$run_dir/complete.ok"
  test -f "$run_dir/model_final.pth"
  test "$(tr -d '[:space:]' < "$run_dir/exit_code.txt")" = "0"
  test "$(tr -d '[:space:]' < "$run_dir/git_commit.txt")" = "$TRAINING_COMMIT"
}

require_completed_run "$UBT_RUN"
require_completed_run "$CDPL_RUN"

if [[ "$ALLOW_LOW_FREE_GPU" != "1" ]]; then
  free_mib=$(nvidia-smi -i "$POSTPROCESS_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if (( free_mib < MIN_GPU_FREE_MIB )); then
    echo "refuse postprocess: GPU $POSTPROCESS_GPU has only ${free_mib} MiB free (required ${MIN_GPU_FREE_MIB} MiB)" >&2
    exit 2
  fi
fi

mkdir -p "$RUN_ROOT"
cd "$REPO_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

{
  echo "postprocess_started_at: $(date '+%F %T')"
  echo "repo_dir: $REPO_DIR"
  echo "run_root: $RUN_ROOT"
  echo "config_file: $CONFIG_FILE"
  echo "training_commit: $TRAINING_COMMIT"
  echo "postprocess_commit: $(git rev-parse HEAD)"
  echo "postprocess_gpu: $POSTPROCESS_GPU"
  echo "ubt_run: $UBT_RUN"
  echo "cdpl_run: $CDPL_RUN"
} > "$RUN_ROOT/postprocess_metadata.txt"
nvidia-smi > "$RUN_ROOT/postprocess_nvidia_smi_start.txt" 2>&1 || true

summary_cmd=(
  python scripts/experiments/summarize_step2_results.py
  --run-root "$RUN_ROOT"
  --annotations "$OCT_SS_TEST_JSON"
  --config-file "$CONFIG_FILE"
  --doc "$GENERATED_DOC"
  --summary-json "$SUMMARY_JSON"
  --runs "$UBT_RUN" "$CDPL_RUN"
  --eval-only
  --no-include-baseline-reference
  --eval-cuda-visible-devices "$POSTPROCESS_GPU"
)
printf '%q ' "${summary_cmd[@]}" > "$RUN_ROOT/postprocess_command.sh"
printf '\n' >> "$RUN_ROOT/postprocess_command.sh"
chmod +x "$RUN_ROOT/postprocess_command.sh"
"${summary_cmd[@]}" 2>&1 | tee "$LOG_PATH"

cdpl_best_info=$(
  python -c 'import json, sys; d={r["run"]: r for r in json.load(open(sys.argv[1]))}; b=d[sys.argv[2]]["best_periodic"]; print("{} {}".format(b["iteration"], b["checkpoint"]))' \
    "$SUMMARY_JSON" "$CDPL_RUN"
)
read -r cdpl_best_iteration cdpl_best_checkpoint <<< "$cdpl_best_info"

run_diagnostic() {
  local checkpoint="$1"
  local output="$2"
  local alpha="$3"
  local minimum="$4"
  local offset="$5"
  CUDA_VISIBLE_DEVICES="$POSTPROCESS_GPU" python scripts/experiments/diagnose_step2_pseudo_labels.py \
    --config-file "$CONFIG_FILE" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    OUTPUT_DIR "$RUN_ROOT/$CDPL_RUN/diagnostics_runtime" \
    MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.05 \
    SEMISUPNET.BBOX_THRESHOLD 0.5 \
    SEMISUPNET.CDPL_ALPHA_TAIL "$alpha" \
    SEMISUPNET.CDPL_MIN_CLS_THRESHOLD "$minimum" \
    SEMISUPNET.CDPL_MAX_THRESHOLD_OFFSET "$offset" \
    2>&1 | tee -a "$LOG_PATH"
}

run_diagnostic "$RUN_ROOT/$CDPL_RUN/model_final.pth" \
  "$RUN_ROOT/$CDPL_RUN/pseudo_label_diagnostics_final.json" 0.15 0.35 -1.0

if (( cdpl_best_iteration < BURN_UP_STEP )); then
  run_diagnostic "$cdpl_best_checkpoint" \
    "$RUN_ROOT/$CDPL_RUN/pseudo_label_diagnostics_best_test.json" 0.0 0.5 -1.0
else
  run_diagnostic "$cdpl_best_checkpoint" \
    "$RUN_ROOT/$CDPL_RUN/pseudo_label_diagnostics_best_test.json" 0.15 0.35 -1.0
fi

python scripts/experiments/audit_oct_ss_repro_pair_results.py \
  --run-root "$RUN_ROOT" \
  --summary-json "$SUMMARY_JSON" \
  --output "$AUDIT_JSON" \
  --ubt-run "$UBT_RUN" \
  --cdpl-run "$CDPL_RUN" \
  --expected-commit "$TRAINING_COMMIT" \
  --burn-up-step "$BURN_UP_STEP" \
  2>&1 | tee -a "$LOG_PATH"

nvidia-smi > "$RUN_ROOT/postprocess_nvidia_smi_end.txt" 2>&1 || true
date '+%F %T' > "$OK_PATH"
