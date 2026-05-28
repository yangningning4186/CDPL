#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data2/ynz/CDPL-src}
RUN_ROOT=${RUN_ROOT:-/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/oct_ss_small_cutout_best_cdpl_multiseed_confirm_24k}
CONDA_SH=${CONDA_SH:-/data2/ynz/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-oct_ss}
CONFIG_FILE=${CONFIG_FILE:-configs/oct_ss_ubt_repro.yaml}
EXPECTED_COMMIT=${EXPECTED_COMMIT:-}
POSTPROCESS_GPU=${POSTPROCESS_GPU:-3}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-12000}
ALLOW_LOW_FREE_GPU=${ALLOW_LOW_FREE_GPU:-0}
SEEDS=${SEEDS:-"1 2 3"}

DEFAULT_UBT_RUNS="ubt_ms_confirm_24000iter_1gpu_seed1 ubt_ms_confirm_24000iter_1gpu_seed2 ubt_ms_confirm_24000iter_1gpu_seed3"
DEFAULT_CDPL_RUNS="cdpl_best_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed1 cdpl_best_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed2 cdpl_best_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed3"
UBT_RUNS=${UBT_RUNS:-$DEFAULT_UBT_RUNS}
CDPL_RUNS=${CDPL_RUNS:-$DEFAULT_CDPL_RUNS}

export OCT_SS_TRAIN_JSON=${OCT_SS_TRAIN_JSON:-/data2/ynz/cdpl_inputs/annotations/train_labeled.json}
export OCT_SS_UNLABEL_JSON=${OCT_SS_UNLABEL_JSON:-/data2/ynz/OCT_SS/datasets/annotations/train_unlabeled_v3_1.json}
export OCT_SS_TEST_JSON=${OCT_SS_TEST_JSON:-/data2/ynz/OCT_SS/datasets/annotations/test_v3_1.json}
export OCT_SS_TRAIN_IMAGE_ROOT=${OCT_SS_TRAIN_IMAGE_ROOT:-/data2/ynz/OCT_SS/datasets/images}
export OCT_SS_UNLABEL_IMAGE_ROOT=${OCT_SS_UNLABEL_IMAGE_ROOT:-/data2/ynz/OCT_SS/datasets/images}
export OCT_SS_IMAGE_ROOT=${OCT_SS_IMAGE_ROOT:-/data2/ynz/OCT_SS/datasets/images}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

SUMMARY_JSON="$RUN_ROOT/best_cdpl_multiseed_eval_summary.json"
GENERATED_DOC="$RUN_ROOT/best_cdpl_multiseed_results_generated.md"
AGGREGATE_JSON="$RUN_ROOT/best_cdpl_multiseed_aggregate_summary.json"
AGGREGATE_MD="$RUN_ROOT/best_cdpl_multiseed_aggregate_summary.md"
AUDIT_JSON="$RUN_ROOT/best_cdpl_multiseed_artifact_audit.json"
LOG_PATH="$RUN_ROOT/postprocess_oct_ss_best_cdpl_multiseed_confirm_24k.log"
OK_PATH="$RUN_ROOT/postprocess.ok"

read -r -a seeds <<< "$SEEDS"
read -r -a ubt_runs <<< "$UBT_RUNS"
read -r -a cdpl_runs <<< "$CDPL_RUNS"
runs=("${ubt_runs[@]}" "${cdpl_runs[@]}")

require_completed_run() {
  local run_dir="$1"
  local expected_commit="$2"
  test -f "$run_dir/complete.ok"
  test -f "$run_dir/model_final.pth"
  test "$(tr -d '[:space:]' < "$run_dir/exit_code.txt")" = "0"
  test "$(tr -d '[:space:]' < "$run_dir/git_commit.txt")" = "$expected_commit"
}

mkdir -p "$RUN_ROOT"
if [[ -z "$EXPECTED_COMMIT" ]]; then
  EXPECTED_COMMIT=$(tr -d '[:space:]' < "$RUN_ROOT/${runs[0]}/git_commit.txt")
fi
for run_name in "${runs[@]}"; do
  require_completed_run "$RUN_ROOT/$run_name" "$EXPECTED_COMMIT"
done

if [[ "$ALLOW_LOW_FREE_GPU" != "1" ]]; then
  free_mib=$(nvidia-smi -i "$POSTPROCESS_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if (( free_mib < MIN_GPU_FREE_MIB )); then
    echo "refuse postprocess: GPU $POSTPROCESS_GPU has only ${free_mib} MiB free (required ${MIN_GPU_FREE_MIB} MiB)" >&2
    exit 2
  fi
fi

cd "$REPO_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

{
  echo "postprocess_started_at: $(date '+%F %T')"
  echo "repo_dir: $REPO_DIR"
  echo "run_root: $RUN_ROOT"
  echo "config_file: $CONFIG_FILE"
  echo "expected_commit: $EXPECTED_COMMIT"
  echo "postprocess_commit: $(git rev-parse HEAD)"
  echo "postprocess_gpu: $POSTPROCESS_GPU"
  echo "seeds: $SEEDS"
  echo "ubt_runs: $UBT_RUNS"
  echo "cdpl_runs: $CDPL_RUNS"
} > "$RUN_ROOT/postprocess_metadata.txt"
nvidia-smi > "$RUN_ROOT/postprocess_nvidia_smi_start.txt" 2>&1 || true

summary_cmd=(
  python scripts/experiments/summarize_step2_results.py
  --run-root "$RUN_ROOT"
  --annotations "$OCT_SS_TEST_JSON"
  --config-file "$CONFIG_FILE"
  --doc "$GENERATED_DOC"
  --summary-json "$SUMMARY_JSON"
  --runs "${runs[@]}"
  --eval-only
  --no-include-baseline-reference
  --eval-cuda-visible-devices "$POSTPROCESS_GPU"
)
printf '%q ' "${summary_cmd[@]}" > "$RUN_ROOT/postprocess_command.sh"
printf '\n' >> "$RUN_ROOT/postprocess_command.sh"
chmod +x "$RUN_ROOT/postprocess_command.sh"
"${summary_cmd[@]}" 2>&1 | tee "$LOG_PATH"

variant_info() {
  local run_name="$1"
  python - "$SUMMARY_JSON" "$run_name" <<'PY'
import json
import sys

summary_path, run_name = sys.argv[1:3]
results = {item["run"]: item for item in json.load(open(summary_path, encoding="utf-8"))}
item = results[run_name]
best = item["best_periodic"]
overrides = item.get("config_overrides") or {}
start = int(overrides.get("cdpl_start_iter", 10000))
ramp = int(overrides.get("cdpl_ramp_iters", 4000))
alpha = float(overrides.get("cdpl_alpha_tail", 0.05))
minimum = float(overrides.get("cdpl_min_cls_threshold", 0.45))
offset = float(overrides.get("cdpl_max_threshold_offset", -1.0))

def ramp_factor(iteration):
    iteration = int(iteration)
    if iteration < start:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, max(0.0, float(iteration - start + 1) / float(ramp)))

print(
    best["iteration"],
    best["checkpoint"],
    alpha,
    minimum,
    offset,
    ramp_factor(best["iteration"]),
)
PY
}

run_diagnostic() {
  local checkpoint="$1"
  local output="$2"
  local alpha="$3"
  local minimum="$4"
  local offset="$5"
  local ramp_factor="$6"
  CUDA_VISIBLE_DEVICES="$POSTPROCESS_GPU" python scripts/experiments/diagnose_step2_pseudo_labels.py \
    --config-file "$CONFIG_FILE" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    --cdpl-ramp-factor "$ramp_factor" \
    OUTPUT_DIR "$RUN_ROOT/diagnostics_runtime" \
    MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.05 \
    SEMISUPNET.BBOX_THRESHOLD 0.5 \
    SEMISUPNET.CDPL_ALPHA_TAIL "$alpha" \
    SEMISUPNET.CDPL_MIN_CLS_THRESHOLD "$minimum" \
    SEMISUPNET.CDPL_MAX_THRESHOLD_OFFSET "$offset" \
    2>&1 | tee -a "$LOG_PATH"
}

for run_name in "${cdpl_runs[@]}"; do
  read -r best_iteration best_checkpoint alpha_tail min_threshold offset best_ramp <<< "$(variant_info "$run_name")"
  echo "diagnostics $run_name best_iter=$best_iteration best_ramp=$best_ramp" | tee -a "$LOG_PATH"
  run_diagnostic "$RUN_ROOT/$run_name/model_final.pth" \
    "$RUN_ROOT/$run_name/pseudo_label_diagnostics_final.json" \
    "$alpha_tail" "$min_threshold" "$offset" 1.0
  run_diagnostic "$best_checkpoint" \
    "$RUN_ROOT/$run_name/pseudo_label_diagnostics_best_test.json" \
    "$alpha_tail" "$min_threshold" "$offset" "$best_ramp"
done

python scripts/experiments/summarize_oct_ss_best_cdpl_multiseed_confirm.py \
  --summary-json "$SUMMARY_JSON" \
  --run-root "$RUN_ROOT" \
  --output-json "$AGGREGATE_JSON" \
  --output-md "$AGGREGATE_MD" \
  --seeds "${seeds[@]}" \
  2>&1 | tee -a "$LOG_PATH"

python scripts/experiments/audit_oct_ss_best_cdpl_multiseed_confirm_results.py \
  --run-root "$RUN_ROOT" \
  --summary-json "$SUMMARY_JSON" \
  --aggregate-json "$AGGREGATE_JSON" \
  --output "$AUDIT_JSON" \
  --ubt-runs "${ubt_runs[@]}" \
  --cdpl-runs "${cdpl_runs[@]}" \
  --seeds "${seeds[@]}" \
  --expected-commit "$EXPECTED_COMMIT" \
  2>&1 | tee -a "$LOG_PATH"

nvidia-smi > "$RUN_ROOT/postprocess_nvidia_smi_end.txt" 2>&1 || true
date '+%F %T' > "$OK_PATH"
