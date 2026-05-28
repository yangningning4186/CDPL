#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data2/ynz/CDPL-src}
RUN_ROOT=${RUN_ROOT:-/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/oct_ss_small_cutout_best_cdpl_multiseed_confirm_24k}
MAX_ITER=${MAX_ITER:-24000}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-12000}
ALLOW_LOW_FREE_GPU=${ALLOW_LOW_FREE_GPU:-0}
GPU_CANDIDATES=${GPU_CANDIDATES:-"2 3 4 5 6 7"}

DEFAULT_RUN_SPECS=$'ubt_ms_confirm_24000iter_1gpu_seed1:ubt_repro:1::::\nubt_ms_confirm_24000iter_1gpu_seed2:ubt_repro:2::::\nubt_ms_confirm_24000iter_1gpu_seed3:ubt_repro:3::::\ncdpl_best_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed1:cdpl_sweep:1:10000:4000:0.45:0.05\ncdpl_best_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed2:cdpl_sweep:2:10000:4000:0.45:0.05\ncdpl_best_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed3:cdpl_sweep:3:10000:4000:0.45:0.05'
RUN_SPECS=${RUN_SPECS:-$DEFAULT_RUN_SPECS}

mkdir -p "$RUN_ROOT"
cd "$REPO_DIR"

run_specs=()
while IFS= read -r spec; do
  [[ -n "$spec" ]] && run_specs+=("$spec")
done <<< "$RUN_SPECS"

gpu_candidates=()
for gpu in $GPU_CANDIDATES; do
  gpu_candidates+=("$gpu")
done

if (( ${#run_specs[@]} == 0 )); then
  echo "refuse launch: no run specs configured" >&2
  exit 2
fi
if (( ${#gpu_candidates[@]} < ${#run_specs[@]} )); then
  echo "refuse launch: ${#run_specs[@]} runs require ${#run_specs[@]} GPU candidates, got ${#gpu_candidates[@]}" >&2
  exit 2
fi

require_sufficient_gpu_memory() {
  local gpu="$1"
  local run_name="$2"
  local free_mib

  if [[ "$ALLOW_LOW_FREE_GPU" == "1" ]]; then
    return 0
  fi
  free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if (( free_mib < MIN_GPU_FREE_MIB )); then
    echo "refuse launch $run_name: GPU $gpu has only ${free_mib} MiB free (required ${MIN_GPU_FREE_MIB} MiB)" >&2
    return 1
  fi
}

check_not_running_or_complete() {
  local name="$1"
  local run_dir="$RUN_ROOT/$name"

  if [[ -f "$run_dir/complete.ok" ]]; then
    echo "refuse launch: $name is already complete" >&2
    return 1
  fi
  if [[ -f "$run_dir/launcher.pid" ]]; then
    local pid
    pid=$(cat "$run_dir/launcher.pid")
    if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
      echo "refuse launch: $name is already running as pid $pid" >&2
      return 1
    fi
  fi
}

for idx in "${!run_specs[@]}"; do
  IFS=: read -r name mode seed start_iter ramp_iters min_threshold alpha_tail <<< "${run_specs[$idx]}"
  check_not_running_or_complete "$name"
  require_sufficient_gpu_memory "${gpu_candidates[$idx]}" "$name"
done

nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.free,utilization.gpu --format=csv,noheader \
  > "$RUN_ROOT/multiseed_launch_nvidia_smi.txt" 2>&1 || true

{
  echo "launched_at: $(date '+%F %T')"
  echo "repo_dir: $REPO_DIR"
  echo "run_root: $RUN_ROOT"
  echo "max_iter: $MAX_ITER"
  echo "gpu_candidates: $GPU_CANDIDATES"
  echo "run_specs:"
  for idx in "${!run_specs[@]}"; do
    echo "  ${gpu_candidates[$idx]} ${run_specs[$idx]}"
  done
} > "$RUN_ROOT/multiseed_launch_manifest.txt"

launch_one() {
  local gpu="$1"
  local name="$2"
  local mode="$3"
  local seed="$4"
  local start_iter="${5:-}"
  local ramp_iters="${6:-}"
  local min_threshold="${7:-}"
  local alpha_tail="${8:-}"
  local run_dir="$RUN_ROOT/$name"
  local log="$RUN_ROOT/${name}.nohup.log"

  mkdir -p "$run_dir"
  if [[ "$mode" == "ubt_repro" ]]; then
    nohup env \
      REPO_DIR="$REPO_DIR" \
      RUN_ROOT="$RUN_ROOT" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      RUN_MODE="$mode" \
      RUN_NAME="$name" \
      SEED="$seed" \
      MAX_ITER="$MAX_ITER" \
      BBOX_THRESHOLD=0.5 \
      BURN_UP_STEP=5000 \
      bash scripts/experiments/run_oct_ss_repro_24k_run.sh \
      > "$log" 2>&1 &
  else
    nohup env \
      REPO_DIR="$REPO_DIR" \
      RUN_ROOT="$RUN_ROOT" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      RUN_MODE="$mode" \
      RUN_NAME="$name" \
      SEED="$seed" \
      MAX_ITER="$MAX_ITER" \
      BBOX_THRESHOLD=0.5 \
      BURN_UP_STEP=5000 \
      CDPL_TAU_BASE=0.5 \
      CDPL_ALPHA_TAIL="$alpha_tail" \
      CDPL_MIN_CLS_THRESHOLD="$min_threshold" \
      CDPL_MAX_THRESHOLD_OFFSET=-1.0 \
      CDPL_START_ITER="$start_iter" \
      CDPL_RAMP_ITERS="$ramp_iters" \
      bash scripts/experiments/run_oct_ss_repro_24k_run.sh \
      > "$log" 2>&1 &
  fi
  local pid=$!
  echo "$pid" > "$run_dir/launcher.pid"
  {
    echo "name: $name"
    echo "mode: $mode"
    echo "seed: $seed"
    echo "gpu: $gpu"
    echo "pid: $pid"
    echo "log: $log"
    echo "launched_at: $(date '+%F %T')"
    [[ -n "$start_iter" ]] && echo "cdpl_start_iter: $start_iter"
    [[ -n "$ramp_iters" ]] && echo "cdpl_ramp_iters: $ramp_iters"
    [[ -n "$min_threshold" ]] && echo "cdpl_min_cls_threshold: $min_threshold"
    [[ -n "$alpha_tail" ]] && echo "cdpl_alpha_tail: $alpha_tail"
  } > "$run_dir/launcher_metadata.txt"
}

for idx in "${!run_specs[@]}"; do
  IFS=: read -r name mode seed start_iter ramp_iters min_threshold alpha_tail <<< "${run_specs[$idx]}"
  launch_one "${gpu_candidates[$idx]}" "$name" "$mode" "$seed" "$start_iter" "$ramp_iters" "$min_threshold" "$alpha_tail"
done

echo "launched ${#run_specs[@]} OCT-SS multiseed confirmation runs under $RUN_ROOT"
