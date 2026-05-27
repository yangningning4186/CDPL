#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data2/ynz/CDPL-src}
RUN_ROOT=${RUN_ROOT:-/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/oct_ss_small_cutout_delayed_weak_cdpl_24k}
MAX_ITER=${MAX_ITER:-24000}
SEED=${SEED:-0}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-12000}
ALLOW_LOW_FREE_GPU=${ALLOW_LOW_FREE_GPU:-0}
GPU_CANDIDATES=${GPU_CANDIDATES:-"1 2 3 4 5 6 7"}

DEFAULT_VARIANT_SPECS=$'cdpl_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed0:10000:4000:0.45:0.05\ncdpl_dw_s10000_r4000_min045_a010_24000iter_1gpu_seed0:10000:4000:0.45:0.10\ncdpl_dw_s10000_r4000_min040_a005_24000iter_1gpu_seed0:10000:4000:0.40:0.05\ncdpl_dw_s12000_r4000_min045_a005_24000iter_1gpu_seed0:12000:4000:0.45:0.05\ncdpl_dw_s8000_r4000_min045_a005_24000iter_1gpu_seed0:8000:4000:0.45:0.05\ncdpl_dw_s10000_r8000_min045_a005_24000iter_1gpu_seed0:10000:8000:0.45:0.05\ncdpl_dw_s10000_r4000_min048_a005_24000iter_1gpu_seed0:10000:4000:0.48:0.05'
VARIANT_SPECS=${VARIANT_SPECS:-$DEFAULT_VARIANT_SPECS}

mkdir -p "$RUN_ROOT"
cd "$REPO_DIR"

variant_specs=()
while IFS= read -r spec; do
  [[ -n "$spec" ]] && variant_specs+=("$spec")
done <<< "$VARIANT_SPECS"

gpu_candidates=()
for gpu in $GPU_CANDIDATES; do
  gpu_candidates+=("$gpu")
done

if (( ${#variant_specs[@]} == 0 )); then
  echo "refuse launch: no variant specs configured" >&2
  exit 2
fi
if (( ${#gpu_candidates[@]} < ${#variant_specs[@]} )); then
  echo "refuse launch: ${#variant_specs[@]} variants require ${#variant_specs[@]} GPU candidates, got ${#gpu_candidates[@]}" >&2
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
    echo "refuse sweep launch: $name is already complete" >&2
    return 1
  fi
  if [[ -f "$run_dir/launcher.pid" ]]; then
    local pid
    pid=$(cat "$run_dir/launcher.pid")
    if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
      echo "refuse sweep launch: $name is already running as pid $pid" >&2
      return 1
    fi
  fi
}

for idx in "${!variant_specs[@]}"; do
  IFS=: read -r name start_iter ramp_iters min_threshold alpha_tail <<< "${variant_specs[$idx]}"
  check_not_running_or_complete "$name"
  require_sufficient_gpu_memory "${gpu_candidates[$idx]}" "$name"
done

nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.free,utilization.gpu --format=csv,noheader \
  > "$RUN_ROOT/sweep_launch_nvidia_smi.txt" 2>&1 || true

{
  echo "launched_at: $(date '+%F %T')"
  echo "repo_dir: $REPO_DIR"
  echo "run_root: $RUN_ROOT"
  echo "max_iter: $MAX_ITER"
  echo "seed: $SEED"
  echo "gpu_candidates: $GPU_CANDIDATES"
  echo "variant_specs:"
  for idx in "${!variant_specs[@]}"; do
    echo "  ${gpu_candidates[$idx]} ${variant_specs[$idx]}"
  done
} > "$RUN_ROOT/sweep_launch_manifest.txt"

launch_one() {
  local gpu="$1"
  local name="$2"
  local start_iter="$3"
  local ramp_iters="$4"
  local min_threshold="$5"
  local alpha_tail="$6"
  local run_dir="$RUN_ROOT/$name"
  local log="$RUN_ROOT/${name}.nohup.log"

  mkdir -p "$run_dir"
  nohup env \
    REPO_DIR="$REPO_DIR" \
    RUN_ROOT="$RUN_ROOT" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    RUN_MODE=cdpl_sweep \
    RUN_NAME="$name" \
    SEED="$SEED" \
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
  local pid=$!
  echo "$pid" > "$run_dir/launcher.pid"
  {
    echo "name: $name"
    echo "mode: cdpl_sweep"
    echo "gpu: $gpu"
    echo "pid: $pid"
    echo "log: $log"
    echo "launched_at: $(date '+%F %T')"
    echo "cdpl_start_iter: $start_iter"
    echo "cdpl_ramp_iters: $ramp_iters"
    echo "cdpl_min_cls_threshold: $min_threshold"
    echo "cdpl_alpha_tail: $alpha_tail"
  } > "$run_dir/launcher_metadata.txt"
}

for idx in "${!variant_specs[@]}"; do
  IFS=: read -r name start_iter ramp_iters min_threshold alpha_tail <<< "${variant_specs[$idx]}"
  launch_one "${gpu_candidates[$idx]}" "$name" "$start_iter" "$ramp_iters" "$min_threshold" "$alpha_tail"
done

echo "launched ${#variant_specs[@]} delayed/weak CDPL variants under $RUN_ROOT"
