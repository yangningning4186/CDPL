#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data2/ynz/CDPL-src}
RUN_ROOT=${RUN_ROOT:-/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/oct_ss_small_cutout_ubt_vs_cdpl_24k}
MAX_ITER=${MAX_ITER:-24000}
BASELINE_GPU=${BASELINE_GPU:-1}
CDPL_GPU=${CDPL_GPU:-2}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-12000}
ALLOW_LOW_FREE_GPU=${ALLOW_LOW_FREE_GPU:-0}

BASELINE_NAME=${BASELINE_NAME:-ubt_repro_${MAX_ITER}iter_1gpu_seed0}
CDPL_NAME=${CDPL_NAME:-cdpl_calib_repro_${MAX_ITER}iter_1gpu_seed0}

mkdir -p "$RUN_ROOT"
cd "$REPO_DIR"

if [[ "$BASELINE_GPU" == "$CDPL_GPU" ]]; then
  echo "refuse launch: baseline and CDPL runs must use distinct GPUs" >&2
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
    echo "refuse pair launch: $name is already complete" >&2
    return 1
  fi
  if [[ -f "$run_dir/launcher.pid" ]]; then
    local pid
    pid=$(cat "$run_dir/launcher.pid")
    if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
      echo "refuse pair launch: $name is already running as pid $pid" >&2
      return 1
    fi
  fi
}

check_not_running_or_complete "$BASELINE_NAME"
check_not_running_or_complete "$CDPL_NAME"
require_sufficient_gpu_memory "$BASELINE_GPU" "$BASELINE_NAME"
require_sufficient_gpu_memory "$CDPL_GPU" "$CDPL_NAME"

nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,utilization.gpu --format=csv,noheader \
  > "$RUN_ROOT/pair_launch_nvidia_smi.txt" 2>&1 || true

launch_one() {
  local gpu="$1"
  local mode="$2"
  local name="$3"
  local run_dir="$RUN_ROOT/$name"
  local log="$RUN_ROOT/${name}.nohup.log"

  mkdir -p "$run_dir"
  nohup env \
    REPO_DIR="$REPO_DIR" \
    RUN_ROOT="$RUN_ROOT" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    RUN_MODE="$mode" \
    RUN_NAME="$name" \
    MAX_ITER="$MAX_ITER" \
    bash scripts/experiments/run_oct_ss_repro_24k_run.sh \
    > "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$run_dir/launcher.pid"
  {
    echo "name: $name"
    echo "mode: $mode"
    echo "gpu: $gpu"
    echo "pid: $pid"
    echo "log: $log"
    echo "launched_at: $(date '+%F %T')"
  } > "$run_dir/launcher_metadata.txt"
}

launch_one "$BASELINE_GPU" ubt_repro "$BASELINE_NAME"
launch_one "$CDPL_GPU" cdpl_calib_repro "$CDPL_NAME"
echo "launched OCT-SS small-cutout 24k pair under $RUN_ROOT"
