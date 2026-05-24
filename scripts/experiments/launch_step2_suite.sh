#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data4/ynz/CDPL-src}
RUN_ROOT=${RUN_ROOT:-/data4/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib}
MAX_ITER=${MAX_ITER:-24000}

BASELINE_GPU=${BASELINE_GPU:-2}
LATE_GPU=${LATE_GPU:-0}
CLAMP_GPU=${CLAMP_GPU:-1}
SUPPLEMENTAL_GPU=${SUPPLEMENTAL_GPU:-3}
RUN_SUPPLEMENTAL_SEED1=${RUN_SUPPLEMENTAL_SEED1:-0}

mkdir -p "$RUN_ROOT"
cd "$REPO_DIR"

launch_one() {
  local gpu="$1"
  local mode="$2"
  local name="$3"
  local seed="$4"
  local run_dir="$RUN_ROOT/$name"
  local log="$RUN_ROOT/${name}.nohup.log"

  mkdir -p "$run_dir"
  if [[ -f "$run_dir/complete.ok" ]]; then
    echo "skip $name: already complete"
    return 0
  fi
  if [[ -f "$run_dir/launcher.pid" ]]; then
    local old_pid
    old_pid=$(cat "$run_dir/launcher.pid")
    if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
      echo "skip $name: pid $old_pid is still running"
      return 0
    fi
  fi

  echo "launch $name on GPU $gpu"
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    RUN_MODE="$mode" \
    RUN_NAME="$name" \
    SEED="$seed" \
    NUM_GPUS=1 \
    MAX_ITER="$MAX_ITER" \
    bash scripts/experiments/run_step2_cdpl_run.sh \
    > "$log" 2>&1 &
  echo "$!" > "$run_dir/launcher.pid"
  {
    echo "name: $name"
    echo "mode: $mode"
    echo "seed: $seed"
    echo "gpu: $gpu"
    echo "pid: $!"
    echo "log: $log"
    echo "launched_at: $(date '+%F %T')"
  } > "$run_dir/launcher_metadata.txt"
}

nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,utilization.gpu --format=csv,noheader \
  > "$RUN_ROOT/suite_launch_nvidia_smi.txt" 2>&1 || true

launch_one "$BASELINE_GPU" ubt_baseline ubt_baseline_24k_1gpu_seed0 0
launch_one "$LATE_GPU" late_weak late_weak_1gpu_seed0 0
launch_one "$CLAMP_GPU" floor_clamp floor_clamp_1gpu_seed0 0

if [[ "$RUN_SUPPLEMENTAL_SEED1" == "1" ]]; then
  launch_one "$SUPPLEMENTAL_GPU" late_weak late_weak_1gpu_seed1 1
fi

echo "Launched Step2 suite under $RUN_ROOT"
