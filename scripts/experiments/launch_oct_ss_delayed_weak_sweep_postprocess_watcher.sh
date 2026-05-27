#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data2/ynz/CDPL-src}
RUN_ROOT=${RUN_ROOT:-/data2/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/oct_ss_small_cutout_delayed_weak_cdpl_24k}
WAIT_SECONDS=${WAIT_SECONDS:-300}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-12000}
GPU_CANDIDATES=${GPU_CANDIDATES:-"3 4 5 6 7 1 2"}
DEFAULT_VARIANT_RUNS="cdpl_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed0 cdpl_dw_s10000_r4000_min045_a010_24000iter_1gpu_seed0 cdpl_dw_s10000_r4000_min040_a005_24000iter_1gpu_seed0 cdpl_dw_s12000_r4000_min045_a005_24000iter_1gpu_seed0 cdpl_dw_s8000_r4000_min045_a005_24000iter_1gpu_seed0 cdpl_dw_s10000_r8000_min045_a005_24000iter_1gpu_seed0 cdpl_dw_s10000_r4000_min048_a005_24000iter_1gpu_seed0"
VARIANT_RUNS=${VARIANT_RUNS:-$DEFAULT_VARIANT_RUNS}

WATCHER_PID="$RUN_ROOT/postprocess_watcher.pid"
WATCHER_LOG="$RUN_ROOT/postprocess_watcher.nohup.log"
WATCHER_METADATA="$RUN_ROOT/postprocess_watcher_metadata.txt"
WATCHER_OK="$RUN_ROOT/postprocess_watcher.ok"
POSTPROCESS_OK="$RUN_ROOT/postprocess.ok"

read -r -a variants <<< "$VARIANT_RUNS"

mkdir -p "$RUN_ROOT"
cd "$REPO_DIR"

if [[ "${1:-}" != "--watch" ]]; then
  if [[ -f "$POSTPROCESS_OK" ]]; then
    echo "skip watcher: postprocess already complete"
    exit 0
  fi
  if [[ -s "$WATCHER_PID" ]] && ps -p "$(cat "$WATCHER_PID")" >/dev/null 2>&1; then
    echo "skip watcher: already running as pid $(cat "$WATCHER_PID")"
    exit 0
  fi

  nohup env \
    REPO_DIR="$REPO_DIR" \
    RUN_ROOT="$RUN_ROOT" \
    VARIANT_RUNS="$VARIANT_RUNS" \
    WAIT_SECONDS="$WAIT_SECONDS" \
    MIN_GPU_FREE_MIB="$MIN_GPU_FREE_MIB" \
    GPU_CANDIDATES="$GPU_CANDIDATES" \
    bash scripts/experiments/launch_oct_ss_delayed_weak_sweep_postprocess_watcher.sh --watch \
    > "$WATCHER_LOG" 2>&1 &
  pid=$!
  echo "$pid" > "$WATCHER_PID"
  {
    echo "pid: $pid"
    echo "launched_at: $(date '+%F %T')"
    echo "repo_dir: $REPO_DIR"
    echo "run_root: $RUN_ROOT"
    echo "variant_runs: $VARIANT_RUNS"
    echo "wait_seconds: $WAIT_SECONDS"
    echo "min_gpu_free_mib: $MIN_GPU_FREE_MIB"
    echo "gpu_candidates: $GPU_CANDIDATES"
  } > "$WATCHER_METADATA"
  echo "launched delayed/weak sweep postprocess watcher pid=$pid"
  exit 0
fi

log_status() {
  local run_name="$1"
  local run_dir="$RUN_ROOT/$run_name"
  local state="running"
  if [[ -f "$run_dir/complete.ok" ]]; then
    state="complete"
  elif [[ -f "$run_dir/exit_code.txt" ]]; then
    state="exited"
  fi
  echo "$(date '+%F %T') $run_name state=$state"
}

failed_run() {
  local run_name="$1"
  local run_dir="$RUN_ROOT/$run_name"
  [[ -f "$run_dir/exit_code.txt" ]] && [[ "$(tr -d '[:space:]' < "$run_dir/exit_code.txt")" != "0" ]]
}

stopped_without_completion() {
  local run_name="$1"
  local run_dir="$RUN_ROOT/$run_name"
  [[ ! -f "$run_dir/complete.ok" ]] \
    && [[ -s "$run_dir/launcher.pid" ]] \
    && ! ps -p "$(cat "$run_dir/launcher.pid")" >/dev/null 2>&1
}

all_complete() {
  local run_name
  for run_name in "${variants[@]}"; do
    [[ -f "$RUN_ROOT/$run_name/complete.ok" ]] || return 1
  done
}

select_free_gpu() {
  local gpu
  local free_mib
  for gpu in $GPU_CANDIDATES; do
    free_mib=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)
    if [[ -n "$free_mib" ]] && (( free_mib >= MIN_GPU_FREE_MIB )); then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

while true; do
  for run_name in "${variants[@]}"; do
    log_status "$run_name"
  done

  for run_name in "${variants[@]}"; do
    if failed_run "$run_name"; then
      echo "stop watcher: $run_name exited with a non-zero code" >&2
      exit 1
    fi
    if stopped_without_completion "$run_name"; then
      echo "stop watcher: $run_name launcher stopped without complete.ok" >&2
      exit 1
    fi
  done

  if all_complete; then
    if [[ -f "$POSTPROCESS_OK" ]]; then
      date '+%F %T' > "$WATCHER_OK"
      echo "postprocess already complete"
      exit 0
    fi
    if gpu=$(select_free_gpu); then
      echo "$(date '+%F %T') starting postprocess on gpu=$gpu"
      POSTPROCESS_GPU="$gpu" bash scripts/experiments/postprocess_oct_ss_delayed_weak_sweep_24k.sh
      date '+%F %T' > "$WATCHER_OK"
      echo "postprocess finished"
      exit 0
    fi
    echo "$(date '+%F %T') waiting for a GPU with at least ${MIN_GPU_FREE_MIB} MiB free"
  fi

  sleep "$WAIT_SECONDS"
done
