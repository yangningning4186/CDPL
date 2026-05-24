#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/data4/ynz/CDPL-src}
STEP1_ROOT=${STEP1_ROOT:-/data4/ynz/cdpl_runs/step1_baseline_vs_cdpl_calib}
STEP2_ROOT=${STEP2_ROOT:-/data4/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib}
CONDA_SH=${CONDA_SH:-/data4/ynz/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-ubt}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DIAG_LIMIT=${DIAG_LIMIT:-0}

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

OUT_DIR="$STEP2_ROOT/diagnostics"
mkdir -p "$OUT_DIR"
export OUT_DIR

run_diag() {
  local name="$1"
  local checkpoint="$2"
  local output="$OUT_DIR/${name}.json"
  local log="$OUT_DIR/${name}.log"

  if [[ ! -f "$checkpoint" ]]; then
    echo "skip $name: missing checkpoint $checkpoint" | tee "$log"
    return 0
  fi

  local cmd=(
    python scripts/experiments/diagnose_step2_pseudo_labels.py
    --config-file configs/oct_ss_cdpl.yaml
    --checkpoint "$checkpoint"
    --output "$output"
    --limit "$DIAG_LIMIT"
    MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.05
    SEMISUPNET.BBOX_THRESHOLD 0.5
    SEMISUPNET.CDPL_ALPHA_TAIL 0.15
    SEMISUPNET.CDPL_MIN_CLS_THRESHOLD 0.35
    SEMISUPNET.CDPL_MAX_THRESHOLD_OFFSET -1.0
  )

  printf '%q ' "${cmd[@]}" > "$OUT_DIR/${name}.command.sh"
  printf '\n' >> "$OUT_DIR/${name}.command.sh"
  chmod +x "$OUT_DIR/${name}.command.sh"
  "${cmd[@]}" 2>&1 | tee "$log"
}

run_diag \
  step1_ubt_baseline_seed0_best_iter17999 \
  "$STEP1_ROOT/ubt_baseline_1gpu_seed0/model_0017999.pth"

run_diag \
  step1_cdpl_calib_seed0_best_iter11999 \
  "$STEP1_ROOT/cdpl_calib_1gpu_seed0/model_0011999.pth"

run_diag \
  step1_ubt_baseline_seed1_best_iter20999 \
  "$STEP1_ROOT/ubt_baseline_1gpu_seed1/model_0020999.pth"

run_diag \
  step1_cdpl_calib_seed1_best_iter11999 \
  "$STEP1_ROOT/cdpl_calib_1gpu_seed1/model_0011999.pth"

python - <<'PY' > "$OUT_DIR/step1_pseudo_label_diagnostics_manifest.json"
import json
import os
from pathlib import Path

out_dir = Path(os.environ.get("OUT_DIR", ""))
if not out_dir:
    out_dir = Path("/data4/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib/diagnostics")

items = []
for path in sorted(out_dir.glob("step1_*_best_iter*.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    items.append(
        {
            "file": str(path),
            "checkpoint": report.get("checkpoint"),
            "num_images": report.get("num_images"),
            "totals": report.get("totals"),
            "focus_classes": {
                name: {
                    "teacher_boxes": data.get("teacher_boxes"),
                    "ubt_kept": data.get("ubt_kept"),
                    "cdpl_kept": data.get("cdpl_kept"),
                    "cdpl_minus_ubt": data.get("cdpl_minus_ubt"),
                    "ubt_keep_ratio": data.get("ubt_keep_ratio"),
                    "cdpl_keep_ratio": data.get("cdpl_keep_ratio"),
                    "cdpl_drop_ratio": data.get("cdpl_drop_ratio"),
                    "thresholds": data.get("thresholds"),
                }
                for name, data in (report.get("focus_classes") or {}).items()
            },
        }
    )

print(json.dumps(items, indent=2, ensure_ascii=False))
PY

echo "Wrote diagnostics to $OUT_DIR"
