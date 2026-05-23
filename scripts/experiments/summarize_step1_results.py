#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


STANDARD_KEYS = {
    "AP": "mAP",
    "AP50": "AP50",
    "AP75": "AP75",
    "APs": "AP_small",
}


def load_last_metrics(metrics_path):
    if not metrics_path.exists():
        return {}

    latest = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key, value in record.items():
                if key.startswith("bbox/"):
                    latest[key] = value
    return latest


def finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean_valid(values):
    valid = [float(v) for v in values if float(v) > -1]
    if not valid:
        return None
    return sum(valid) / len(valid) * 100.0


def compute_ap30(annotations, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    coco_gt = COCO(str(annotations))
    coco_dt = coco_gt.loadRes(str(predictions))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.iouThrs = np.array([0.30])
    evaluator.evaluate()
    evaluator.accumulate()

    params = evaluator.params
    area_index = list(params.areaRngLbl).index("all")
    max_det_index = list(params.maxDets).index(100)
    precision = evaluator.eval["precision"]

    ap30 = mean_valid(precision[0, :, :, area_index, max_det_index].reshape(-1))

    categories = coco_gt.loadCats(params.catIds)
    per_class = {}
    for class_index, category in enumerate(categories):
        class_precision = precision[0, :, class_index, area_index, max_det_index]
        per_class[category["name"]] = mean_valid(class_precision)

    return {"AP30": ap30, "AP30_per_class": per_class}


def collect_run(run_dir, annotations):
    metrics = load_last_metrics(run_dir / "metrics.json")
    predictions = run_dir / "inference" / "coco_instances_results.json"

    result = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "complete": (run_dir / "complete.ok").exists(),
        "exit_code": None,
        "checkpoint": str(run_dir / "model_final.pth"),
        "predictions": str(predictions),
        "metrics_json": str(run_dir / "metrics.json"),
        "git_commit": None,
        "metrics": {},
        "per_class_ap": {},
    }

    exit_code_path = run_dir / "exit_code.txt"
    if exit_code_path.exists():
        result["exit_code"] = exit_code_path.read_text(encoding="utf-8").strip()

    commit_path = run_dir / "git_commit.txt"
    if commit_path.exists():
        result["git_commit"] = commit_path.read_text(encoding="utf-8").strip()

    for source_key, output_key in STANDARD_KEYS.items():
        result["metrics"][output_key] = finite_or_none(metrics.get(f"bbox/{source_key}"))

    for key, value in sorted(metrics.items()):
        if not key.startswith("bbox/AP-"):
            continue
        result["per_class_ap"][key.replace("bbox/AP-", "")] = finite_or_none(value)

    if predictions.exists():
        ap30 = compute_ap30(annotations, predictions)
        result["metrics"]["AP30"] = finite_or_none(ap30["AP30"])
        result["per_class_ap30"] = ap30["AP30_per_class"]
    else:
        result["metrics"]["AP30"] = None
        result["per_class_ap30"] = {}

    result["artifacts_exist"] = {
        "config": (run_dir / "config_input.yaml").exists(),
        "command": (run_dir / "launch_command.sh").exists(),
        "log": (run_dir / "train.log").exists(),
        "checkpoint": (run_dir / "model_final.pth").exists(),
        "predictions": predictions.exists(),
        "git_commit": (run_dir / "git_commit.txt").exists(),
    }
    return result


def format_value(value):
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def write_markdown(doc_path, run_root, results):
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step 1: UBT Baseline vs CDPL Class Calibration",
        "",
        "This document is generated from server run artifacts under:",
        "",
        f"`{run_root}`",
        "",
        "Important scope note: this step evaluates the currently implemented CDPL class-calibration stage. It is not Full CDPL with localization-aware loss routing.",
        "",
        "## Runs",
        "",
        "| Run | Commit | Exit | Complete | mAP | AP30 | AP50 | AP75 | AP_small |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        metrics = result["metrics"]
        lines.append(
            "| {run} | {commit} | {exit_code} | {complete} | {mAP} | {AP30} | {AP50} | {AP75} | {AP_small} |".format(
                run=result["run"],
                commit=(result["git_commit"] or "NA")[:12],
                exit_code=result["exit_code"],
                complete="yes" if result["complete"] else "no",
                mAP=format_value(metrics.get("mAP")),
                AP30=format_value(metrics.get("AP30")),
                AP50=format_value(metrics.get("AP50")),
                AP75=format_value(metrics.get("AP75")),
                AP_small=format_value(metrics.get("AP_small")),
            )
        )

    lines.extend(["", "## Artifact Check", ""])
    for result in results:
        lines.append(f"### {result['run']}")
        lines.append("")
        for name, exists in result["artifacts_exist"].items():
            lines.append(f"- {name}: {'yes' if exists else 'no'}")
        lines.append(f"- run directory: `{result['run_dir']}`")
        lines.append("")

    lines.extend(["## Per-Class AP", ""])
    class_names = sorted({name for result in results for name in result["per_class_ap"]})
    if class_names:
        header = "| Class | " + " | ".join(result["run"] for result in results) + " |"
        sep = "| --- | " + " | ".join("---:" for _ in results) + " |"
        lines.extend([header, sep])
        for class_name in class_names:
            row = [class_name]
            for result in results:
                row.append(format_value(result["per_class_ap"].get(class_name)))
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("Per-class AP was not found in `metrics.json` yet.")

    lines.extend(["", "## Judgment", ""])
    complete_results = [result for result in results if result["complete"]]
    if len(complete_results) == len(results) and len(results) >= 2:
        baseline = next((r for r in results if r["run"].startswith("ubt_baseline")), None)
        cdpl = next((r for r in results if r["run"].startswith("cdpl_calib")), None)
        if baseline and cdpl:
            base_map = baseline["metrics"].get("mAP")
            cdpl_map = cdpl["metrics"].get("mAP")
            base_ap75 = baseline["metrics"].get("AP75")
            cdpl_ap75 = cdpl["metrics"].get("AP75")
            if base_map is not None and cdpl_map is not None:
                delta_map = cdpl_map - base_map
                delta_ap75 = (
                    cdpl_ap75 - base_ap75
                    if base_ap75 is not None and cdpl_ap75 is not None
                    else None
                )
                lines.append(
                    f"CDPL class calibration changed mAP by {delta_map:+.3f}"
                    + (
                        f" and AP75 by {delta_ap75:+.3f}."
                        if delta_ap75 is not None
                        else "."
                    )
                )
                lines.append(
                    "Full CDPL remains worth implementing if class calibration improves tail/per-class AP, AP75, or preserves mAP while improving rare classes. If it regresses broadly, inspect pseudo-label retention before adding localization-aware routing."
                )
            else:
                lines.append(
                    "Both runs completed, but summary metrics are missing. Inspect `metrics.json` and evaluator output before deciding on Full CDPL."
                )
    else:
        lines.append(
            "Training/evaluation is still incomplete. Do not use this document for conclusions yet."
        )

    lines.append("")
    doc_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--doc", required=True)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["ubt_baseline_4gpu_seed0", "cdpl_calib_4gpu_seed0"],
        help="Run directory names under --run-root to summarize.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    annotations = Path(args.annotations)
    doc_path = Path(args.doc)

    run_dirs = [run_root / run_name for run_name in args.runs]
    results = [collect_run(run_dir, annotations) for run_dir in run_dirs]

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(doc_path, run_root, results)
    print(f"Wrote {summary_path}")
    print(f"Wrote {doc_path}")


if __name__ == "__main__":
    main()
