#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


STANDARD_KEYS = {
    "mAP": "mAP",
    "AP30": "AP30",
    "AP50": "AP50",
    "AP75": "AP75",
    "AP_small": "AP_small",
}

DEFAULT_RUN_ROOT = "/data4/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib"
DEFAULT_ANNOTATIONS = "/data4/ynz/OCT_SS-main/datasets/annotations/test_v3_1.json"


def finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean_valid(values):
    valid = [float(value) for value in values if float(value) > -1]
    if not valid:
        return None
    return sum(valid) / len(valid) * 100.0


def metric_value(precision, params, iou_thr=None, area_label="all", class_index=None):
    area_index = list(params.areaRngLbl).index(area_label)
    max_det_index = list(params.maxDets).index(100)
    if iou_thr is None:
        iou_indices = slice(None)
    else:
        matches = [idx for idx, value in enumerate(params.iouThrs) if abs(float(value) - iou_thr) < 1e-6]
        if not matches:
            return None
        iou_indices = matches

    class_indices = [class_index] if class_index is not None else slice(None)
    values = precision[iou_indices, :, class_indices, area_index, max_det_index].reshape(-1)
    return mean_valid(values)


def compute_coco_metrics(annotations, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    annotations = Path(annotations)
    predictions = Path(predictions)
    if not predictions.exists():
        return None

    coco_gt = COCO(str(annotations))
    coco_dt = coco_gt.loadRes(str(predictions))

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval["precision"]
    params = evaluator.params
    categories = coco_gt.loadCats(params.catIds)

    metrics = {
        "mAP": metric_value(precision, params),
        "AP50": metric_value(precision, params, iou_thr=0.50),
        "AP75": metric_value(precision, params, iou_thr=0.75),
        "AP_small": metric_value(precision, params, area_label="small"),
    }
    per_class_ap = {}
    for class_index, category in enumerate(categories):
        per_class_ap[category["name"]] = metric_value(precision, params, class_index=class_index)

    ap30_evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    ap30_evaluator.params.iouThrs = np.array([0.30])
    ap30_evaluator.evaluate()
    ap30_evaluator.accumulate()
    ap30_precision = ap30_evaluator.eval["precision"]
    ap30_params = ap30_evaluator.params

    metrics["AP30"] = metric_value(ap30_precision, ap30_params, iou_thr=0.30)
    per_class_ap30 = {}
    for class_index, category in enumerate(coco_gt.loadCats(ap30_params.catIds)):
        per_class_ap30[category["name"]] = metric_value(
            ap30_precision,
            ap30_params,
            iou_thr=0.30,
            class_index=class_index,
        )

    return {
        "metrics": {key: finite_or_none(value) for key, value in metrics.items()},
        "per_class_ap": {key: finite_or_none(value) for key, value in per_class_ap.items()},
        "per_class_ap30": {key: finite_or_none(value) for key, value in per_class_ap30.items()},
    }


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(path):
    metadata = {}
    path = Path(path)
    if not path.exists():
        return metadata
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def load_periodic_records(metrics_path):
    metrics_path = Path(metrics_path)
    records = []
    if not metrics_path.exists():
        return records
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = finite_or_none(record.get("bbox/AP"))
            iteration = record.get("iteration")
            if value is None or iteration is None:
                continue
            records.append(
                {
                    "iteration": int(iteration),
                    "bbox/AP": value,
                    "metrics": {
                        "mAP": finite_or_none(record.get("bbox/AP")),
                        "AP50": finite_or_none(record.get("bbox/AP50")),
                        "AP75": finite_or_none(record.get("bbox/AP75")),
                        "AP_small": finite_or_none(record.get("bbox/APs")),
                    },
                    "per_class_ap": {
                        key.replace("bbox/AP-", ""): finite_or_none(val)
                        for key, val in sorted(record.items())
                        if key.startswith("bbox/AP-")
                    },
                }
            )
    return records


def select_best_periodic(run_dir):
    records = load_periodic_records(run_dir / "metrics.json")
    if not records:
        return None
    best = max(records, key=lambda item: (item["bbox/AP"], item["iteration"]))
    checkpoint = run_dir / f"model_{best['iteration']:07d}.pth"
    return {
        "iteration": best["iteration"],
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "periodic_metrics": best["metrics"],
        "periodic_per_class_ap": best["per_class_ap"],
    }


def build_eval_command(config_file, checkpoint, output_dir, overrides):
    return [
        sys.executable,
        "train_net.py",
        "--num-gpus",
        "1",
        "--eval-only",
        "--config-file",
        config_file,
        "MODEL.WEIGHTS",
        str(checkpoint),
        "OUTPUT_DIR",
        str(output_dir),
        "SEMISUPNET.Trainer",
        "ubteacher",
        "SEMISUPNET.CDPL_ENABLED",
        "False",
        "SOLVER.IMS_PER_BATCH",
        str(overrides.get("ims_per_batch", 8)),
        "SOLVER.IMG_PER_BATCH_LABEL",
        str(overrides.get("img_per_batch_label", 4)),
        "SOLVER.IMG_PER_BATCH_UNLABEL",
        str(overrides.get("img_per_batch_unlabel", 4)),
        "DATALOADER.NUM_WORKERS",
        str(overrides.get("num_workers", 2)),
    ]


def env_for_run(overrides):
    env = os.environ.copy()
    for json_key, env_key in [
        ("oct_ss_train_json", "OCT_SS_TRAIN_JSON"),
        ("oct_ss_unlabel_json", "OCT_SS_UNLABEL_JSON"),
        ("oct_ss_test_json", "OCT_SS_TEST_JSON"),
        ("oct_ss_train_image_root", "OCT_SS_TRAIN_IMAGE_ROOT"),
        ("oct_ss_unlabel_image_root", "OCT_SS_UNLABEL_IMAGE_ROOT"),
        ("oct_ss_image_root", "OCT_SS_IMAGE_ROOT"),
    ]:
        value = overrides.get(json_key)
        if value:
            env[env_key] = str(value)
    env.setdefault("OMP_NUM_THREADS", "1")
    env["CUDA_VISIBLE_DEVICES"] = str(overrides.get("eval_cuda_visible_devices", overrides.get("cuda_visible_devices", "0"))).split(",")[0]
    return env


def run_eval(repo_dir, config_file, checkpoint, output_dir, overrides, force=False):
    output_dir = Path(output_dir)
    predictions = output_dir / "inference" / "coco_instances_results.json"
    if predictions.exists() and not force:
        return {
            "output_dir": str(output_dir),
            "predictions": str(predictions),
            "ran_eval": False,
            "exit_code": 0,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_eval_command(config_file, checkpoint, output_dir, overrides)
    log_path = output_dir / "eval.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=env_for_run(overrides),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    (output_dir / "exit_code.txt").write_text(f"{proc.returncode}\n", encoding="utf-8")
    if proc.returncode == 0 and predictions.exists():
        (output_dir / "eval.ok").write_text("ok\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "predictions": str(predictions),
        "ran_eval": True,
        "exit_code": proc.returncode,
    }


def run_diagnostics(repo_dir, config_file, checkpoint, output_path, overrides, limit, force=False):
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return {"output": str(output_path), "ran_diagnostics": False, "exit_code": 0}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/experiments/diagnose_step2_pseudo_labels.py",
        "--config-file",
        config_file,
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_path),
        "--limit",
        str(limit),
        "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
        "0.05",
        "SEMISUPNET.BBOX_THRESHOLD",
        "0.5",
        "SEMISUPNET.CDPL_ALPHA_TAIL",
        str(overrides.get("cdpl_alpha_tail", 0.15)),
        "SEMISUPNET.CDPL_MIN_CLS_THRESHOLD",
        str(overrides.get("cdpl_min_cls_threshold", 0.35)),
        "SEMISUPNET.CDPL_MAX_THRESHOLD_OFFSET",
        str(overrides.get("cdpl_max_threshold_offset", -1.0)),
    ]
    log_path = output_path.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=env_for_run(overrides),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {"output": str(output_path), "ran_diagnostics": True, "exit_code": proc.returncode}


def collect_run(repo_dir, run_root, run_name, annotations, config_file, args):
    run_dir = Path(run_root) / run_name
    overrides = load_json(run_dir / "config_overrides.json", default={}) or {}
    metadata = load_metadata(run_dir / "run_metadata.txt")
    for key, value in metadata.items():
        overrides.setdefault(key, value)
    overrides["eval_cuda_visible_devices"] = args.eval_cuda_visible_devices

    result = {
        "run": run_name,
        "run_dir": str(run_dir),
        "complete": (run_dir / "complete.ok").exists(),
        "exit_code": (run_dir / "exit_code.txt").read_text(encoding="utf-8").strip()
        if (run_dir / "exit_code.txt").exists()
        else None,
        "git_commit": (run_dir / "git_commit.txt").read_text(encoding="utf-8").strip()
        if (run_dir / "git_commit.txt").exists()
        else None,
        "config_overrides": overrides,
        "artifacts_exist": {
            "config_input": (run_dir / "config_input.yaml").exists(),
            "config_overrides": (run_dir / "config_overrides.json").exists(),
            "launch_command": (run_dir / "launch_command.sh").exists(),
            "train_log": (run_dir / "train.log").exists(),
            "model_final": (run_dir / "model_final.pth").exists(),
            "periodic_metrics": (run_dir / "metrics.json").exists(),
            "pseudo_label_diagnostics_final": (run_dir / "pseudo_label_diagnostics_final.json").exists(),
        },
    }

    best = select_best_periodic(run_dir)
    result["best_periodic"] = best

    checkpoints = {
        "final": {
            "checkpoint": str(run_dir / "model_final.pth"),
            "eval_dir": str(run_dir / "eval_final"),
        }
    }
    if best:
        checkpoints["best_test"] = {
            "checkpoint": best["checkpoint"],
            "iteration": best["iteration"],
            "eval_dir": str(Path(run_root) / "best_test_eval" / f"{run_name}_iter{best['iteration']}"),
        }

    evals = {}
    for label, info in checkpoints.items():
        checkpoint = Path(info["checkpoint"])
        eval_result = {
            "checkpoint": str(checkpoint),
            "checkpoint_exists": checkpoint.exists(),
            "eval_dir": info["eval_dir"],
            "iteration": info.get("iteration"),
        }
        if checkpoint.exists() and (args.eval_only or args.force_eval):
            eval_result.update(
                run_eval(
                    repo_dir,
                    config_file,
                    checkpoint,
                    info["eval_dir"],
                    overrides,
                    force=args.force_eval,
                )
            )
        else:
            predictions = Path(info["eval_dir"]) / "inference" / "coco_instances_results.json"
            eval_result.update(
                {
                    "predictions": str(predictions),
                    "ran_eval": False,
                    "exit_code": None,
                }
            )

        metrics = compute_coco_metrics(annotations, eval_result["predictions"])
        if metrics:
            eval_result.update(metrics)
        evals[label] = eval_result

    result["evals"] = evals

    if args.diagnose:
        final_checkpoint = Path(checkpoints["final"]["checkpoint"])
        if final_checkpoint.exists():
            result["final_diagnostics"] = run_diagnostics(
                repo_dir,
                config_file,
                final_checkpoint,
                run_dir / "pseudo_label_diagnostics_final.json",
                overrides,
                args.diagnose_limit,
                force=args.force_diagnostics,
            )
        if best and Path(best["checkpoint"]).exists():
            result["best_test_diagnostics"] = run_diagnostics(
                repo_dir,
                config_file,
                Path(best["checkpoint"]),
                run_dir / "pseudo_label_diagnostics_best_test.json",
                overrides,
                args.diagnose_limit,
                force=args.force_diagnostics,
            )

    return result


def collect_baseline_reference(run_root, annotations):
    ref_root = Path(run_root) / "baseline_reference"
    metadata = load_json(ref_root / "reference_metadata.json", default={}) or {}
    if not ref_root.exists():
        return None

    checkpoints = {
        "final": {
            "checkpoint": metadata.get("final_checkpoint"),
            "eval_dir": str(ref_root / "ubt_baseline_seed0_24k_final_iter23999"),
            "iteration": 23999,
        },
        "best_test": {
            "checkpoint": metadata.get("best_test_checkpoint"),
            "eval_dir": str(ref_root / "ubt_baseline_seed0_24k_best_iter17999"),
            "iteration": 17999,
        },
    }
    evals = {}
    for label, info in checkpoints.items():
        predictions = Path(info["eval_dir"]) / "inference" / "coco_instances_results.json"
        eval_result = {
            "checkpoint": info.get("checkpoint"),
            "checkpoint_exists": Path(info["checkpoint"]).exists()
            if info.get("checkpoint")
            else False,
            "eval_dir": info["eval_dir"],
            "iteration": info["iteration"],
            "predictions": str(predictions),
            "ran_eval": False,
            "exit_code": 0 if predictions.exists() else None,
        }
        metrics = compute_coco_metrics(annotations, predictions)
        if metrics:
            eval_result.update(metrics)
        evals[label] = eval_result

    return {
        "run": "ubt_baseline_step1_reference_24k_seed0",
        "run_dir": str(ref_root),
        "complete": all(
            (Path(info["eval_dir"]) / "inference" / "coco_instances_results.json").exists()
            for info in checkpoints.values()
        ),
        "exit_code": None,
        "git_commit": metadata.get("source_git_commit"),
        "config_overrides": metadata,
        "artifacts_exist": {
            "baseline_reference_metadata": (ref_root / "reference_metadata.json").exists(),
            "final_predictions": (
                ref_root
                / "ubt_baseline_seed0_24k_final_iter23999"
                / "inference"
                / "coco_instances_results.json"
            ).exists(),
            "best_test_predictions": (
                ref_root
                / "ubt_baseline_seed0_24k_best_iter17999"
                / "inference"
                / "coco_instances_results.json"
            ).exists(),
        },
        "best_periodic": {
            "iteration": 17999,
            "checkpoint": metadata.get("best_test_checkpoint"),
            "checkpoint_exists": Path(metadata.get("best_test_checkpoint", "")).exists()
            if metadata.get("best_test_checkpoint")
            else False,
            "selection": metadata.get("best_test_selection"),
        },
        "evals": evals,
    }


def format_value(value):
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def metrics_row(run, label, payload):
    metrics = payload.get("metrics") or {}
    return [
        run,
        label,
        str(payload.get("iteration") or "final"),
        Path(payload.get("checkpoint", "")).name,
        format_value(metrics.get("mAP")),
        format_value(metrics.get("AP30")),
        format_value(metrics.get("AP50")),
        format_value(metrics.get("AP75")),
        format_value(metrics.get("AP_small")),
    ]


def write_markdown(doc_path, run_root, results):
    doc_path = Path(doc_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step 2: CDPL 诊断与弱校准实验",
        "",
        f"运行根目录：`{run_root}`",
        "",
        "## 当前状态",
        "",
        "- 本页由 `scripts/experiments/summarize_step2_results.py` 根据服务器 artifacts 生成或更新。",
        "- 选择 best checkpoint 的口径是 periodic test `bbox/AP` 最大值，不使用 val。",
        "- `best_test` checkpoint 会单独 eval-only，并从预测 JSON 补算 AP30、AP50、AP75、AP_small 和每类 AP。",
        "",
        "## 主指标",
        "",
        "| Run | Checkpoint | Iter | 文件 | mAP | AP30 | AP50 | AP75 | AP_small |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        for label in ["final", "best_test"]:
            payload = result.get("evals", {}).get(label)
            if not payload:
                continue
            lines.append("| " + " | ".join(metrics_row(result["run"], label, payload)) + " |")

    lines.extend(["", "## Artifact 检查", ""])
    for result in results:
        lines.append(f"### {result['run']}")
        lines.append("")
        lines.append(f"- 完成标记：{'yes' if result.get('complete') else 'no'}")
        lines.append(f"- 退出码：`{result.get('exit_code')}`")
        lines.append(f"- 提交：`{(result.get('git_commit') or 'NA')[:12]}`")
        for name, exists in result.get("artifacts_exist", {}).items():
            lines.append(f"- {name}: {'yes' if exists else 'no'}")
        best = result.get("best_periodic")
        if best:
            lines.append(
                f"- periodic test best：iter `{best['iteration']}`，checkpoint `{Path(best['checkpoint']).name}`，存在：{'yes' if best['checkpoint_exists'] else 'no'}"
            )
        else:
            lines.append("- periodic test best：NA")
        lines.append("")

    lines.extend(["## 每类 AP", ""])
    class_names = sorted(
        {
            class_name
            for result in results
            for payload in result.get("evals", {}).values()
            for class_name in (payload.get("per_class_ap") or {})
        }
    )
    if class_names:
        headers = []
        payloads = []
        for result in results:
            for label in ["final", "best_test"]:
                payload = result.get("evals", {}).get(label)
                if payload and payload.get("per_class_ap"):
                    headers.append(f"{result['run']} {label}")
                    payloads.append(payload)
        lines.append("| Class | " + " | ".join(headers) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in headers) + " |")
        for class_name in class_names:
            row = [class_name]
            for payload in payloads:
                row.append(format_value((payload.get("per_class_ap") or {}).get(class_name)))
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("当前还没有可用的每类 AP。")

    lines.extend(
        [
            "",
            "## 初步结论",
            "",
            "等待 Step2 诊断和至少两组改进实验完成后，在这里用中文总结：",
            "",
            "- pseudo-label 诊断是否解释了 Step 1 中 RS、F-PED、MH 等类别回退。",
            "- late/weak class calibration 是否缓解 mAP/AP30/AP75 回退。",
            "- threshold floor/offset clamp 是否减少高风险类别的伪标签扰动。",
            "- 是否值得继续实现 localization-aware Full CDPL。",
            "",
        ]
    )
    doc_path.write_text("\n".join(lines), encoding="utf-8")


def discover_runs(run_root):
    run_root = Path(run_root)
    if not run_root.exists():
        return []
    skipped = {"diagnostics", "best_test_eval"}
    return sorted(
        path.name
        for path in run_root.iterdir()
        if path.is_dir() and path.name not in skipped and (path / "run_metadata.txt").exists()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--config-file", default="configs/oct_ss_cdpl.yaml")
    parser.add_argument("--doc", default="docs/superpowers/experiments/step2-cdpl-diagnostics-and-weak-calibration.md")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--eval-only", action="store_true", help="Run missing final/best eval-only jobs.")
    parser.add_argument("--force-eval", action="store_true", help="Re-run eval-only even if predictions exist.")
    parser.add_argument("--diagnose", action="store_true", help="Run missing pseudo-label diagnostics for final/best checkpoints.")
    parser.add_argument("--force-diagnostics", action="store_true")
    parser.add_argument("--diagnose-limit", type=int, default=0)
    parser.add_argument("--eval-cuda-visible-devices", default="0")
    parser.add_argument(
        "--include-baseline-reference",
        dest="include_baseline_reference",
        action="store_true",
        default=True,
        help="Include Step1 UBT baseline reference evals saved under run_root/baseline_reference.",
    )
    parser.add_argument(
        "--no-include-baseline-reference",
        dest="include_baseline_reference",
        action="store_false",
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parents[2]
    run_root = Path(args.run_root)
    annotations = Path(args.annotations)
    run_names = args.runs if args.runs else discover_runs(run_root)
    results = [
        collect_run(repo_dir, run_root, run_name, annotations, args.config_file, args)
        for run_name in run_names
    ]
    if args.include_baseline_reference:
        baseline_reference = collect_baseline_reference(run_root, annotations)
        if baseline_reference:
            results.insert(0, baseline_reference)

    summary_json = Path(args.summary_json) if args.summary_json else run_root / "step2_eval_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(args.doc, run_root, results)
    print(f"Wrote {summary_json}")
    print(f"Wrote {args.doc}")


if __name__ == "__main__":
    main()
