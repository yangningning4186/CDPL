#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


DEFAULT_RUN_ROOT = "/data4/ynz/cdpl_runs/step2_cdpl_diagnostics_and_weak_calib"


def load_json_lines(path):
    records = []
    path = Path(path)
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def read_text(path):
    path = Path(path)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def pid_running(pid):
    if not pid:
        return False
    proc = subprocess.run(
        ["ps", "-p", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def gpu_snapshot():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,pci.bus_id,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()}

    gpus = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "pci_bus_id": parts[2],
                "memory_used_mib": parts[3],
                "utilization_gpu_percent": parts[4],
            }
        )
    return {"gpus": gpus}


def latest_metrics(records):
    if not records:
        return {}

    latest = records[-1]
    bbox_records = [record for record in records if "bbox/AP" in record]
    best = None
    if bbox_records:
        best = max(bbox_records, key=lambda record: (float(record.get("bbox/AP", -1)), int(record.get("iteration", -1))))

    return {
        "iteration": latest.get("iteration"),
        "eta_seconds": latest.get("eta_seconds"),
        "time": latest.get("time"),
        "total_loss": latest.get("total_loss"),
        "loss_cls": latest.get("loss_cls"),
        "loss_box_reg": latest.get("loss_box_reg"),
        "loss_cls_pseudo": latest.get("loss_cls_pseudo"),
        "loss_box_reg_pseudo": latest.get("loss_box_reg_pseudo"),
        "latest_bbox_ap": latest.get("bbox/AP"),
        "best_periodic": {
            "iteration": best.get("iteration"),
            "bbox/AP": best.get("bbox/AP"),
            "bbox/AP50": best.get("bbox/AP50"),
            "bbox/AP75": best.get("bbox/AP75"),
            "bbox/APs": best.get("bbox/APs"),
        }
        if best
        else None,
    }


def discover_runs(run_root):
    skipped = {"diagnostics", "best_test_eval"}
    return sorted(
        path
        for path in Path(run_root).iterdir()
        if path.is_dir() and path.name not in skipped
    )


def collect_run(run_dir):
    pid = read_text(run_dir / "launcher.pid")
    records = load_json_lines(run_dir / "metrics.json")
    return {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "pid": pid,
        "pid_running": pid_running(pid),
        "complete": (run_dir / "complete.ok").exists(),
        "exit_code": read_text(run_dir / "exit_code.txt"),
        "git_commit": read_text(run_dir / "git_commit.txt"),
        "latest_metrics": latest_metrics(records),
        "artifacts": {
            "train_log": (run_dir / "train.log").exists(),
            "model_final": (run_dir / "model_final.pth").exists(),
            "metrics_json": (run_dir / "metrics.json").exists(),
            "pseudo_label_diagnostics_final": (run_dir / "pseudo_label_diagnostics_final.json").exists(),
        },
    }


def format_eta(seconds):
    if seconds is None:
        return "NA"
    seconds = int(float(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_text(report):
    print("Step2 run status")
    print(json.dumps(report.get("gpu_snapshot"), indent=2))
    for run in report["runs"]:
        metrics = run.get("latest_metrics") or {}
        best = metrics.get("best_periodic") or {}
        print(
            "{run}: pid={pid} running={running} complete={complete} exit={exit_code} "
            "iter={iteration} eta={eta} loss={loss} best_iter={best_iter} best_mAP={best_map}".format(
                run=run["run"],
                pid=run.get("pid"),
                running=run.get("pid_running"),
                complete=run.get("complete"),
                exit_code=run.get("exit_code"),
                iteration=metrics.get("iteration"),
                eta=format_eta(metrics.get("eta_seconds")),
                loss=metrics.get("total_loss"),
                best_iter=best.get("iteration"),
                best_map=best.get("bbox/AP"),
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--text", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    runs = [collect_run(run_dir) for run_dir in discover_runs(run_root)] if run_root.exists() else []
    report = {
        "run_root": str(run_root),
        "gpu_snapshot": gpu_snapshot(),
        "runs": runs,
    }

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.text or not args.json_output:
        print_text(report)


if __name__ == "__main__":
    main()
