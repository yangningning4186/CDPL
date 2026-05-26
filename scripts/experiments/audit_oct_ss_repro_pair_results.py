#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


EXPECTED_CLASSES = {
    "D-PED",
    "D-SHM",
    "ERM",
    "F-PED",
    "IRF",
    "MH",
    "RS",
    "S-PED",
    "SRF",
}
METRIC_KEYS = {"mAP", "AP30", "AP50", "AP75", "AP_small"}
REQUIRED_ARTIFACTS = {
    "config_input",
    "launch_command",
    "train_log",
    "model_final",
    "periodic_metrics",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def add_check(checks, name, passed, detail):
    checks.append({"requirement": name, "passed": bool(passed), "detail": detail})


def validate_eval(checks, run_name, label, payload):
    metrics = payload.get("metrics") or {}
    per_class = payload.get("per_class_ap") or {}
    predictions = Path(payload.get("predictions", ""))
    missing_metrics = sorted(METRIC_KEYS - set(metrics))
    missing_values = sorted(key for key in METRIC_KEYS if metrics.get(key) is None)

    add_check(
        checks,
        f"{run_name} {label} predictions",
        predictions.exists(),
        str(predictions),
    )
    add_check(
        checks,
        f"{run_name} {label} standard metrics",
        not missing_metrics and not missing_values,
        f"missing_keys={missing_metrics}, missing_values={missing_values}",
    )
    add_check(
        checks,
        f"{run_name} {label} nine-class AP mapping",
        set(per_class) == EXPECTED_CLASSES and not any(key.isdecimal() for key in per_class),
        f"classes={sorted(per_class)}",
    )


def validate_diagnostic(checks, path, label, expect_inactive_policy=False):
    path = Path(path)
    add_check(checks, f"CDPL {label} diagnostics file", path.exists(), str(path))
    if not path.exists():
        return

    report = load_json(path)
    per_class = report.get("per_class") or {}
    totals = report.get("totals") or {}
    thresholds = report.get("cdpl_thresholds") or {}
    delta = totals.get("cdpl_minus_ubt")
    calculated_delta = (
        totals.get("cdpl_kept") - totals.get("ubt_kept")
        if totals.get("cdpl_kept") is not None and totals.get("ubt_kept") is not None
        else None
    )
    add_check(
        checks,
        f"CDPL {label} class-id mapping",
        set(per_class) == EXPECTED_CLASSES
        and "F-PED" in per_class
        and not any(key.isdecimal() for key in per_class),
        f"classes={sorted(per_class)}",
    )
    add_check(
        checks,
        f"CDPL {label} retained-box delta",
        delta is not None and delta == calculated_delta,
        f"cdpl_minus_ubt={delta}, calculated={calculated_delta}",
    )
    if expect_inactive_policy:
        values = [float(value) for value in thresholds.values()]
        add_check(
            checks,
            "CDPL best-test pre-burn actual-policy thresholds",
            bool(values) and all(abs(value - 0.5) < 1e-8 for value in values),
            f"thresholds={thresholds}",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ubt-run", required=True)
    parser.add_argument("--cdpl-run", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--burn-up-step", type=int, default=5000)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    results = {item["run"]: item for item in load_json(args.summary_json)}
    checks = []

    for run_name in (args.ubt_run, args.cdpl_run):
        result = results.get(run_name)
        add_check(checks, f"{run_name} summary entry", result is not None, run_name)
        if result is None:
            continue
        add_check(checks, f"{run_name} training complete", result.get("complete"), str(result.get("complete")))
        add_check(checks, f"{run_name} exit code", str(result.get("exit_code")) == "0", str(result.get("exit_code")))
        add_check(
            checks,
            f"{run_name} training commit",
            result.get("git_commit") == args.expected_commit,
            str(result.get("git_commit")),
        )
        artifacts = result.get("artifacts_exist") or {}
        missing_artifacts = sorted(key for key in REQUIRED_ARTIFACTS if not artifacts.get(key))
        add_check(
            checks,
            f"{run_name} required artifacts",
            not missing_artifacts,
            f"missing={missing_artifacts}",
        )
        best = result.get("best_periodic") or {}
        add_check(
            checks,
            f"{run_name} periodic best checkpoint",
            bool(best.get("checkpoint_exists")),
            f"iter={best.get('iteration')}, checkpoint={best.get('checkpoint')}",
        )
        for label in ("final", "best_test"):
            payload = (result.get("evals") or {}).get(label)
            add_check(checks, f"{run_name} {label} eval entry", payload is not None, label)
            if payload:
                validate_eval(checks, run_name, label, payload)

    cdpl = results.get(args.cdpl_run) or {}
    best_iteration = int((cdpl.get("best_periodic") or {}).get("iteration", -1))
    cdpl_dir = run_root / args.cdpl_run
    validate_diagnostic(checks, cdpl_dir / "pseudo_label_diagnostics_final.json", "final")
    validate_diagnostic(
        checks,
        cdpl_dir / "pseudo_label_diagnostics_best_test.json",
        "best-test",
        expect_inactive_policy=best_iteration >= 0 and best_iteration < args.burn_up_step,
    )

    failed = [check for check in checks if not check["passed"]]
    report = {
        "run_root": str(run_root),
        "summary_json": args.summary_json,
        "expected_commit": args.expected_commit,
        "ubt_run": args.ubt_run,
        "cdpl_run": args.cdpl_run,
        "cdpl_best_iteration": best_iteration,
        "checks": checks,
        "passed": not failed,
        "failures": failed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    if failed:
        print(f"Audit failed: {len(failed)} requirement(s) unsatisfied.", file=sys.stderr)
        return 1
    print("Audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
