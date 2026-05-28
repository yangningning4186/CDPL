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
    "config_overrides",
    "launch_command",
    "train_log",
    "model_final",
    "periodic_metrics",
}
BEST_CONFIG = {
    "cdpl_start_iter": "10000",
    "cdpl_ramp_iters": "4000",
    "cdpl_min_cls_threshold": "0.45",
    "cdpl_alpha_tail": "0.05",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def add_check(checks, name, passed, detail):
    checks.append({"requirement": name, "passed": bool(passed), "detail": detail})


def ramp_factor(iteration, start_iter, ramp_iters):
    if iteration is None or int(iteration) < int(start_iter):
        return 0.0
    if int(ramp_iters) <= 0:
        return 1.0
    value = float(int(iteration) - int(start_iter) + 1) / float(int(ramp_iters))
    return min(1.0, max(0.0, value))


def seed_of(result):
    overrides = result.get("config_overrides") or {}
    seed = overrides.get("seed") or overrides.get("SEED")
    if seed is None and "_seed" in result.get("run", ""):
        seed = result["run"].rsplit("_seed", 1)[1].split("_", 1)[0]
    return int(seed) if seed is not None else None


def validate_eval(checks, run_name, label, payload):
    metrics = payload.get("metrics") or {}
    per_class = payload.get("per_class_ap") or {}
    predictions = Path(payload.get("predictions", ""))
    missing_metrics = sorted(METRIC_KEYS - set(metrics))
    missing_values = sorted(key for key in METRIC_KEYS if metrics.get(key) is None)

    add_check(checks, f"{run_name} {label} predictions", predictions.exists(), str(predictions))
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


def validate_run(checks, result, run_name, expected_commit, expected_seed):
    add_check(checks, f"{run_name} summary entry", result is not None, run_name)
    if result is None:
        return
    add_check(checks, f"{run_name} seed", seed_of(result) == expected_seed, f"seed={seed_of(result)}")
    add_check(checks, f"{run_name} training complete", result.get("complete"), str(result.get("complete")))
    add_check(checks, f"{run_name} exit code", str(result.get("exit_code")) == "0", str(result.get("exit_code")))
    add_check(
        checks,
        f"{run_name} training commit",
        result.get("git_commit") == expected_commit,
        str(result.get("git_commit")),
    )
    artifacts = result.get("artifacts_exist") or {}
    missing_artifacts = sorted(key for key in REQUIRED_ARTIFACTS if not artifacts.get(key))
    add_check(checks, f"{run_name} required artifacts", not missing_artifacts, f"missing={missing_artifacts}")
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


def validate_best_cdpl_config(checks, result, run_name):
    overrides = result.get("config_overrides") or {}
    for key, expected in BEST_CONFIG.items():
        actual = str(overrides.get(key))
        add_check(checks, f"{run_name} best CDPL config {key}", actual == expected, f"actual={actual}")
    add_check(
        checks,
        f"{run_name} CDPL enabled",
        str(overrides.get("cdpl_enabled")).lower() == "true",
        f"actual={overrides.get('cdpl_enabled')}",
    )


def validate_diagnostic(checks, run_root, result, run_name, label, expected_ramp):
    path = Path(run_root) / run_name / f"pseudo_label_diagnostics_{label}.json"
    add_check(checks, f"{run_name} {label} diagnostics file", path.exists(), str(path))
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
        f"{run_name} {label} diagnostic class-id mapping",
        set(per_class) == EXPECTED_CLASSES
        and "F-PED" in per_class
        and not any(key.isdecimal() for key in per_class),
        f"classes={sorted(per_class)}",
    )
    add_check(
        checks,
        f"{run_name} {label} diagnostic threshold mapping",
        set(thresholds) == EXPECTED_CLASSES and not any(key.isdecimal() for key in thresholds),
        f"classes={sorted(thresholds)}",
    )
    add_check(
        checks,
        f"{run_name} {label} retained-box delta",
        delta is not None and delta == calculated_delta,
        f"cdpl_minus_ubt={delta}, calculated={calculated_delta}",
    )
    actual_ramp = float(report.get("cdpl_ramp_factor", -1.0))
    add_check(
        checks,
        f"{run_name} {label} actual-policy ramp",
        abs(actual_ramp - expected_ramp) < 1e-6,
        f"actual={actual_ramp}, expected={expected_ramp}",
    )


def validate_aggregate(checks, aggregate_json, expected_seeds):
    path = Path(aggregate_json)
    add_check(checks, "multiseed aggregate JSON", path.exists(), str(path))
    if not path.exists():
        return
    payload = load_json(path)
    add_check(
        checks,
        "aggregate result policy",
        payload.get("result_policy") == "best_test",
        str(payload.get("result_policy")),
    )
    add_check(checks, "aggregate seeds", payload.get("seeds") == expected_seeds, str(payload.get("seeds")))
    pairs = payload.get("pairs") or []
    add_check(checks, "aggregate pair count", len(pairs) == len(expected_seeds), f"pairs={len(pairs)}")
    for seed in expected_seeds:
        add_check(
            checks,
            f"aggregate seed {seed} pair",
            any(pair.get("seed") == seed for pair in pairs),
            f"seeds={[pair.get('seed') for pair in pairs]}",
        )
    aggregate = payload.get("aggregate") or {}
    metric_deltas = aggregate.get("metric_deltas") or {}
    focus_class_deltas = aggregate.get("focus_class_deltas") or {}
    add_check(
        checks,
        "aggregate metric delta coverage",
        set(metric_deltas) == METRIC_KEYS,
        f"metrics={sorted(metric_deltas)}",
    )
    add_check(
        checks,
        "aggregate focus class coverage",
        {"MH", "RS", "F-PED", "S-PED"}.issubset(set(focus_class_deltas)),
        f"classes={sorted(focus_class_deltas)}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--aggregate-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ubt-runs", nargs="+", required=True)
    parser.add_argument("--cdpl-runs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    results = {item["run"]: item for item in load_json(args.summary_json)}
    checks = []
    if len(args.ubt_runs) != len(args.seeds):
        add_check(checks, "UBT run count", False, f"ubt_runs={len(args.ubt_runs)}, seeds={len(args.seeds)}")
    if len(args.cdpl_runs) != len(args.seeds):
        add_check(checks, "CDPL run count", False, f"cdpl_runs={len(args.cdpl_runs)}, seeds={len(args.seeds)}")

    for seed, run_name in zip(args.seeds, args.ubt_runs):
        validate_run(checks, results.get(run_name), run_name, args.expected_commit, seed)

    for seed, run_name in zip(args.seeds, args.cdpl_runs):
        result = results.get(run_name)
        validate_run(checks, result, run_name, args.expected_commit, seed)
        if result is None:
            continue
        validate_best_cdpl_config(checks, result, run_name)
        best = result.get("best_periodic") or {}
        expected_best_ramp = ramp_factor(best.get("iteration"), 10000, 4000)
        validate_diagnostic(checks, args.run_root, result, run_name, "final", 1.0)
        validate_diagnostic(checks, args.run_root, result, run_name, "best_test", expected_best_ramp)

    validate_aggregate(checks, args.aggregate_json, args.seeds)

    failed = [check for check in checks if not check["passed"]]
    report = {
        "run_root": args.run_root,
        "summary_json": args.summary_json,
        "aggregate_json": args.aggregate_json,
        "ubt_runs": args.ubt_runs,
        "cdpl_runs": args.cdpl_runs,
        "seeds": args.seeds,
        "expected_commit": args.expected_commit,
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
