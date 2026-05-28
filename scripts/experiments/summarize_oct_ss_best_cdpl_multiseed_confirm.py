#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


METRIC_KEYS = ["mAP", "AP30", "AP50", "AP75", "AP_small"]
FOCUS_CLASSES = ["MH", "RS", "F-PED", "S-PED"]
BEST_CONFIG = {
    "CDPL_START_ITER": 10000,
    "CDPL_RAMP_ITERS": 4000,
    "CDPL_MIN_CLS_THRESHOLD": 0.45,
    "CDPL_ALPHA_TAIL": 0.05,
    "BBOX_THRESHOLD": 0.5,
    "BURN_UP_STEP": 5000,
    "CDPL_TAU_BASE": 0.5,
    "CDPL_MAX_THRESHOLD_OFFSET": -1.0,
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_of(result):
    overrides = result.get("config_overrides") or {}
    seed = overrides.get("seed") or overrides.get("SEED")
    if seed is None:
        name = result.get("run", "")
        marker = "_seed"
        if marker in name:
            seed = name.rsplit(marker, 1)[1].split("_", 1)[0]
    if seed is None:
        raise ValueError(f"Cannot determine seed for {result.get('run')}")
    return int(seed)


def is_cdpl(result):
    overrides = result.get("config_overrides") or {}
    enabled = str(overrides.get("cdpl_enabled") or overrides.get("CDPL_ENABLED") or "").lower()
    mode = str(overrides.get("run_mode") or overrides.get("RUN_MODE") or "").lower()
    name = result.get("run", "")
    return enabled == "true" or mode.startswith("cdpl") or name.startswith("cdpl")


def best_metrics(result):
    return (result.get("evals") or {}).get("best_test", {}).get("metrics") or {}


def final_metrics(result):
    return (result.get("evals") or {}).get("final", {}).get("metrics") or {}


def best_per_class(result):
    return (result.get("evals") or {}).get("best_test", {}).get("per_class_ap") or {}


def rounded(value):
    if value is None:
        return None
    return round(float(value), 6)


def subtract_dict(left, right, keys):
    return {
        key: rounded(left[key] - right[key])
        for key in keys
        if left.get(key) is not None and right.get(key) is not None
    }


def mean_std(values):
    values = [float(value) for value in values]
    if not values:
        return {"mean": None, "std": None}
    return {
        "mean": rounded(statistics.mean(values)),
        "std": rounded(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def aggregate_pairs(pairs):
    aggregate = {
        "metrics": {},
        "metric_deltas": {},
        "focus_class_deltas": {},
        "final_metric_deltas": {},
    }
    for metric in METRIC_KEYS:
        aggregate["metrics"][f"ubt_{metric}"] = mean_std(
            pair["ubt_best_metrics"][metric]
            for pair in pairs
            if pair["ubt_best_metrics"].get(metric) is not None
        )
        aggregate["metrics"][f"cdpl_{metric}"] = mean_std(
            pair["cdpl_best_metrics"][metric]
            for pair in pairs
            if pair["cdpl_best_metrics"].get(metric) is not None
        )
        aggregate["metric_deltas"][metric] = mean_std(
            pair["best_metric_delta"][metric]
            for pair in pairs
            if pair["best_metric_delta"].get(metric) is not None
        )
        aggregate["final_metric_deltas"][metric] = mean_std(
            pair["final_metric_delta"][metric]
            for pair in pairs
            if pair["final_metric_delta"].get(metric) is not None
        )
    for class_name in FOCUS_CLASSES:
        aggregate["focus_class_deltas"][class_name] = mean_std(
            pair["best_per_class_delta"][class_name]
            for pair in pairs
            if pair["best_per_class_delta"].get(class_name) is not None
        )
    return aggregate


def format_value(value):
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def write_markdown(path, payload):
    lines = [
        "# OCT-SS Best CDPL Multiseed Confirmation",
        "",
        f"运行根目录：`{payload['run_root']}`",
        "",
        "主结果口径：`best_test`。`model_final.pth` 仅用于稳定性审计。",
        "",
        "## Best-Test 逐 Seed",
        "",
        "| Seed | UBT mAP | CDPL mAP | Delta mAP | Delta AP30 | Delta AP50 | Delta AP75 | Delta APs | Delta MH | Delta RS | Delta F-PED | Delta S-PED |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in payload["pairs"]:
        delta = pair["best_metric_delta"]
        class_delta = pair["best_per_class_delta"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(pair["seed"]),
                    format_value(pair["ubt_best_metrics"].get("mAP")),
                    format_value(pair["cdpl_best_metrics"].get("mAP")),
                    format_value(delta.get("mAP")),
                    format_value(delta.get("AP30")),
                    format_value(delta.get("AP50")),
                    format_value(delta.get("AP75")),
                    format_value(delta.get("AP_small")),
                    format_value(class_delta.get("MH")),
                    format_value(class_delta.get("RS")),
                    format_value(class_delta.get("F-PED")),
                    format_value(class_delta.get("S-PED")),
                ]
            )
            + " |"
        )

    aggregate = payload["aggregate"]
    lines.extend(
        [
            "",
            "## Seed1-3 汇总",
            "",
            "| Metric | Mean delta | Std delta |",
            "| --- | ---: | ---: |",
        ]
    )
    for metric in METRIC_KEYS:
        item = aggregate["metric_deltas"][metric]
        lines.append(f"| {metric} | {format_value(item['mean'])} | {format_value(item['std'])} |")
    for class_name in FOCUS_CLASSES:
        item = aggregate["focus_class_deltas"][class_name]
        lines.append(f"| {class_name} AP | {format_value(item['mean'])} | {format_value(item['std'])} |")

    lines.extend(
        [
            "",
            "## Final 稳定性审计",
            "",
            "| Metric | Mean final delta | Std final delta |",
            "| --- | ---: | ---: |",
        ]
    )
    for metric in METRIC_KEYS:
        item = aggregate["final_metric_deltas"][metric]
        lines.append(f"| {metric} | {format_value(item['mean'])} | {format_value(item['std'])} |")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()

    results = load_json(args.summary_json)
    by_seed = {seed: {"ubt": None, "cdpl": None} for seed in args.seeds}
    for result in results:
        seed = seed_of(result)
        if seed not in by_seed:
            continue
        by_seed[seed]["cdpl" if is_cdpl(result) else "ubt"] = result

    pairs = []
    for seed in args.seeds:
        ubt = by_seed[seed]["ubt"]
        cdpl = by_seed[seed]["cdpl"]
        if ubt is None or cdpl is None:
            missing = [name for name, item in by_seed[seed].items() if item is None]
            raise ValueError(f"Missing seed {seed} entries: {missing}")
        ubt_best = best_metrics(ubt)
        cdpl_best = best_metrics(cdpl)
        ubt_final = final_metrics(ubt)
        cdpl_final = final_metrics(cdpl)
        ubt_class = best_per_class(ubt)
        cdpl_class = best_per_class(cdpl)
        pairs.append(
            {
                "seed": seed,
                "ubt_run": ubt["run"],
                "cdpl_run": cdpl["run"],
                "ubt_best_iter": (ubt.get("best_periodic") or {}).get("iteration"),
                "cdpl_best_iter": (cdpl.get("best_periodic") or {}).get("iteration"),
                "ubt_best_metrics": {key: rounded(ubt_best.get(key)) for key in METRIC_KEYS},
                "cdpl_best_metrics": {key: rounded(cdpl_best.get(key)) for key in METRIC_KEYS},
                "best_metric_delta": subtract_dict(cdpl_best, ubt_best, METRIC_KEYS),
                "final_metric_delta": subtract_dict(cdpl_final, ubt_final, METRIC_KEYS),
                "best_per_class_delta": subtract_dict(cdpl_class, ubt_class, sorted(set(ubt_class) | set(cdpl_class))),
            }
        )

    payload = {
        "result_policy": "best_test",
        "run_root": args.run_root,
        "seeds": args.seeds,
        "best_config": BEST_CONFIG,
        "pairs": pairs,
        "aggregate": aggregate_pairs(pairs),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
