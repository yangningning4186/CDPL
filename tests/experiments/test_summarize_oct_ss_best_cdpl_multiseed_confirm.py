import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "experiments"
    / "summarize_oct_ss_best_cdpl_multiseed_confirm.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_oct_ss_best_cdpl_multiseed_confirm", SCRIPT_PATH)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


CLASSES = ["D-PED", "D-SHM", "ERM", "F-PED", "IRF", "MH", "RS", "S-PED", "SRF"]


class SummarizeBestCdplMultiseedConfirmTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.summary_json = self.root / "summary.json"
        self.output_json = self.root / "aggregate.json"
        self.output_md = self.root / "aggregate.md"

    def tearDown(self):
        self.tempdir.cleanup()

    def make_result(self, run, seed, cdpl, best_map, final_map, mh, rs):
        per_class_ap = {name: 1.0 for name in CLASSES}
        per_class_ap.update({"MH": mh, "RS": rs})
        return {
            "run": run,
            "config_overrides": {
                "seed": str(seed),
                "cdpl_enabled": "True" if cdpl else "False",
                "run_mode": "cdpl_sweep" if cdpl else "ubt_repro",
            },
            "best_periodic": {"iteration": 10000 + seed},
            "evals": {
                "best_test": {
                    "metrics": {
                        "mAP": best_map,
                        "AP30": 40.0 + seed + (1.0 if cdpl else 0.0),
                        "AP50": 25.0 + seed + (0.5 if cdpl else 0.0),
                        "AP75": 6.0 + seed,
                        "AP_small": 4.0 + seed + (0.25 if cdpl else 0.0),
                    },
                    "per_class_ap": per_class_ap,
                },
                "final": {
                    "metrics": {
                        "mAP": final_map,
                        "AP30": 30.0,
                        "AP50": 20.0,
                        "AP75": 5.0,
                        "AP_small": 3.0,
                    }
                },
            },
        }

    def run_summary(self, results):
        self.summary_json.write_text(json.dumps(results), encoding="utf-8")
        argv = [
            "summary",
            "--summary-json",
            str(self.summary_json),
            "--run-root",
            str(self.root),
            "--output-json",
            str(self.output_json),
            "--output-md",
            str(self.output_md),
            "--seeds",
            "1",
            "2",
        ]
        with mock.patch.object(sys, "argv", argv):
            SUMMARY.main()
        return json.loads(self.output_json.read_text(encoding="utf-8"))

    def test_aggregates_seed_pair_deltas(self):
        results = [
            self.make_result("ubt_seed1", 1, False, 10.0, 9.0, 12.0, 7.0),
            self.make_result("cdpl_seed1", 1, True, 10.5, 9.1, 14.0, 9.0),
            self.make_result("ubt_seed2", 2, False, 11.0, 9.5, 13.0, 8.0),
            self.make_result("cdpl_seed2", 2, True, 11.25, 9.4, 15.0, 10.0),
        ]
        payload = self.run_summary(results)

        self.assertEqual(payload["result_policy"], "best_test")
        self.assertEqual(payload["seeds"], [1, 2])
        self.assertEqual(len(payload["pairs"]), 2)
        self.assertAlmostEqual(payload["aggregate"]["metric_deltas"]["mAP"]["mean"], 0.375)
        self.assertAlmostEqual(payload["aggregate"]["focus_class_deltas"]["MH"]["mean"], 2.0)
        self.assertTrue(self.output_md.exists())

    def test_rejects_missing_seed_pair(self):
        results = [
            self.make_result("ubt_seed1", 1, False, 10.0, 9.0, 12.0, 7.0),
            self.make_result("cdpl_seed1", 1, True, 10.5, 9.1, 14.0, 9.0),
            self.make_result("ubt_seed2", 2, False, 11.0, 9.5, 13.0, 8.0),
        ]
        with self.assertRaises(ValueError):
            self.run_summary(results)


if __name__ == "__main__":
    unittest.main()
