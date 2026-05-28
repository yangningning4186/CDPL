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
    / "audit_oct_ss_best_cdpl_multiseed_confirm_results.py"
)
SPEC = importlib.util.spec_from_file_location("audit_oct_ss_best_cdpl_multiseed_confirm_results", SCRIPT_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


CLASSES = {
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
COMMIT = "5421ca257282c2a5244d9cac59c7f80923aa4271"


class AuditBestCdplMultiseedConfirmTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.ubt = ["ubt_seed1", "ubt_seed2"]
        self.cdpl = ["cdpl_seed1", "cdpl_seed2"]
        self.summary = self.root / "summary.json"
        self.aggregate = self.root / "aggregate.json"
        self.audit_output = self.root / "audit.json"
        self.results = [
            self.make_result(self.ubt[0], 1, False),
            self.make_result(self.ubt[1], 2, False),
            self.make_result(self.cdpl[0], 1, True),
            self.make_result(self.cdpl[1], 2, True),
        ]
        for run in self.cdpl:
            self.write_diagnostics(run, ramp=0.25)
        self.write_aggregate()

    def tearDown(self):
        self.tempdir.cleanup()

    def make_result(self, run_name, seed, cdpl):
        run_dir = self.root / run_name
        run_dir.mkdir()
        checkpoint = run_dir / "model_0010999.pth"
        checkpoint.touch()
        for filename in ["config_input.yaml", "config_overrides.json", "launch_command.sh", "train.log", "model_final.pth", "metrics.json"]:
            (run_dir / filename).touch()
        evals = {}
        for label in ["final", "best_test"]:
            predictions = run_dir / f"{label}_predictions.json"
            predictions.touch()
            evals[label] = {
                "predictions": str(predictions),
                "metrics": {
                    "mAP": 10.0,
                    "AP30": 40.0,
                    "AP50": 25.0,
                    "AP75": 6.0,
                    "AP_small": 4.0,
                },
                "per_class_ap": {name: 1.0 for name in CLASSES},
            }
        overrides = {
            "seed": str(seed),
            "cdpl_enabled": "True" if cdpl else "False",
            "run_mode": "cdpl_sweep" if cdpl else "ubt_repro",
        }
        if cdpl:
            overrides.update(
                {
                    "cdpl_start_iter": "10000",
                    "cdpl_ramp_iters": "4000",
                    "cdpl_min_cls_threshold": "0.45",
                    "cdpl_alpha_tail": "0.05",
                }
            )
        return {
            "run": run_name,
            "complete": True,
            "exit_code": "0",
            "git_commit": COMMIT,
            "config_overrides": overrides,
            "artifacts_exist": {
                "config_input": True,
                "config_overrides": True,
                "launch_command": True,
                "train_log": True,
                "model_final": True,
                "periodic_metrics": True,
            },
            "best_periodic": {
                "iteration": 10999,
                "checkpoint": str(checkpoint),
                "checkpoint_exists": True,
            },
            "evals": evals,
        }

    def write_diagnostics(self, run_name, ramp):
        diagnostic = {
            "cdpl_ramp_factor": ramp,
            "cdpl_thresholds": {name: 0.48 for name in CLASSES},
            "totals": {"ubt_kept": 100, "cdpl_kept": 120, "cdpl_minus_ubt": 20},
            "per_class": {name: {"class_id": idx} for idx, name in enumerate(sorted(CLASSES))},
        }
        run_dir = self.root / run_name
        for label, factor in [("final", 1.0), ("best_test", ramp)]:
            payload = dict(diagnostic)
            payload["cdpl_ramp_factor"] = factor
            (run_dir / f"pseudo_label_diagnostics_{label}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

    def write_aggregate(self):
        self.aggregate.write_text(
            json.dumps(
                {
                    "result_policy": "best_test",
                    "seeds": [1, 2],
                    "pairs": [{"seed": 1}, {"seed": 2}],
                    "aggregate": {
                        "metric_deltas": {key: {"mean": 0.1, "std": 0.0} for key in AUDIT.METRIC_KEYS},
                        "focus_class_deltas": {
                            "MH": {"mean": 1.0, "std": 0.0},
                            "RS": {"mean": 1.0, "std": 0.0},
                            "F-PED": {"mean": 1.0, "std": 0.0},
                            "S-PED": {"mean": 1.0, "std": 0.0},
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    def run_audit(self):
        self.summary.write_text(json.dumps(self.results), encoding="utf-8")
        argv = [
            "audit",
            "--run-root",
            str(self.root),
            "--summary-json",
            str(self.summary),
            "--aggregate-json",
            str(self.aggregate),
            "--output",
            str(self.audit_output),
            "--ubt-runs",
            *self.ubt,
            "--cdpl-runs",
            *self.cdpl,
            "--seeds",
            "1",
            "2",
            "--expected-commit",
            COMMIT,
        ]
        with mock.patch.object(sys, "argv", argv):
            return AUDIT.main()

    def test_passes_complete_multiseed_evidence(self):
        self.assertEqual(self.run_audit(), 0)
        report = json.loads(self.audit_output.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])

    def test_rejects_wrong_cdpl_config(self):
        self.results[2]["config_overrides"]["cdpl_alpha_tail"] = "0.10"
        self.assertEqual(self.run_audit(), 1)
        report = json.loads(self.audit_output.read_text(encoding="utf-8"))
        self.assertFalse(report["passed"])

    def test_rejects_numeric_diagnostic_key(self):
        path = self.root / self.cdpl[0] / "pseudo_label_diagnostics_best_test.json"
        diagnostic = json.loads(path.read_text(encoding="utf-8"))
        diagnostic["per_class"]["3"] = diagnostic["per_class"].pop("F-PED")
        path.write_text(json.dumps(diagnostic), encoding="utf-8")

        self.assertEqual(self.run_audit(), 1)
        report = json.loads(self.audit_output.read_text(encoding="utf-8"))
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
