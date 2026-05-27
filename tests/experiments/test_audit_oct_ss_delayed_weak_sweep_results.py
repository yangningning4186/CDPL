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
    / "audit_oct_ss_delayed_weak_sweep_results.py"
)
SPEC = importlib.util.spec_from_file_location("audit_oct_ss_delayed_weak_sweep_results", SCRIPT_PATH)
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
BASELINE_COMMIT = "3555268ad84e2489eeef9ca6513a99d255d2ef32"
VARIANT_COMMIT = "405e18bf1d4b1402692a5d8125ce947ad5ef58fa"


class AuditDelayedWeakSweepTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.baseline = "ubt_repro_24000iter_1gpu_seed0"
        self.variant = "cdpl_dw_s10000_r4000_min045_a005_24000iter_1gpu_seed0"
        self.summary = self.root / "summary.json"
        self.audit_output = self.root / "audit.json"
        self.results = [
            self.make_result(self.baseline, BASELINE_COMMIT),
            self.make_result(
                self.variant,
                VARIANT_COMMIT,
                overrides={
                    "cdpl_start_iter": "10000",
                    "cdpl_ramp_iters": "4000",
                    "cdpl_alpha_tail": "0.05",
                    "cdpl_min_cls_threshold": "0.45",
                },
            ),
        ]
        self.write_diagnostics(ramp=0.5)

    def tearDown(self):
        self.tempdir.cleanup()

    def make_result(self, run_name, commit, overrides=None):
        run_dir = self.root / run_name
        run_dir.mkdir()
        checkpoint = run_dir / "model_0011999.pth"
        checkpoint.touch()
        evals = {}
        for label in ("final", "best_test"):
            predictions = run_dir / f"{label}_results.json"
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
        return {
            "run": run_name,
            "complete": True,
            "exit_code": "0",
            "git_commit": commit,
            "config_overrides": overrides or {},
            "artifacts_exist": {
                key: True
                for key in ("config_input", "launch_command", "train_log", "model_final", "periodic_metrics")
            },
            "best_periodic": {
                "iteration": 11999,
                "checkpoint": str(checkpoint),
                "checkpoint_exists": True,
            },
            "evals": evals,
        }

    def write_diagnostics(self, ramp):
        diagnostic = {
            "cdpl_ramp_factor": ramp,
            "cdpl_thresholds": {name: 0.475 for name in CLASSES},
            "totals": {"ubt_kept": 100, "cdpl_kept": 130, "cdpl_minus_ubt": 30},
            "per_class": {name: {"class_id": idx} for idx, name in enumerate(sorted(CLASSES))},
        }
        run_dir = self.root / self.variant
        for label, factor in (("final", 1.0), ("best_test", ramp)):
            payload = dict(diagnostic)
            payload["cdpl_ramp_factor"] = factor
            (run_dir / f"pseudo_label_diagnostics_{label}.json").write_text(
                json.dumps(payload),
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
            "--output",
            str(self.audit_output),
            "--baseline-run",
            self.baseline,
            "--variant-runs",
            self.variant,
            "--baseline-expected-commit",
            BASELINE_COMMIT,
            "--variant-expected-commit",
            VARIANT_COMMIT,
        ]
        with mock.patch.object(sys, "argv", argv):
            return AUDIT.main()

    def test_passes_complete_sweep_evidence(self):
        self.assertEqual(self.run_audit(), 0)
        report = json.loads(self.audit_output.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])

    def test_rejects_wrong_best_ramp_factor(self):
        path = self.root / self.variant / "pseudo_label_diagnostics_best_test.json"
        diagnostic = json.loads(path.read_text(encoding="utf-8"))
        diagnostic["cdpl_ramp_factor"] = 1.0
        path.write_text(json.dumps(diagnostic), encoding="utf-8")

        self.assertEqual(self.run_audit(), 1)
        report = json.loads(self.audit_output.read_text(encoding="utf-8"))
        self.assertFalse(report["passed"])

    def test_rejects_numeric_class_key(self):
        path = self.root / self.variant / "pseudo_label_diagnostics_final.json"
        diagnostic = json.loads(path.read_text(encoding="utf-8"))
        diagnostic["per_class"]["3"] = diagnostic["per_class"].pop("F-PED")
        path.write_text(json.dumps(diagnostic), encoding="utf-8")

        self.assertEqual(self.run_audit(), 1)
        report = json.loads(self.audit_output.read_text(encoding="utf-8"))
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
