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
    / "audit_oct_ss_repro_pair_results.py"
)
SPEC = importlib.util.spec_from_file_location("audit_oct_ss_repro_pair_results", SCRIPT_PATH)
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
COMMIT = "3555268ad84e2489eeef9ca6513a99d255d2ef32"


class AuditResultsTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.ubt = "ubt_repro_24000iter_1gpu_seed0"
        self.cdpl = "cdpl_calib_repro_24000iter_1gpu_seed0"
        self.summary = self.root / "summary.json"
        self.audit_output = self.root / "audit.json"
        self.results = [self.make_result(self.ubt), self.make_result(self.cdpl)]
        self.write_diagnostics()

    def tearDown(self):
        self.tempdir.cleanup()

    def make_result(self, run_name):
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
            "git_commit": COMMIT,
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

    def write_diagnostics(self):
        diagnostic = {
            "cdpl_thresholds": {name: 0.45 for name in CLASSES},
            "totals": {"ubt_kept": 100, "cdpl_kept": 130, "cdpl_minus_ubt": 30},
            "per_class": {name: {"class_id": idx} for idx, name in enumerate(sorted(CLASSES))},
        }
        run_dir = self.root / self.cdpl
        for label in ("final", "best_test"):
            (run_dir / f"pseudo_label_diagnostics_{label}.json").write_text(
                json.dumps(diagnostic),
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
            "--ubt-run",
            self.ubt,
            "--cdpl-run",
            self.cdpl,
            "--expected-commit",
            COMMIT,
        ]
        with mock.patch.object(sys, "argv", argv):
            return AUDIT.main()

    def test_passes_with_complete_nine_class_evidence(self):
        self.assertEqual(self.run_audit(), 0)
        report = json.loads(self.audit_output.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])

    def test_rejects_numeric_class_key_in_cdpl_diagnostics(self):
        diagnostic_path = self.root / self.cdpl / "pseudo_label_diagnostics_final.json"
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        diagnostic["per_class"]["3"] = diagnostic["per_class"].pop("F-PED")
        diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")

        self.assertEqual(self.run_audit(), 1)
        report = json.loads(self.audit_output.read_text(encoding="utf-8"))
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
