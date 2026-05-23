import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = (
    REPO_ROOT / "server_patch" / "OCT_SS-CDPL" / "oct_ss" / "cdpl" / "trainer_cdpl.py"
)


@contextmanager
def stub_modules(mapping):
    old_modules = {name: sys.modules.get(name) for name in mapping}
    sys.modules.update(mapping)
    try:
        yield
    finally:
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_cdpl_trainer_module():
    class UBTeacherTrainer:
        pass

    class_calibration = types.ModuleType("oct_ss.cdpl.class_calibration")
    class_calibration.build_contiguous_class_thresholds = lambda cfg: {}

    pseudo_labeling = types.ModuleType("oct_ss.cdpl.pseudo_labeling")
    pseudo_labeling.filter_instances_by_class_thresholds = (
        lambda instances, thresholds, tau_base, boost_score_to=None: (instances, {})
    )

    trainer_ubt = types.ModuleType("oct_ss.engine.trainer_ubt")
    trainer_ubt.UBTeacherTrainer = UBTeacherTrainer
    trainer_ubt._cuda_batch_detectron = lambda batch, device: None
    trainer_ubt._ema_update = lambda teacher, student, keep: None
    trainer_ubt._extract_teacher_instances = lambda teacher_out: []
    trainer_ubt._teacher_preds_to_pseudo = lambda preds: []
    trainer_ubt._unlabel_strong_with_pseudo = lambda batch, pseudos: batch
    trainer_ubt._unlabel_weak_only = lambda batch: batch

    modules = {
        "torch": types.ModuleType("torch"),
        "oct_ss": types.ModuleType("oct_ss"),
        "oct_ss.cdpl": types.ModuleType("oct_ss.cdpl"),
        "oct_ss.cdpl.class_calibration": class_calibration,
        "oct_ss.cdpl.pseudo_labeling": pseudo_labeling,
        "oct_ss.engine": types.ModuleType("oct_ss.engine"),
        "oct_ss.engine.trainer_ubt": trainer_ubt,
    }
    with stub_modules(modules):
        spec = importlib.util.spec_from_file_location(
            "cdpl_trainer_compat_under_test", TRAINER_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class CDPLTrainerCompatTest(unittest.TestCase):
    def test_after_backward_compat_noops_when_base_hook_is_missing(self):
        module = load_cdpl_trainer_module()
        trainer = module.CDPLTeacherTrainer.__new__(module.CDPLTeacherTrainer)

        trainer._call_after_backward_if_supported()

    def test_after_backward_compat_calls_base_hook_when_available(self):
        module = load_cdpl_trainer_module()
        calls = []

        module.UBTeacherTrainer.after_backward = lambda self: calls.append("called")
        trainer = module.CDPLTeacherTrainer.__new__(module.CDPLTeacherTrainer)

        trainer._call_after_backward_if_supported()

        self.assertEqual(calls, ["called"])


if __name__ == "__main__":
    unittest.main()
