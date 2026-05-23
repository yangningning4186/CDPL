import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WRITER_PERIOD_PATH = REPO_ROOT / "ubteacher" / "engine" / "writer_period.py"


def load_resolve_writer_period():
    spec = importlib.util.spec_from_file_location(
        "writer_period_under_test", WRITER_PERIOD_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve_writer_period, module.should_register_writers


class WriterPeriodTest(unittest.TestCase):
    def setUp(self):
        self.resolve_writer_period, self.should_register_writers = (
            load_resolve_writer_period()
        )

    def test_uses_default_for_long_runs(self):
        self.assertEqual(self.resolve_writer_period(100), 20)

    def test_uses_max_iter_for_short_runs(self):
        self.assertEqual(self.resolve_writer_period(2), 2)

    def test_never_returns_less_than_one(self):
        self.assertEqual(self.resolve_writer_period(0), 1)

    def test_skips_writers_for_short_smoke_runs(self):
        self.assertFalse(self.should_register_writers(2))

    def test_keeps_writers_for_long_runs(self):
        self.assertTrue(self.should_register_writers(100))


if __name__ == "__main__":
    unittest.main()
