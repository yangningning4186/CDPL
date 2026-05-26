import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_UTILS_PATH = REPO_ROOT / "ubteacher" / "data" / "detection_utils.py"


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


def load_detection_utils():
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

    class RandomErasing:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class MarkerTransform:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    transforms = types.ModuleType("torchvision.transforms")
    transforms.Compose = Compose
    transforms.ToTensor = MarkerTransform
    transforms.ToPILImage = MarkerTransform
    transforms.RandomErasing = RandomErasing
    transforms.RandomApply = MarkerTransform
    transforms.ColorJitter = MarkerTransform
    transforms.RandomGrayscale = MarkerTransform

    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = transforms

    augmentation_impl = types.ModuleType("ubteacher.data.transforms.augmentation_impl")
    augmentation_impl.GaussianBlur = MarkerTransform

    with stub_modules(
        {
            "torchvision": torchvision,
            "torchvision.transforms": transforms,
            "ubteacher.data.transforms.augmentation_impl": augmentation_impl,
        }
    ):
        spec = importlib.util.spec_from_file_location(
            "cutout_augmentation_under_test", DETECTION_UTILS_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def cfg_with_cutout(enabled=True):
    datasets = types.SimpleNamespace(
        Cutout=enabled,
        Cutout_p=(0.7, 0.5, 0.3),
        Cutout_scale_l=(0.005, 0.002, 0.002),
        Cutout_scale_r=(0.01, 0.01, 0.01),
        Cutout_ratio_l=(0.3, 0.1, 0.05),
        Cutout_ratio_r=(3.3, 6, 8),
        Cutout_value=("random", "random", "random"),
    )
    return types.SimpleNamespace(DATASETS=datasets)


class CutoutAugmentationTest(unittest.TestCase):
    def test_builds_oct_ss_small_scale_cutout(self):
        module = load_detection_utils()

        augmentation = module._build_cutout_augmentation(cfg_with_cutout())
        erasing = augmentation.transforms[1:-1]

        self.assertEqual(
            [item.kwargs["p"] for item in erasing],
            [0.7, 0.5, 0.3],
        )
        self.assertEqual(
            [item.kwargs["scale"] for item in erasing],
            [(0.005, 0.01), (0.002, 0.01), (0.002, 0.01)],
        )
        self.assertEqual(
            [item.kwargs["ratio"] for item in erasing],
            [(0.3, 3.3), (0.1, 6), (0.05, 8)],
        )

    def test_disabled_cutout_returns_none(self):
        module = load_detection_utils()

        self.assertIsNone(module._build_cutout_augmentation(cfg_with_cutout(False)))

    def test_rejects_mismatched_cutout_parameter_lengths(self):
        module = load_detection_utils()
        cfg = cfg_with_cutout()
        cfg.DATASETS.Cutout_p = (0.7,)

        with self.assertRaisesRegex(ValueError, "same length"):
            module._build_cutout_augmentation(cfg)


if __name__ == "__main__":
    unittest.main()
