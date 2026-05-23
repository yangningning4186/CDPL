import os

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances


def _path_from_env(name, default):
    return os.path.abspath(os.getenv(name, default))


def register_oct_ss_datasets():
    dataset_root = os.getenv("OCT_SS_DATA_ROOT", os.getenv("DETECTRON2_DATASETS", "datasets"))
    ann_root = os.getenv("OCT_SS_ANN_ROOT", os.path.join(dataset_root, "annotations"))

    splits = {
        "oct_ss_train": (
            "OCT_SS_TRAIN_JSON",
            os.path.join(ann_root, "train_labeled.json"),
            "OCT_SS_TRAIN_IMAGE_ROOT",
            os.path.join(dataset_root, "images", "train"),
        ),
        "oct_ss_train_unlabel": (
            "OCT_SS_UNLABEL_JSON",
            os.path.join(ann_root, "train_unlabeled_v3_1.json"),
            "OCT_SS_UNLABEL_IMAGE_ROOT",
            os.path.join(dataset_root, "images", "unlabeled"),
        ),
        "oct_ss_test": (
            "OCT_SS_TEST_JSON",
            os.path.join(ann_root, "test_v3_1.json"),
            "OCT_SS_IMAGE_ROOT",
            os.path.join(dataset_root, "images", "test"),
        ),
    }

    for dataset_name, (json_env, json_default, image_env, image_default) in splits.items():
        if dataset_name in DatasetCatalog.list():
            continue
        register_coco_instances(
            dataset_name,
            {},
            _path_from_env(json_env, json_default),
            _path_from_env(image_env, image_default),
        )


register_oct_ss_datasets()
