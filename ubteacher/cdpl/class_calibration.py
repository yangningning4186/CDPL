import json
from collections import Counter

from research.cdpl.class_calibration import compute_class_thresholds, compute_tail_scores


def load_coco_class_counts(json_file):
    with open(json_file, "r", encoding="utf-8") as handle:
        coco = json.load(handle)

    counts = Counter()
    for ann in coco.get("annotations", []):
        counts[int(ann["category_id"])] += 1
    return dict(counts)


def build_contiguous_class_thresholds(cfg):
    from detectron2.data import MetadataCatalog

    dataset_name = cfg.DATASETS.TRAIN_LABEL[0]
    metadata = MetadataCatalog.get(dataset_name)
    dataset_counts = load_coco_class_counts(metadata.json_file)
    dataset_thresholds = compute_class_thresholds(
        dataset_counts,
        tau_base=cfg.SEMISUPNET.CDPL_TAU_BASE,
        alpha_tail=cfg.SEMISUPNET.CDPL_ALPHA_TAIL,
        min_threshold=cfg.SEMISUPNET.CDPL_MIN_CLS_THRESHOLD,
    )

    mapping = getattr(metadata, "thing_dataset_id_to_contiguous_id", None) or {}
    if not mapping:
        return dataset_thresholds

    return {
        int(mapping[dataset_id]): threshold
        for dataset_id, threshold in dataset_thresholds.items()
        if dataset_id in mapping
    }

