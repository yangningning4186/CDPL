import json
from collections import Counter

from detectron2.data import MetadataCatalog


def compute_tail_scores(class_counts):
    if not class_counts:
        return {}

    max_count = max(float(count) for count in class_counts.values())
    if max_count <= 0.0:
        return {category_id: 1.0 for category_id in class_counts}

    scores = {}
    for category_id, count in class_counts.items():
        frequency = float(count) / max_count
        scores[category_id] = max(0.0, min(1.0, 1.0 - frequency))
    return scores


def compute_class_thresholds(
    class_counts, tau_base=0.7, alpha_tail=0.15, min_threshold=0.5
):
    tail_scores = compute_tail_scores(class_counts)
    thresholds = {}
    for category_id, tail_score in tail_scores.items():
        threshold = float(tau_base) - float(alpha_tail) * float(tail_score)
        thresholds[category_id] = max(float(min_threshold), threshold)
    return thresholds


def load_coco_class_counts(json_file):
    with open(json_file, "r", encoding="utf-8") as handle:
        coco = json.load(handle)

    counts = Counter()
    for ann in coco.get("annotations", []):
        counts[int(ann["category_id"])] += 1
    return dict(counts)


def build_contiguous_class_thresholds(cfg):
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
