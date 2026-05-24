#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

import ubteacher.data.datasets.builtin  # noqa: F401
from ubteacher import add_ubteacher_config
from ubteacher.cdpl.class_calibration import build_contiguous_class_thresholds
from ubteacher.engine.trainer import UBTeacherTrainer
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel


def percentile_summary(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
        }

    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p25": float(torch.quantile(tensor, 0.25).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p75": float(torch.quantile(tensor, 0.75).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "max": float(tensor.max().item()),
    }


def setup_cfg(args):
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.checkpoint
    cfg.DATASETS.TEST = (args.dataset,)
    cfg.SEMISUPNET.Trainer = "ubteacher"
    cfg.SEMISUPNET.CDPL_ENABLED = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.raw_score_threshold
    cfg.freeze()
    return cfg


def build_teacher(cfg):
    model = UBTeacherTrainer.build_model(cfg)
    model_teacher = UBTeacherTrainer.build_model(cfg)
    ensemble = EnsembleTSModel(model_teacher, model)
    DetectionCheckpointer(ensemble, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    teacher = ensemble.modelTeacher
    teacher.eval()
    return teacher


def empty_class_record():
    return {
        "teacher_boxes": 0,
        "ubt_kept": 0,
        "cdpl_kept": 0,
        "scores": [],
        "ubt_scores": [],
        "cdpl_scores": [],
        "areas_px": [],
        "ubt_areas_px": [],
        "cdpl_areas_px": [],
        "areas_frac": [],
        "ubt_areas_frac": [],
        "cdpl_areas_frac": [],
    }


def add_instances(class_records, instances, image_height, image_width, ubt_threshold, cdpl_thresholds):
    if len(instances) == 0:
        return

    classes = instances.pred_classes.detach().cpu().tolist()
    scores = instances.scores.detach().cpu().tolist()
    boxes = instances.pred_boxes.tensor.detach().cpu()
    image_area = max(float(image_height * image_width), 1.0)

    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    areas = (widths * heights).tolist()

    for cls, score, area in zip(classes, scores, areas):
        cls = int(cls)
        score = float(score)
        area = float(area)
        threshold = float(cdpl_thresholds.get(cls, ubt_threshold))
        record = class_records[cls]
        record["teacher_boxes"] += 1
        record["scores"].append(score)
        record["areas_px"].append(area)
        record["areas_frac"].append(area / image_area)

        if score >= ubt_threshold:
            record["ubt_kept"] += 1
            record["ubt_scores"].append(score)
            record["ubt_areas_px"].append(area)
            record["ubt_areas_frac"].append(area / image_area)
        if score >= threshold:
            record["cdpl_kept"] += 1
            record["cdpl_scores"].append(score)
            record["cdpl_areas_px"].append(area)
            record["cdpl_areas_frac"].append(area / image_area)


def finalize_records(class_records, class_names, cdpl_thresholds, ubt_threshold):
    output = {}
    for cls in sorted(class_records):
        record = class_records[cls]
        teacher_boxes = record["teacher_boxes"]
        cdpl_kept = record["cdpl_kept"]
        ubt_kept = record["ubt_kept"]
        output[class_names.get(cls, str(cls))] = {
            "class_id": cls,
            "thresholds": {
                "ubt": ubt_threshold,
                "cdpl": float(cdpl_thresholds.get(cls, ubt_threshold)),
            },
            "teacher_boxes": teacher_boxes,
            "ubt_kept": ubt_kept,
            "cdpl_kept": cdpl_kept,
            "cdpl_minus_ubt": cdpl_kept - ubt_kept,
            "ubt_keep_ratio": (ubt_kept / teacher_boxes if teacher_boxes else None),
            "cdpl_keep_ratio": (cdpl_kept / teacher_boxes if teacher_boxes else None),
            "cdpl_drop_ratio": (
                (teacher_boxes - cdpl_kept) / teacher_boxes if teacher_boxes else None
            ),
            "score_distribution": percentile_summary(record["scores"]),
            "ubt_kept_score_distribution": percentile_summary(record["ubt_scores"]),
            "cdpl_kept_score_distribution": percentile_summary(record["cdpl_scores"]),
            "area_px_distribution": percentile_summary(record["areas_px"]),
            "ubt_kept_area_px_distribution": percentile_summary(record["ubt_areas_px"]),
            "cdpl_kept_area_px_distribution": percentile_summary(record["cdpl_areas_px"]),
            "area_frac_distribution": percentile_summary(record["areas_frac"]),
            "ubt_kept_area_frac_distribution": percentile_summary(record["ubt_areas_frac"]),
            "cdpl_kept_area_frac_distribution": percentile_summary(record["cdpl_areas_frac"]),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="oct_ss_train_unlabel")
    parser.add_argument("--raw-score-threshold", type=float, default=0.05)
    parser.add_argument("--ubt-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--focus-classes", nargs="*", default=["RS", "F-PED", "MH"])
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    setup_logger()
    cfg = setup_cfg(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    teacher = build_teacher(cfg)
    loader = UBTeacherTrainer.build_test_loader(cfg, args.dataset)
    metadata = MetadataCatalog.get(args.dataset)
    class_names = {idx: name for idx, name in enumerate(getattr(metadata, "thing_classes", []))}
    cdpl_thresholds = build_contiguous_class_thresholds(cfg)

    class_records = defaultdict(empty_class_record)
    num_images = 0
    with torch.no_grad():
        for batch in loader:
            outputs = teacher(batch)
            for sample, output in zip(batch, outputs):
                image_shape = sample.get("height"), sample.get("width")
                instances = output["instances"].to("cpu")
                add_instances(
                    class_records,
                    instances,
                    int(image_shape[0]),
                    int(image_shape[1]),
                    args.ubt_threshold,
                    cdpl_thresholds,
                )
                num_images += 1
                if args.limit and num_images >= args.limit:
                    break
            if args.limit and num_images >= args.limit:
                break

    per_class = finalize_records(
        class_records,
        class_names,
        cdpl_thresholds,
        args.ubt_threshold,
    )
    focus = {name: per_class[name] for name in args.focus_classes if name in per_class}
    totals = {
        "teacher_boxes": sum(item["teacher_boxes"] for item in per_class.values()),
        "ubt_kept": sum(item["ubt_kept"] for item in per_class.values()),
        "cdpl_kept": sum(item["cdpl_kept"] for item in per_class.values()),
    }
    totals["cdpl_minus_ubt"] = totals["cdpl_kept"] - totals["ubt_kept"]

    report = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "num_images": num_images,
        "raw_score_threshold": args.raw_score_threshold,
        "ubt_threshold": args.ubt_threshold,
        "cdpl_thresholds": {
            class_names.get(int(cls), str(cls)): float(value)
            for cls, value in sorted(cdpl_thresholds.items())
        },
        "totals": totals,
        "focus_classes": focus,
        "per_class": per_class,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
