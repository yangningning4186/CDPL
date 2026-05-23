#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=os.path.join(REPO_ROOT, "configs", "oct_ss_cdpl.yaml")
    )
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "output_cdpl"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Detectron2 overrides, e.g. SOLVER.MAX_ITER 10000",
    )
    args = parser.parse_args()

    import oct_ss.data.datasets.builtin as builtin_mod  # noqa: F401
    import oct_ss.modeling.meta_arch.rcnn  # noqa: F401
    import oct_ss.modeling.proposal_generator.rpn  # noqa: F401
    import oct_ss.modeling.roi_heads.roi_heads  # noqa: F401

    if not os.path.isfile(builtin_mod.Unlabeled_JSON):
        print(
            f"未找到无标签 COCO JSON：{builtin_mod.Unlabeled_JSON}\n"
            "请放置 train_unlabeled_v3_1.json 或设置环境变量 OCT_SS_UNLABEL_JSON。",
            file=sys.stderr,
        )
        sys.exit(1)

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.engine import default_setup
    from detectron2.modeling import build_model
    from detectron2.solver import build_lr_scheduler, build_optimizer
    from detectron2.utils.logger import setup_logger

    from oct_ss.cdpl.config import add_cdpl_config
    from oct_ss.cdpl.trainer_cdpl import CDPLTeacherTrainer
    from oct_ss.config import add_oct_ss_config
    from oct_ss.config_eval_iou import add_test_coco_bbox_iou_config
    from oct_ss.config_semisup_aug import add_semisup_strong_aug_config
    from oct_ss.data.ubt_build import build_ubt_train_loaders
    from oct_ss.engine.trainer_ubt import get_ubt_train_hooks

    setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = get_cfg()
    add_oct_ss_config(cfg)
    add_semisup_strong_aug_config(cfg)
    add_test_coco_bbox_iou_config(cfg)
    add_cdpl_config(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SOLVER.IMS_PER_BATCH = (
        cfg.SOLVER.IMG_PER_BATCH_LABEL + cfg.SOLVER.IMG_PER_BATCH_UNLABEL
    )
    cfg.set_new_allowed(False)
    cfg.freeze()
    default_setup(cfg, Namespace())

    student = build_model(cfg)
    teacher = build_model(cfg)
    teacher.eval()
    optimizer = build_optimizer(cfg, student)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(student, cfg.OUTPUT_DIR, optimizer=optimizer)
    if args.resume and checkpointer.has_checkpoint():
        ck = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
        start_iter = ck.get("iteration", -1) + 1
    else:
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        start_iter = 0
    teacher.load_state_dict(student.state_dict())

    loader_l, loader_u = build_ubt_train_loaders(cfg)
    trainer = CDPLTeacherTrainer(
        cfg, student, teacher, loader_l, loader_u, optimizer, scheduler
    )
    trainer.register_hooks(
        get_ubt_train_hooks(cfg, trainer, student, optimizer, scheduler)
    )
    trainer.train(start_iter, cfg.SOLVER.MAX_ITER)


if __name__ == "__main__":
    main()
