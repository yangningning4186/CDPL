import logging
import time
from typing import Any, Dict

import torch

from oct_ss.cdpl.class_calibration import build_contiguous_class_thresholds
from oct_ss.cdpl.pseudo_labeling import filter_instances_by_class_thresholds
from oct_ss.engine.trainer_ubt import (
    UBTeacherTrainer,
    _cuda_batch_detectron,
    _ema_update,
    _extract_teacher_instances,
    _teacher_preds_to_pseudo,
    _unlabel_strong_with_pseudo,
    _unlabel_weak_only,
)

logger = logging.getLogger(__name__)


class CDPLTeacherTrainer(UBTeacherTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if bool(self.cfg.SEMISUPNET.CDPL_ENABLED):
            self.cdpl_class_thresholds = build_contiguous_class_thresholds(self.cfg)
        else:
            self.cdpl_class_thresholds = {}

    def run_step(self):
        start = time.perf_counter()
        labeled = self._next_batch("label")
        unlabeled = self._next_batch("unlabel")
        data_time = time.perf_counter() - start

        cfg = self.cfg
        device = self.student.device
        _cuda_batch_detectron(labeled, device)
        _cuda_batch_detectron(unlabeled, device)

        self.optimizer.zero_grad(set_to_none=True)
        burn_up = self.iter < cfg.SEMISUPNET.BURN_UP_STEP

        loss_dict: Dict[str, Any] = {}
        extra_metrics: Dict[str, float] = {}
        sup_weight = cfg.SEMISUPNET.SUP_LOSS_WEIGHT
        unsup_w = cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT

        self.student.train()
        losses_sup, _, _, _ = self.student(labeled, branch="supervised")
        for k, v in losses_sup.items():
            loss_dict[f"sup_{k}"] = v * sup_weight

        if not burn_up:
            weak = _unlabel_weak_only(unlabeled)
            self.teacher.eval()
            with torch.no_grad():
                teacher_out = self.teacher(
                    weak, branch="unsup_data_weak", val_mode=True
                )
                pred_instances = _extract_teacher_instances(teacher_out)

            if bool(cfg.SEMISUPNET.CDPL_ENABLED):
                boost_score_to = None
                if bool(cfg.SEMISUPNET.CDPL_BOOST_KEPT_SCORES):
                    boost_score_to = float(cfg.SEMISUPNET.BBOX_THRESHOLD) + 1e-6
                pred_instances, cdpl_stats = filter_instances_by_class_thresholds(
                    pred_instances,
                    self.cdpl_class_thresholds,
                    cfg.SEMISUPNET.CDPL_TAU_BASE,
                    boost_score_to=boost_score_to,
                )
                extra_metrics.update(
                    {
                        "pseudo/cdpl_teacher_boxes": float(
                            cdpl_stats["num_teacher_boxes"]
                        ),
                        "pseudo/cdpl_cls_kept": float(cdpl_stats["num_cls_kept"]),
                        "pseudo/cdpl_cls_dropped": float(
                            cdpl_stats["num_cls_dropped"]
                        ),
                    }
                )

            if bool(cfg.SEMISUPNET.DEBUG_PSEUDO_STATS):
                n_teacher = int(sum(len(x) for x in pred_instances))
                max_score = 0.0
                if n_teacher > 0:
                    max_score = max(
                        float(x.scores.max().item())
                        for x in pred_instances
                        if len(x) > 0
                    )
                th = float(cfg.SEMISUPNET.BBOX_THRESHOLD)
                n_after_score = int(
                    sum(
                        int((x.scores > th).sum().item())
                        for x in pred_instances
                        if len(x) > 0
                    )
                )
                extra_metrics["pseudo/teacher_max_score"] = max_score
                extra_metrics["pseudo/n_teacher_boxes"] = float(n_teacher)
                extra_metrics["pseudo/n_after_score_filter"] = float(n_after_score)
                print(
                    f"[PseudoDebug][iter {self.iter}] "
                    f"teacher_max_score={max_score:.4f}, "
                    f"n_teacher_boxes={n_teacher}, "
                    f"n_after_score_filter={n_after_score}"
                )

            if any(len(x) == 0 for x in pred_instances):
                logger.warning(
                    "[CDPL] Skipping unsup: empty teacher preds for at least one image."
                )
            else:
                pseudos = _teacher_preds_to_pseudo(pred_instances)
                pseudos = [p.to(device) for p in pseudos]
                strong = _unlabel_strong_with_pseudo(unlabeled, pseudos)

                images = self.student.preprocess_image(strong)
                feats = self.student.backbone(images.tensor)
                reg_unc = self.student.roi_heads.compute_uncertainty_with_aug(
                    feats, pseudos
                )
                if reg_unc.numel() > 0:
                    full_mask = reg_unc > float(cfg.SEMISUPNET.UNCERTAINTY_THRESHOLD)
                    extra_metrics["pseudo/cdpl_p_full"] = float(
                        full_mask.sum().item()
                    )
                    extra_metrics["pseudo/cdpl_p_cls"] = float(
                        reg_unc.numel() - full_mask.sum().item()
                    )

                if bool(cfg.SEMISUPNET.DEBUG_PSEUDO_STATS):
                    uth = float(cfg.SEMISUPNET.UNCERTAINTY_THRESHOLD)
                    n_after_unc = (
                        int((reg_unc > uth).sum().item()) if reg_unc.numel() > 0 else 0
                    )
                    extra_metrics["pseudo/n_after_unc_filter"] = float(n_after_unc)
                    print(
                        f"[PseudoDebug][iter {self.iter}] "
                        f"n_after_unc_filter={n_after_unc}, unc_th={uth:.4f}"
                    )

                losses_u, _, _, _ = self.student(
                    strong, branch="softteacher", reg_unc=reg_unc
                )
                for k, v in losses_u.items():
                    loss_dict[f"unsup_{k}"] = v * unsup_w

        total_loss = sum(loss_dict.values())
        total_loss.backward()
        self.after_backward()

        self._write_ubt_metrics(loss_dict, data_time, extra_metrics=extra_metrics)
        self.optimizer.step()

        t_every = max(int(cfg.SEMISUPNET.TEACHER_UPDATE_ITER), 1)
        if (self.iter + 1) % t_every == 0:
            _ema_update(self.teacher, self.student, float(cfg.SEMISUPNET.EMA_KEEP_RATE))
