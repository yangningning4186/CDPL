import logging

from ubteacher.cdpl.class_calibration import build_contiguous_class_thresholds
from ubteacher.cdpl.pseudo_labeling import (
    filter_roih_instances_by_class_thresholds,
)
from ubteacher.engine.trainer import UBTeacherTrainer


logger = logging.getLogger(__name__)


class CDPLUBTeacherTrainer(UBTeacherTrainer):
    """UBT trainer with class-adaptive pseudo-label thresholds."""

    def __init__(self, cfg):
        super().__init__(cfg)
        if bool(cfg.SEMISUPNET.CDPL_ENABLED):
            self.cdpl_class_thresholds = build_contiguous_class_thresholds(cfg)
            logger.info(
                "CDPL class thresholds: %s; start_iter=%s ramp_iters=%s max_offset=%s",
                self.cdpl_class_thresholds,
                cfg.SEMISUPNET.CDPL_START_ITER,
                cfg.SEMISUPNET.CDPL_RAMP_ITERS,
                cfg.SEMISUPNET.CDPL_MAX_THRESHOLD_OFFSET,
            )
        else:
            self.cdpl_class_thresholds = {}

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        if (
            bool(self.cfg.SEMISUPNET.CDPL_ENABLED)
            and proposal_type == "roih"
            and psedo_label_method == "thresholding"
            and self._cdpl_is_active()
        ):
            return self._process_cdpl_roih_pseudo_label(
                proposals_rpn_unsup_k, cur_threshold
            )
        return super().process_pseudo_label(
            proposals_rpn_unsup_k,
            cur_threshold,
            proposal_type,
            psedo_label_method,
        )

    def _cdpl_is_active(self):
        start_iter = int(self.cfg.SEMISUPNET.CDPL_START_ITER)
        if start_iter < 0:
            start_iter = int(self.cfg.SEMISUPNET.BURN_UP_STEP)
        return int(self.iter) >= start_iter

    def _current_cdpl_thresholds(self, cur_threshold):
        start_iter = int(self.cfg.SEMISUPNET.CDPL_START_ITER)
        if start_iter < 0:
            start_iter = int(self.cfg.SEMISUPNET.BURN_UP_STEP)
        ramp_iters = int(self.cfg.SEMISUPNET.CDPL_RAMP_ITERS)
        if ramp_iters <= 0:
            ramp = 1.0
        else:
            ramp = min(1.0, max(0.0, float(self.iter - start_iter + 1) / ramp_iters))

        if ramp >= 1.0:
            return self.cdpl_class_thresholds, ramp

        thresholds = {}
        for cls, threshold in self.cdpl_class_thresholds.items():
            thresholds[cls] = float(cur_threshold) - (
                float(cur_threshold) - float(threshold)
            ) * ramp
        return thresholds, ramp

    def _process_cdpl_roih_pseudo_label(self, proposals_roih_unsup_k, cur_threshold):
        list_instances = []
        num_proposal_output = 0.0
        class_thresholds, ramp = self._current_cdpl_thresholds(cur_threshold)
        stats = {
            "num_teacher_boxes": 0,
            "num_cls_kept": 0,
            "num_cls_dropped": 0,
        }

        for proposal_bbox_inst in proposals_roih_unsup_k:
            proposal_bbox_inst, inst_stats = filter_roih_instances_by_class_thresholds(
                proposal_bbox_inst,
                class_thresholds,
                cur_threshold,
            )
            for key, value in inst_stats.items():
                stats[key] += value
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)

        if bool(self.cfg.SEMISUPNET.DEBUG_PSEUDO_STATS):
            logger.info(
                "[CDPL][iter %s] teacher_boxes=%s cls_kept=%s cls_dropped=%s",
                self.iter,
                stats["num_teacher_boxes"],
                stats["num_cls_kept"],
                stats["num_cls_dropped"],
            )
            logger.info("[CDPL][iter %s] ramp=%s", self.iter, ramp)

        num_proposal_output = num_proposal_output / len(proposals_roih_unsup_k)
        return list_instances, num_proposal_output
