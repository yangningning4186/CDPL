import torch
from detectron2.structures import Boxes, Instances


def filter_instances_by_class_thresholds(
    instances,
    class_thresholds,
    tau_base,
    *,
    boost_score_to=None,
):
    filtered = []
    stats = {
        "num_teacher_boxes": 0,
        "num_cls_kept": 0,
        "num_cls_dropped": 0,
    }

    for inst in instances:
        stats["num_teacher_boxes"] += len(inst)
        if len(inst) == 0:
            filtered.append(inst)
            continue

        thresholds = torch.tensor(
            [
                float(class_thresholds.get(int(cls), tau_base))
                for cls in inst.pred_classes.detach().cpu().tolist()
            ],
            device=inst.scores.device,
            dtype=inst.scores.dtype,
        )
        keep = inst.scores >= thresholds
        kept = inst[keep]

        if boost_score_to is not None and len(kept) > 0:
            kept.raw_scores = kept.scores.clone()
            kept.scores = torch.clamp(kept.scores, min=float(boost_score_to))

        stats["num_cls_kept"] += len(kept)
        stats["num_cls_dropped"] += int((~keep).sum().item())
        filtered.append(kept)

    return filtered, stats


def empty_pseudo_like(image_size, device):
    pseudo = Instances(image_size)
    pseudo.gt_boxes = Boxes(torch.zeros(0, 4, device=device))
    pseudo.gt_classes = torch.zeros(0, dtype=torch.int64, device=device)
    pseudo.scores = torch.zeros(0, device=device)
    return pseudo
