import torch
from detectron2.structures import Boxes, Instances


def filter_roih_instances_by_class_thresholds(inst, class_thresholds, tau_base):
    image_shape = inst.image_size
    new_inst = Instances(image_shape)

    if len(inst) == 0:
        device = inst.pred_boxes.tensor.device
        new_inst.gt_boxes = Boxes(torch.zeros(0, 4, device=device))
        new_inst.gt_classes = torch.zeros(0, dtype=torch.int64, device=device)
        new_inst.scores = torch.zeros(0, device=device)
        return new_inst, {"num_teacher_boxes": 0, "num_cls_kept": 0, "num_cls_dropped": 0}

    thresholds = torch.tensor(
        [
            float(class_thresholds.get(int(cls), tau_base))
            for cls in inst.pred_classes.detach().cpu().tolist()
        ],
        device=inst.scores.device,
        dtype=inst.scores.dtype,
    )
    keep = inst.scores >= thresholds

    new_inst.gt_boxes = Boxes(inst.pred_boxes.tensor[keep, :])
    new_inst.gt_classes = inst.pred_classes[keep]
    new_inst.scores = inst.scores[keep]

    return new_inst, {
        "num_teacher_boxes": len(inst),
        "num_cls_kept": len(new_inst),
        "num_cls_dropped": int((~keep).sum().item()),
    }

