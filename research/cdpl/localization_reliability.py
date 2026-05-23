def box_iou_xyxy(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
        0.0, float(box_a[3]) - float(box_a[1])
    )
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
        0.0, float(box_b[3]) - float(box_b[1])
    )
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


def mean_pairwise_iou(boxes):
    if len(boxes) < 2:
        return 0.0

    scores = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            scores.append(box_iou_xyxy(boxes[i], boxes[j]))
    return sum(scores) / len(scores) if scores else 0.0
