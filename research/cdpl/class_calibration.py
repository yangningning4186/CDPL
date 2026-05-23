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
