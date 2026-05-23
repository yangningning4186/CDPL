def partition_pseudo_labels(pseudo_labels, class_thresholds, tau_loc, tau_base=0.7):
    partitions = {"P_full": [], "P_cls": [], "P_drop": []}

    for label in pseudo_labels:
        category_id = label["category_id"]
        cls_score = float(label["score"])
        loc_score = float(label.get("loc_score", 0.0))
        tau_cls = float(class_thresholds.get(category_id, tau_base))
        enriched = dict(label)

        if cls_score >= tau_cls and loc_score >= tau_loc:
            enriched["partition"] = "P_full"
            partitions["P_full"].append(enriched)
        elif cls_score >= tau_cls:
            enriched["partition"] = "P_cls"
            partitions["P_cls"].append(enriched)
        else:
            enriched["partition"] = "P_drop"
            partitions["P_drop"].append(enriched)

    return partitions
