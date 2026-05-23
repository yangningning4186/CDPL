def build_unlabeled_supervision(partitions):
    p_full = list(partitions.get("P_full", []))
    p_cls = list(partitions.get("P_cls", []))
    p_drop = list(partitions.get("P_drop", []))

    return {
        "classification": p_full + p_cls,
        "regression": p_full,
        "dropped": p_drop,
        "stats": {
            "num_p_full": len(p_full),
            "num_p_cls": len(p_cls),
            "num_p_drop": len(p_drop),
        },
    }
