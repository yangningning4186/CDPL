# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torchvision.transforms as transforms
from ubteacher.data.transforms.augmentation_impl import (
    GaussianBlur,
)


def _build_cutout_augmentation(cfg):
    if not cfg.DATASETS.Cutout:
        return None

    parameters = (
        cfg.DATASETS.Cutout_p,
        cfg.DATASETS.Cutout_scale_l,
        cfg.DATASETS.Cutout_scale_r,
        cfg.DATASETS.Cutout_ratio_l,
        cfg.DATASETS.Cutout_ratio_r,
        cfg.DATASETS.Cutout_value,
    )
    parameter_lengths = {len(values) for values in parameters}
    if len(parameter_lengths) != 1:
        raise ValueError("All DATASETS.Cutout_* sequences must have the same length")

    specs = zip(*parameters)
    cutout = [transforms.ToTensor()]
    for probability, scale_l, scale_r, ratio_l, ratio_r, value in specs:
        cutout.append(
            transforms.RandomErasing(
                p=probability,
                scale=(scale_l, scale_r),
                ratio=(ratio_l, ratio_r),
                value=value,
            )
        )
    cutout.append(transforms.ToPILImage())
    return transforms.Compose(cutout)


def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))

        cutout_augmentation = _build_cutout_augmentation(cfg)
        if cutout_augmentation is not None:
            augmentation.append(cutout_augmentation)

        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)
