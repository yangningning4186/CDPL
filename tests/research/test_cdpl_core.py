import unittest

from research.cdpl.class_calibration import (
    compute_class_thresholds,
    compute_tail_scores,
)
from research.cdpl.localization_reliability import box_iou_xyxy, mean_pairwise_iou
from research.cdpl.pseudo_label_partition import partition_pseudo_labels
from research.cdpl.training_adapter import build_unlabeled_supervision


class TestPseudoLabelPartition(unittest.TestCase):
    def test_routes_full_cls_and_drop(self):
        pseudo_labels = [
            {"category_id": 1, "score": 0.9, "loc_score": 0.8},
            {"category_id": 1, "score": 0.9, "loc_score": 0.2},
            {"category_id": 1, "score": 0.3, "loc_score": 0.9},
        ]

        result = partition_pseudo_labels(
            pseudo_labels, class_thresholds={1: 0.7}, tau_loc=0.6
        )

        self.assertEqual(len(result["P_full"]), 1)
        self.assertEqual(len(result["P_cls"]), 1)
        self.assertEqual(len(result["P_drop"]), 1)
        self.assertEqual(result["P_full"][0]["partition"], "P_full")
        self.assertEqual(result["P_cls"][0]["partition"], "P_cls")
        self.assertEqual(result["P_drop"][0]["partition"], "P_drop")

    def test_unknown_class_uses_base_threshold(self):
        result = partition_pseudo_labels(
            [{"category_id": 9, "score": 0.65, "loc_score": 0.8}],
            class_thresholds={},
            tau_loc=0.6,
            tau_base=0.7,
        )

        self.assertEqual(len(result["P_drop"]), 1)


class TestLocalizationReliability(unittest.TestCase):
    def test_box_iou_xyxy_identical_boxes(self):
        box = [0.0, 0.0, 10.0, 10.0]
        self.assertEqual(box_iou_xyxy(box, box), 1.0)

    def test_mean_pairwise_iou_is_high_for_consistent_boxes(self):
        boxes = [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [0.0, 0.0, 9.0, 9.0],
        ]

        self.assertGreater(mean_pairwise_iou(boxes), 0.65)

    def test_mean_pairwise_iou_is_zero_for_single_box(self):
        self.assertEqual(mean_pairwise_iou([[0.0, 0.0, 10.0, 10.0]]), 0.0)


class TestClassCalibration(unittest.TestCase):
    def test_tail_scores_are_higher_for_rare_classes(self):
        scores = compute_tail_scores({1: 100, 2: 10})

        self.assertGreater(scores[2], scores[1])
        self.assertGreaterEqual(scores[1], 0.0)
        self.assertLessEqual(scores[1], 1.0)
        self.assertGreaterEqual(scores[2], 0.0)
        self.assertLessEqual(scores[2], 1.0)

    def test_thresholds_are_lower_for_tail_classes(self):
        thresholds = compute_class_thresholds(
            {1: 100, 2: 10}, tau_base=0.7, alpha_tail=0.15
        )

        self.assertLess(thresholds[2], thresholds[1])
        self.assertGreaterEqual(thresholds[2], 0.5)


class TestTrainingAdapter(unittest.TestCase):
    def test_routes_classification_and_regression_sets(self):
        partitions = {
            "P_full": [{"id": "full-1"}, {"id": "full-2"}],
            "P_cls": [{"id": "cls-1"}],
            "P_drop": [{"id": "drop-1"}],
        }

        result = build_unlabeled_supervision(partitions)

        self.assertEqual(
            [item["id"] for item in result["classification"]],
            ["full-1", "full-2", "cls-1"],
        )
        self.assertEqual(
            [item["id"] for item in result["regression"]], ["full-1", "full-2"]
        )
        self.assertEqual(result["dropped"][0]["id"], "drop-1")

    def test_reports_partition_counts(self):
        result = build_unlabeled_supervision(
            {
                "P_full": [{"id": "full-1"}],
                "P_cls": [{"id": "cls-1"}, {"id": "cls-2"}],
                "P_drop": [],
            }
        )

        self.assertEqual(
            result["stats"],
            {"num_p_full": 1, "num_p_cls": 2, "num_p_drop": 0},
        )


if __name__ == "__main__":
    unittest.main()
