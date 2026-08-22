from __future__ import annotations

from backend.app.evaluation.metrics import (
    compute_confusion_matrix,
    compute_per_class_metrics,
    compute_precision_recall,
    compute_mAP,
    match_predictions_to_gt,
)


class TestMatchPredictionsToGt:
    def test_empty_predictions_returns_empty(self) -> None:
        result = match_predictions_to_gt([], [], iou_threshold=0.5)
        assert result == {"tp": [], "fp": [], "fn": []}

    def test_no_overlap_all_fn(self) -> None:
        gts = [{"class_id": 0, "bbox": [0, 0, 10, 10]}]
        preds = [{"class_id": 0, "bbox": [50, 50, 60, 60], "score": 0.9}]
        result = match_predictions_to_gt(preds, gts, iou_threshold=0.5)
        assert len(result["fn"]) == 1
        assert len(result["fp"]) == 1
        assert len(result["tp"]) == 0

    def test_perfect_match_tp(self) -> None:
        gts = [{"class_id": 0, "bbox": [0, 0, 10, 10]}]
        preds = [{"class_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9}]
        result = match_predictions_to_gt(preds, gts, iou_threshold=0.5)
        assert len(result["tp"]) == 1
        assert len(result["fp"]) == 0
        assert len(result["fn"]) == 0

    def test_low_iou_counts_as_fp_fn(self) -> None:
        gts = [{"class_id": 0, "bbox": [0, 0, 10, 10]}]
        preds = [{"class_id": 0, "bbox": [5, 5, 15, 15], "score": 0.9}]
        result = match_predictions_to_gt(preds, gts, iou_threshold=0.5)
        assert len(result["tp"]) == 0
        assert len(result["fp"]) == 1
        assert len(result["fn"]) == 1

    def test_class_mismatch_counts_as_fp_fn(self) -> None:
        gts = [{"class_id": 0, "bbox": [0, 0, 10, 10]}]
        preds = [{"class_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}]
        result = match_predictions_to_gt(preds, gts, iou_threshold=0.5)
        assert len(result["tp"]) == 0
        assert len(result["fp"]) == 1
        assert len(result["fn"]) == 1

    def test_multiple_gt_same_class_greedy_match(self) -> None:
        gts = [
            {"class_id": 0, "bbox": [0, 0, 10, 10]},
            {"class_id": 0, "bbox": [20, 20, 30, 30]},
        ]
        preds = [
            {"class_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"class_id": 0, "bbox": [20, 20, 30, 30], "score": 0.8},
        ]
        result = match_predictions_to_gt(preds, gts, iou_threshold=0.5)
        assert len(result["tp"]) == 2
        assert len(result["fp"]) == 0
        assert len(result["fn"]) == 0

    def test_more_predictions_than_gt_extra_are_fp(self) -> None:
        gts = [{"class_id": 0, "bbox": [0, 0, 10, 10]}]
        preds = [
            {"class_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"class_id": 0, "bbox": [5, 5, 15, 15], "score": 0.7},
        ]
        result = match_predictions_to_gt(preds, gts, iou_threshold=0.5)
        assert len(result["tp"]) == 1
        assert len(result["fp"]) == 1
        assert len(result["fn"]) == 0


class TestComputePrecisionRecall:
    def test_perfect_precision_recall(self) -> None:
        tp, fp, fn = 10, 0, 0
        p, r = compute_precision_recall(tp, fp, fn)
        assert p == 1.0
        assert r == 1.0

    def test_zero_tp_precision_is_zero(self) -> None:
        p, r = compute_precision_recall(0, 5, 3)
        assert p == 0.0
        assert r == 0.0

    def test_normal_case(self) -> None:
        tp, fp, fn = 8, 2, 3
        p, r = compute_precision_recall(tp, fp, fn)
        assert abs(p - 0.8) < 1e-9
        assert abs(r - 8 / 11) < 1e-9

    def test_all_false_positives(self) -> None:
        p, r = compute_precision_recall(0, 10, 5)
        assert p == 0.0
        assert r == 0.0


class TestComputeConfusionMatrix:
    def test_single_class(self) -> None:
        preds = [{"class_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9}]
        gts = [{"class_id": 0, "bbox": [0, 0, 10, 10]}]
        cm = compute_confusion_matrix(preds, gts, class_ids=[0], iou_threshold=0.5)
        assert cm[0]["tp"] == 1
        assert cm[0]["fp"] == 0
        assert cm[0]["fn"] == 0

    def test_multi_class(self) -> None:
        preds = [
            {"class_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"class_id": 1, "bbox": [20, 20, 30, 30], "score": 0.8},
            {"class_id": 0, "bbox": [50, 50, 60, 60], "score": 0.7},
        ]
        gts = [
            {"class_id": 0, "bbox": [0, 0, 10, 10]},
            {"class_id": 1, "bbox": [20, 20, 30, 30]},
        ]
        cm = compute_confusion_matrix(preds, gts, class_ids=[0, 1], iou_threshold=0.5)
        assert cm[0]["tp"] == 1
        assert cm[0]["fp"] == 1
        assert cm[0]["fn"] == 0
        assert cm[1]["tp"] == 1
        assert cm[1]["fp"] == 0
        assert cm[1]["fn"] == 0


class TestComputePerClassMetrics:
    def test_returns_precision_recall_per_class(self) -> None:
        preds = [
            {"class_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"class_id": 1, "bbox": [20, 20, 30, 30], "score": 0.8},
        ]
        gts = [
            {"class_id": 0, "bbox": [0, 0, 10, 10]},
            {"class_id": 1, "bbox": [50, 50, 60, 60]},
        ]
        result = compute_per_class_metrics(preds, gts, class_ids=[0, 1], iou_threshold=0.5)
        assert 0 in result
        assert 1 in result
        assert result[0]["precision"] == 1.0
        assert result[0]["recall"] == 1.0
        assert result[1]["precision"] == 0.0
        assert result[1]["recall"] == 0.0


class TestComputeMAP:
    def test_perfect_map(self) -> None:
        preds = [
            {"class_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"class_id": 1, "bbox": [20, 20, 30, 30], "score": 0.8},
        ]
        gts = [
            {"class_id": 0, "bbox": [0, 0, 10, 10]},
            {"class_id": 1, "bbox": [20, 20, 30, 30]},
        ]
        ap50 = compute_mAP(preds, gts, class_ids=[0, 1], iou_threshold=0.5)
        assert ap50 == 1.0

    def test_zero_map_when_no_matches(self) -> None:
        preds = [{"class_id": 0, "bbox": [50, 50, 60, 60], "score": 0.9}]
        gts = [{"class_id": 0, "bbox": [0, 0, 10, 10]}]
        ap50 = compute_mAP(preds, gts, class_ids=[0], iou_threshold=0.5)
        assert ap50 == 0.0

    def test_empty_preds_returns_zero(self) -> None:
        gts = [{"class_id": 0, "bbox": [0, 0, 10, 10]}]
        ap50 = compute_mAP([], gts, class_ids=[0], iou_threshold=0.5)
        assert ap50 == 0.0

    def test_map50_vs_map50_95(self) -> None:
        preds = [{"class_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9}]
        gts = [{"class_id": 0, "bbox": [0, 0, 10, 10]}]
        ap50 = compute_mAP(preds, gts, class_ids=[0], iou_threshold=0.5)
        ap95 = compute_mAP(preds, gts, class_ids=[0], iou_threshold=0.95)
        assert ap50 >= ap95
