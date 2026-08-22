from __future__ import annotations

import math
from typing import Any


def _iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_gt(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, list[dict[str, Any]]]:
    tp: list[dict[str, Any]] = []
    fp: list[dict[str, Any]] = []
    fn: list[dict[str, Any]] = []

    gt_matched: list[bool] = [False] * len(ground_truths)

    sorted_preds = sorted(predictions, key=lambda p: p.get("score", 0), reverse=True)

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truths):
            if gt_matched[gt_idx]:
                continue
            if pred["class_id"] != gt["class_id"]:
                continue
            iou_val = _iou(pred["bbox"], gt["bbox"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            tp.append({"pred": pred, "gt": ground_truths[best_gt_idx], "iou": best_iou})
            gt_matched[best_gt_idx] = True
        else:
            fp.append(pred)

    for gt_idx, matched in enumerate(gt_matched):
        if not matched:
            fn.append(ground_truths[gt_idx])

    return {"tp": tp, "fp": fp, "fn": fn}


def compute_precision_recall(
    tp: int, fp: int, fn: int
) -> tuple[float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def compute_confusion_matrix(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    class_ids: list[int],
    iou_threshold: float = 0.5,
) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    for cid in class_ids:
        preds_c = [p for p in predictions if p["class_id"] == cid]
        gts_c = [g for g in ground_truths if g["class_id"] == cid]
        matched = match_predictions_to_gt(preds_c, gts_c, iou_threshold)
        result[cid] = {
            "tp": len(matched["tp"]),
            "fp": len(matched["fp"]),
            "fn": len(matched["fn"]),
        }
    return result


def compute_per_class_metrics(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    class_ids: list[int],
    iou_threshold: float = 0.5,
) -> dict[int, dict[str, float]]:
    cm = compute_confusion_matrix(predictions, ground_truths, class_ids, iou_threshold)
    result: dict[int, dict[str, float]] = {}
    for cid in class_ids:
        entry = cm[cid]
        p, r = compute_precision_recall(entry["tp"], entry["fp"], entry["fn"])
        result[cid] = {"precision": p, "recall": r}
    return result


def compute_mAP(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    class_ids: list[int],
    iou_threshold: float = 0.5,
) -> float:
    if not class_ids:
        return 0.0

    ap_per_class: list[float] = []
    for cid in class_ids:
        preds_c = sorted(
            [p for p in predictions if p["class_id"] == cid],
            key=lambda p: p.get("score", 0),
            reverse=True,
        )
        gts_c = [g for g in ground_truths if g["class_id"] == cid]
        n_gt = len(gts_c)
        if n_gt == 0:
            continue

        tp_cumsum = 0
        fp_cumsum = 0
        matched_gts: set[int] = set()
        precisions: list[float] = []
        recalls: list[float] = []

        for pred in preds_c:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gts_c):
                if gt_idx in matched_gts:
                    continue
                iou_val = _iou(pred["bbox"], gt["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0 and best_iou >= iou_threshold:
                tp_cumsum += 1
                matched_gts.add(best_gt_idx)
            else:
                fp_cumsum += 1

            prec = tp_cumsum / (tp_cumsum + fp_cumsum)
            rec = tp_cumsum / n_gt
            precisions.append(prec)
            recalls.append(rec)

        if not precisions:
            ap_per_class.append(0.0)
            continue

        mrec = [0.0] + recalls + [1.0]
        mpre = [0.0] + precisions + [0.0]

        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        ap = 0.0
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]

        ap_per_class.append(ap)

    return sum(ap_per_class) / len(ap_per_class) if ap_per_class else 0.0
