from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from backend.app.evaluation.metrics import (
    compute_mAP,
    compute_per_class_metrics,
    compute_precision_recall,
    match_predictions_to_gt,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    model_path: str
    num_classes: int
    class_names: dict[int, str]


@dataclass
class EvalResult:
    success: bool
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    per_class: dict[int, dict[str, float]]
    num_images: int
    num_predictions: int
    num_ground_truths: int
    error: str | None = None


class YoloEvaluator:
    def __init__(
        self,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model: Any | None = None
        self._model_info: ModelInfo | None = None

    def load_model(self, model_path: str) -> ModelInfo:
        if self._model is not None:
            return self._model_info

        model = YOLO(model_path)
        class_names: dict[int, str] = {}
        if hasattr(model, "names") and model.names:
            class_names = dict(model.names)

        info = ModelInfo(
            model_path=model_path,
            num_classes=len(class_names),
            class_names=class_names,
        )
        self._model = model
        self._model_info = info
        logger.info(
            "Loaded model from %s (%d classes)", model_path, len(class_names)
        )
        return info

    def unload_model(self) -> bool:
        if self._model is not None:
            self._model = None
            self._model_info = None
            return True
        return False

    def _predict_single(self, model: Any, image: Any) -> list[dict[str, Any]]:
        results = model(image, verbose=False)
        predictions: list[dict[str, Any]] = []

        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, "boxes") and result.boxes is not None:
                xyxy = result.boxes.xyxy
                confs = result.boxes.conf
                clss = result.boxes.cls
                for i in range(len(xyxy)):
                    box = (
                        xyxy[i].tolist()
                        if hasattr(xyxy[i], "tolist")
                        else list(xyxy[i])
                    )
                    conf = float(confs[i]) if hasattr(confs, "__getitem__") else 0.0
                    cls_id = int(clss[i]) if hasattr(clss, "__getitem__") else 0

                    if conf >= self.confidence_threshold:
                        predictions.append(
                            {
                                "class_id": cls_id,
                                "bbox": box,
                                "score": conf,
                            }
                        )

        return predictions

    def predict_on_dataset(
        self, images: list[Any]
    ) -> list[list[dict[str, Any]]]:
        if self._model is None:
            raise ValueError("Model not loaded")

        all_predictions: list[list[dict[str, Any]]] = []
        for image in images:
            preds = self._predict_single(self._model, image)
            all_predictions.append(preds)
        return all_predictions

    def _compute_overall_metrics(
        self,
        all_predictions: list[list[dict[str, Any]]],
        all_ground_truths: list[list[dict[str, Any]]],
        class_ids: list[int],
    ) -> dict[str, Any]:
        flat_preds = [p for preds in all_predictions for p in preds]
        flat_gts = [gt for gts in all_ground_truths for gt in gts]

        if not flat_preds and not flat_gts:
            return {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}

        mAP50 = compute_mAP(flat_preds, flat_gts, class_ids, iou_threshold=0.5)
        mAP50_95 = compute_mAP(flat_preds, flat_gts, class_ids, iou_threshold=0.95)

        matched = match_predictions_to_gt(flat_preds, flat_gts, iou_threshold=0.5)
        total_tp = len(matched["tp"])
        total_fp = len(matched["fp"])
        total_fn = len(matched["fn"])

        precision, recall = compute_precision_recall(total_tp, total_fp, total_fn)

        return {
            "mAP50": mAP50,
            "mAP50_95": mAP50_95,
            "precision": precision,
            "recall": recall,
        }

    def evaluate(
        self,
        dataset_images: list[Any],
        dataset_ground_truths: list[list[dict[str, Any]]],
        class_ids: list[int],
    ) -> EvalResult:
        if self._model is None:
            raise ValueError("Model not loaded")

        try:
            all_predictions = self.predict_on_dataset(dataset_images)

            overall = self._compute_overall_metrics(
                all_predictions, dataset_ground_truths, class_ids
            )

            per_class = compute_per_class_metrics(
                [p for preds in all_predictions for p in preds],
                [gt for gts in dataset_ground_truths for gt in gts],
                class_ids,
                iou_threshold=0.5,
            )

            num_preds = sum(len(preds) for preds in all_predictions)
            num_gts = sum(len(gts) for gts in dataset_ground_truths)

            return EvalResult(
                success=True,
                mAP50=overall["mAP50"],
                mAP50_95=overall["mAP50_95"],
                precision=overall["precision"],
                recall=overall["recall"],
                per_class=per_class,
                num_images=len(dataset_images),
                num_predictions=num_preds,
                num_ground_truths=num_gts,
            )

        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            return EvalResult(
                success=False,
                mAP50=0.0,
                mAP50_95=0.0,
                precision=0.0,
                recall=0.0,
                per_class={},
                num_images=0,
                num_predictions=0,
                num_ground_truths=0,
                error=str(exc),
            )
