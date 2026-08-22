from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.app.evaluation.yolo_evaluator import EvalResult, YoloEvaluator


class TestEvalResult:
    def test_dataclass_fields(self) -> None:
        result = EvalResult(
            success=True,
            mAP50=0.85,
            mAP50_95=0.65,
            precision=0.9,
            recall=0.8,
            per_class={0: {"precision": 0.9, "recall": 0.8}},
            num_images=100,
            num_predictions=50,
            num_ground_truths=45,
        )
        assert result.success is True
        assert result.mAP50 == pytest.approx(0.85)
        assert result.mAP50_95 == pytest.approx(0.65)
        assert result.precision == pytest.approx(0.9)
        assert result.recall == pytest.approx(0.8)
        assert result.per_class[0]["precision"] == pytest.approx(0.9)
        assert result.num_images == 100

    def test_failed_result(self) -> None:
        result = EvalResult(
            success=False,
            mAP50=0.0,
            mAP50_95=0.0,
            precision=0.0,
            recall=0.0,
            per_class={},
            num_images=0,
            num_predictions=0,
            num_ground_truths=0,
            error="Model load failed",
        )
        assert result.success is False
        assert result.error == "Model load failed"


class TestYoloEvaluator:
    def test_init_default_params(self) -> None:
        evaluator = YoloEvaluator()
        assert evaluator.confidence_threshold == 0.25
        assert evaluator.iou_threshold == 0.45
        assert evaluator.device == "cpu"

    def test_init_custom_params(self) -> None:
        evaluator = YoloEvaluator(
            confidence_threshold=0.5,
            iou_threshold=0.6,
            device="cuda:0",
        )
        assert evaluator.confidence_threshold == 0.5
        assert evaluator.iou_threshold == 0.6
        assert evaluator.device == "cuda:0"

    def test_load_model_success(self, tmp_path: Path) -> None:
        evaluator = YoloEvaluator()
        with patch("backend.app.evaluation.yolo_evaluator.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "fire", 1: "smoke"}
            MockYOLO.return_value = mock_model

            model_info = evaluator.load_model(str(tmp_path / "model.pt"))
            assert model_info.num_classes == 2
            assert model_info.class_names == {0: "fire", 1: "smoke"}

    def test_load_model_caches(self, tmp_path: Path) -> None:
        evaluator = YoloEvaluator()
        with patch("backend.app.evaluation.yolo_evaluator.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "obj"}
            MockYOLO.return_value = mock_model

            evaluator.load_model(str(tmp_path / "model.pt"))
            evaluator.load_model(str(tmp_path / "model.pt"))
            assert MockYOLO.call_count == 1

    def test_unload_model(self, tmp_path: Path) -> None:
        evaluator = YoloEvaluator()
        with patch("backend.app.evaluation.yolo_evaluator.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "obj"}
            MockYOLO.return_value = mock_model

            evaluator.load_model(str(tmp_path / "model.pt"))
            assert evaluator.unload_model() is True

    def test_predict_empty_dataset(self, tmp_path: Path) -> None:
        evaluator = YoloEvaluator()
        with patch("backend.app.evaluation.yolo_evaluator.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "fire"}
            MockYOLO.return_value = mock_model

            evaluator.load_model(str(tmp_path / "model.pt"))
            result = evaluator.predict_on_dataset([])
            assert result == []

    def test_predict_single_image(self, tmp_path: Path) -> None:
        evaluator = YoloEvaluator()
        with patch("backend.app.evaluation.yolo_evaluator.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "fire"}

            mock_result = MagicMock()
            mock_boxes = MagicMock()
            mock_boxes.xyxy = [[10, 20, 100, 200]]
            mock_boxes.conf = [0.92]
            mock_boxes.cls = [0]
            mock_result.boxes = mock_boxes
            mock_model.return_value = [mock_result]

            MockYOLO.return_value = mock_model
            evaluator.load_model(str(tmp_path / "model.pt"))

            mock_image = MagicMock()
            predictions = evaluator._predict_single(mock_model, mock_image)
            assert len(predictions) == 1
            assert predictions[0]["class_id"] == 0
            assert predictions[0]["score"] == pytest.approx(0.92)
            assert predictions[0]["bbox"] == [10, 20, 100, 200]

    def test_evaluate_no_model_raises(self) -> None:
        evaluator = YoloEvaluator()
        with pytest.raises(ValueError, match="Model not loaded"):
            evaluator.evaluate(
                dataset_images=[],
                dataset_ground_truths=[],
                class_ids=[0],
            )

    def test_evaluate_perfect_match(self, tmp_path: Path) -> None:
        evaluator = YoloEvaluator()
        with patch("backend.app.evaluation.yolo_evaluator.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "fire"}

            mock_result = MagicMock()
            mock_boxes = MagicMock()
            mock_boxes.xyxy = [[0, 0, 10, 10]]
            mock_boxes.conf = [0.95]
            mock_boxes.cls = [0]
            mock_result.boxes = mock_boxes
            mock_model.return_value = [mock_result]

            MockYOLO.return_value = mock_model
            evaluator.load_model(str(tmp_path / "model.pt"))

            images = [MagicMock()]
            ground_truths = [[{"class_id": 0, "bbox": [0, 0, 10, 10]}]]

            result = evaluator.evaluate(
                dataset_images=images,
                dataset_ground_truths=ground_truths,
                class_ids=[0],
            )
            assert result.success is True
            assert result.mAP50 == pytest.approx(1.0)
            assert result.precision == pytest.approx(1.0)
            assert result.recall == pytest.approx(1.0)

    def test_evaluate_no_match(self, tmp_path: Path) -> None:
        evaluator = YoloEvaluator()
        with patch("backend.app.evaluation.yolo_evaluator.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "fire"}

            mock_result = MagicMock()
            mock_boxes = MagicMock()
            mock_boxes.xyxy = [[50, 50, 60, 60]]
            mock_boxes.conf = [0.9]
            mock_boxes.cls = [0]
            mock_result.boxes = mock_boxes
            mock_model.return_value = [mock_result]

            MockYOLO.return_value = mock_model
            evaluator.load_model(str(tmp_path / "model.pt"))

            images = [MagicMock()]
            ground_truths = [[{"class_id": 0, "bbox": [0, 0, 10, 10]}]]

            result = evaluator.evaluate(
                dataset_images=images,
                dataset_ground_truths=ground_truths,
                class_ids=[0],
            )
            assert result.success is True
            assert result.mAP50 == pytest.approx(0.0)
            assert result.num_predictions == 1
            assert result.num_ground_truths == 1

    def test_compute_overall_metrics(self) -> None:
        evaluator = YoloEvaluator()
        all_predictions = [
            [{"class_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9}],
            [{"class_id": 0, "bbox": [20, 20, 30, 30], "score": 0.8}],
        ]
        all_ground_truths = [
            [{"class_id": 0, "bbox": [0, 0, 10, 10]}],
            [{"class_id": 0, "bbox": [20, 20, 30, 30]}],
        ]

        metrics = evaluator._compute_overall_metrics(
            all_predictions, all_ground_truths, class_ids=[0]
        )
        assert metrics["mAP50"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)

    def test_compute_overall_metrics_empty(self) -> None:
        evaluator = YoloEvaluator()
        metrics = evaluator._compute_overall_metrics([], [], class_ids=[0])
        assert metrics["mAP50"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
