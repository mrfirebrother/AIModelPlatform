from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from backend.app.inference.yolo_engine import YoloEngine
from backend.app.inference.preprocessor import Preprocessor
from backend.app.inference.postprocessor import Postprocessor, Detection
from backend.app.workers.inference_worker import InferenceWorker


def _make_tiny_image_bytes(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_image_base64(width: int = 64, height: int = 64) -> str:
    return base64.b64encode(_make_tiny_image_bytes(width, height)).decode()


class TestEndToEndInference:
    """Integration test: decode -> preprocess -> engine -> postprocess -> detections."""

    def test_full_pipeline_with_mock_engine(self, tmp_path: Path) -> None:
        preprocessor = Preprocessor(target_size=(64, 64))
        postprocessor = Postprocessor(
            confidence_threshold=0.5,
            class_names={0: "fire", 1: "smoke"},
        )

        b64 = _make_image_base64()
        pil_img = preprocessor.decode_image(b64, "png")
        resized = preprocessor.resize(pil_img)
        input_tensor = preprocessor.to_numpy(resized)

        assert input_tensor.shape == (1, 3, 64, 64)

        raw_detections = [
            Detection(class_name="", confidence=0.9, bbox=[10, 20, 100, 200], class_id=0),
            Detection(class_name="", confidence=0.3, bbox=[5, 5, 50, 50], class_id=1),
            Detection(class_name="", confidence=0.7, bbox=[12, 22, 102, 202], class_id=0),
        ]

        filtered = postprocessor.filter_by_confidence(raw_detections)
        assert len(filtered) == 2

        mapped = [postprocessor.apply_class_mapping(d) for d in filtered]
        assert mapped[0].class_name == "fire"
        assert mapped[1].class_name == "fire"

        final = postprocessor.apply_nms(mapped)
        assert len(final) >= 1

        results = [d.to_dict() for d in final]
        assert all("class_name" in r for r in results)
        assert all("confidence" in r for r in results)
        assert all("bbox" in r for r in results)

    def test_full_pipeline_with_yolo_engine(self, tmp_path: Path) -> None:
        engine = YoloEngine(cache_dir=tmp_path)
        with patch("backend.app.inference.yolo_engine.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "fire", 1: "smoke"}
            mock_result = MagicMock()
            mock_boxes = MagicMock()
            mock_boxes.xyxy = [[10, 20, 100, 200], [50, 60, 150, 160]]
            mock_boxes.conf = [0.95, 0.2]
            mock_boxes.cls = [0, 1]
            mock_result.boxes = mock_boxes
            mock_model.return_value = [mock_result]
            MockYOLO.return_value = mock_model

            engine.load_model("/fake.pt", model_key="test-model")

            preprocessor = Preprocessor()
            postprocessor = Postprocessor(
                confidence_threshold=0.5,
                class_names={0: "fire", 1: "smoke"},
            )

            b64 = _make_image_base64()
            pil_img = preprocessor.decode_image(b64, "png")

            detections = engine.predict(
                model_key="test-model",
                image_bytes=_make_tiny_image_bytes(),
                image_format="png",
            )

            filtered = postprocessor.filter_by_confidence(detections)
            mapped = [postprocessor.apply_class_mapping(d) for d in filtered]
            final = postprocessor.apply_nms(mapped)

            assert len(final) == 1
            assert final[0].class_name == "fire"
            assert final[0].confidence == pytest.approx(0.95, abs=0.01)


class TestInferenceWorkerIntegration:
    def test_worker_start_stop_cycle(self, tmp_path: Path) -> None:
        worker = InferenceWorker()
        from uuid import uuid4

        iid = uuid4()
        worker.start(iid)
        assert worker.is_running(iid) is True
        worker.stop(iid)
        assert worker.is_running(iid) is False

    def test_worker_health_check(self, tmp_path: Path) -> None:
        worker = InferenceWorker()
        from uuid import uuid4

        iid = uuid4()
        worker.start(iid)
        assert worker.health_check(iid) is True
        worker.stop(iid)
        assert worker.health_check(iid) is False
