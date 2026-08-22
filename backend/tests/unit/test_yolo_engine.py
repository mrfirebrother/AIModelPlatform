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


def _make_tiny_image_bytes(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_image_base64(width: int = 64, height: int = 64) -> str:
    return base64.b64encode(_make_tiny_image_bytes(width, height)).decode()


class TestPreprocessor:
    def test_decode_base64_to_pil(self) -> None:
        pp = Preprocessor()
        b64 = _make_image_base64()
        pil = pp.decode_image(b64, "png")
        assert pil.size == (64, 64)

    def test_decode_supports_jpeg(self) -> None:
        pp = Preprocessor()
        img = Image.new("RGB", (32, 32))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        pil = pp.decode_image(b64, "jpg")
        assert pil.size == (32, 32)

    def test_invalid_base64_raises(self) -> None:
        pp = Preprocessor()
        with pytest.raises(ValueError, match="Invalid base64"):
            pp.decode_image("not-valid-base64!!!", "png")

    def test_resize_to_model_input(self) -> None:
        pp = Preprocessor(target_size=(320, 320))
        img = Image.new("RGB", (1920, 1080))
        resized = pp.resize(img)
        assert resized.size == (320, 320)

    def test_normalize_returns_float_tensor(self) -> None:
        pp = Preprocessor()
        img = Image.new("RGB", (10, 10), color=(255, 128, 0))
        tensor = pp.to_tensor(img)
        assert tensor.dtype.name == "float32"
        assert tensor.max() <= 1.0
        assert tensor.min() >= 0.0

    def test_to_numpy_returns_nchw(self) -> None:
        pp = Preprocessor()
        img = Image.new("RGB", (16, 16))
        arr = pp.to_numpy(img)
        assert arr.shape == (1, 3, 16, 16)


class TestPostprocessor:
    def test_detection_dataclass(self) -> None:
        d = Detection(
            class_name="fire",
            confidence=0.95,
            bbox=[10, 20, 100, 200],
            class_id=0,
        )
        assert d.class_name == "fire"
        assert d.confidence == pytest.approx(0.95)
        assert d.bbox == [10, 20, 100, 200]

    def test_to_dict(self) -> None:
        d = Detection(
            class_name="smoke",
            confidence=0.8,
            bbox=[0, 0, 50, 50],
            class_id=1,
        )
        result = d.to_dict()
        assert result["class_name"] == "smoke"
        assert result["confidence"] == pytest.approx(0.8)
        assert result["bbox"] == [0, 0, 50, 50]

    def test_confidence_filter(self) -> None:
        pp = Postprocessor(confidence_threshold=0.5)
        detections = [
            Detection("a", 0.9, [0, 0, 10, 10], 0),
            Detection("b", 0.3, [0, 0, 10, 10], 1),
            Detection("c", 0.6, [0, 0, 10, 10], 2),
        ]
        filtered = pp.filter_by_confidence(detections)
        assert len(filtered) == 2
        assert all(d.confidence >= 0.5 for d in filtered)

    def test_class_mapping(self) -> None:
        pp = Postprocessor(class_names={0: "fire", 1: "smoke"})
        d = Detection(class_name="", confidence=0.9, bbox=[0, 0, 10, 10], class_id=0)
        mapped = pp.apply_class_mapping(d)
        assert mapped.class_name == "fire"

    def test_nms_filters_overlapping(self) -> None:
        pp = Postprocessor(nms_iou_threshold=0.5)
        detections = [
            Detection("a", 0.9, [10, 10, 50, 50], 0),
            Detection("a", 0.8, [12, 12, 52, 52], 0),
            Detection("b", 0.7, [200, 200, 300, 300], 1),
        ]
        result = pp.apply_nms(detections)
        assert len(result) == 2

    def test_empty_detections(self) -> None:
        pp = Postprocessor()
        assert pp.filter_by_confidence([]) == []
        assert pp.apply_nms([]) == []
        assert pp.process([]) == []


class TestYoloEngine:
    def test_load_model_returns_model_info(self, tmp_path: Path) -> None:
        engine = YoloEngine(cache_dir=tmp_path)
        with patch("backend.app.inference.yolo_engine.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "fire", 1: "smoke"}
            MockYOLO.return_value = mock_model

            info = engine.load_model("/fake/path.pt", model_key="test-model")
            assert info.model_key == "test-model"
            assert info.num_classes == 2
            assert info.class_names == {0: "fire", 1: "smoke"}

    def test_model_caching(self, tmp_path: Path) -> None:
        engine = YoloEngine(cache_dir=tmp_path)
        with patch("backend.app.inference.yolo_engine.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "obj"}
            MockYOLO.return_value = mock_model

            engine.load_model("/fake/path.pt", model_key="cached")
            engine.load_model("/fake/path.pt", model_key="cached")
            assert MockYOLO.call_count == 1

    def test_unload_model(self, tmp_path: Path) -> None:
        engine = YoloEngine(cache_dir=tmp_path)
        with patch("backend.app.inference.yolo_engine.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "obj"}
            MockYOLO.return_value = mock_model

            engine.load_model("/fake/path.pt", model_key="unload-me")
            assert engine.unload_model("unload-me") is True
            assert engine.get_model("unload-me") is None

    def test_unload_nonexistent_returns_false(self, tmp_path: Path) -> None:
        engine = YoloEngine(cache_dir=tmp_path)
        assert engine.unload_model("no-such-model") is False

    def test_inference_returns_detections(self, tmp_path: Path) -> None:
        engine = YoloEngine(cache_dir=tmp_path)
        with patch("backend.app.inference.yolo_engine.YOLO") as MockYOLO:
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

            engine.load_model("/fake/path.pt", model_key="detect")
            detections = engine.predict(
                model_key="detect",
                image_bytes=_make_tiny_image_bytes(),
                image_format="png",
            )
            assert len(detections) >= 1
            assert detections[0].class_name == "fire"
            assert detections[0].confidence == pytest.approx(0.92, abs=0.01)

    def test_predict_unloaded_model_raises(self, tmp_path: Path) -> None:
        engine = YoloEngine(cache_dir=tmp_path)
        with pytest.raises(KeyError, match="not loaded"):
            engine.predict(
                model_key="missing",
                image_bytes=_make_tiny_image_bytes(),
                image_format="png",
            )

    def test_list_models(self, tmp_path: Path) -> None:
        engine = YoloEngine(cache_dir=tmp_path)
        with patch("backend.app.inference.yolo_engine.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "a"}
            MockYOLO.return_value = mock_model

            engine.load_model("/f1.pt", model_key="m1")
            engine.load_model("/f2.pt", model_key="m2")
            models = engine.list_models()
            assert set(models) == {"m1", "m2"}
