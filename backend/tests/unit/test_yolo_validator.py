from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.app.training.yolo_validator import ValidationResult, YoloValidator


class TestYoloValidatorBasic:
    def test_validate_nonexistent_file(self, tmp_path: Path) -> None:
        validator = YoloValidator(device="cpu")
        fake_path = tmp_path / "nonexistent.pt"
        result = validator.validate(fake_path)

        assert result.is_valid is False
        assert "does not exist" in result.errors[0]

    def test_validate_directory_instead_of_file(self, tmp_path: Path) -> None:
        validator = YoloValidator(device="cpu")
        result = validator.validate(tmp_path)

        assert result.is_valid is False
        assert "not a file" in result.errors[0]

    def test_validate_unsupported_format(self, tmp_path: Path) -> None:
        validator = YoloValidator(device="cpu")
        unsupported = tmp_path / "model.bin"
        unsupported.write_bytes(b"fake data")

        result = validator.validate(unsupported)

        assert result.is_valid is False
        assert "Unsupported model format" in result.errors[0]

    def test_validate_empty_pt_file(self, tmp_path: Path) -> None:
        validator = YoloValidator(device="cpu")
        empty_file = tmp_path / "empty.pt"
        empty_file.write_bytes(b"")

        result = validator.validate(empty_file)

        assert result.is_valid is False


class TestYoloValidatorResult:
    def test_validation_result_defaults(self) -> None:
        result = ValidationResult(is_valid=True, model_type="yolo")

        assert result.num_classes is None
        assert result.input_size is None
        assert result.framework == "pytorch"
        assert result.metadata == {}
        assert result.errors == []
        assert result.warnings == []


class TestYoloValidatorMocked:
    @patch("backend.app.training.yolo_validator.YoloValidator._validate_pytorch_model")
    def test_validate_pt_file_calls_pytorch_validator(self, mock_validate: MagicMock, tmp_path: Path) -> None:
        mock_validate.return_value = ValidationResult(is_valid=True, model_type="yolo")
        pt_file = tmp_path / "model.pt"
        pt_file.write_bytes(b"\x80\x04" + b"\x00" * 100)

        validator = YoloValidator(device="cpu")
        result = validator.validate(pt_file)

        mock_validate.assert_called_once_with(pt_file)
        assert result.is_valid is True

    def test_extract_metadata_returns_dict(self, tmp_path: Path) -> None:
        fake_model = tmp_path / "model.pt"
        fake_model.write_bytes(b"\x80\x04" + b"\x00" * 100)

        validator = YoloValidator(device="cpu")
        with patch.object(validator, "_validate_pytorch_model") as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True,
                model_type="yolo",
                num_classes=80,
                metadata={"class_names": ["person", "car"]},
            )
            metadata = validator.extract_metadata(fake_model)

        assert isinstance(metadata, dict)
        assert metadata["is_valid"] is True
        assert metadata["num_classes"] == 80


class TestYoloValidatorWithRealModel:
    @pytest.mark.slow
    def test_validate_real_yolov8n(self, tmp_path: Path) -> None:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        model_path = tmp_path / "test_model.pt"
        model.save(str(model_path))

        validator = YoloValidator(device="cpu")
        result = validator.validate(model_path)

        assert result.is_valid is True
        assert result.model_type == "yolo"
        assert result.num_classes is not None
        assert result.framework == "pytorch"

    @pytest.mark.slow
    def test_validate_real_model_inference(self, tmp_path: Path) -> None:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        model_path = tmp_path / "test_model.pt"
        model.save(str(model_path))

        validator = YoloValidator(device="cpu")
        result = validator.validate(model_path)

        assert "inference_test" in result.metadata
        assert result.metadata["inference_test"] == "passed"

    @pytest.mark.slow
    def test_extract_metadata_real_model(self, tmp_path: Path) -> None:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        model_path = tmp_path / "test_model.pt"
        model.save(str(model_path))

        validator = YoloValidator(device="cpu")
        metadata = validator.extract_metadata(model_path)

        assert metadata["is_valid"] is True
        assert metadata["num_classes"] == 80
        assert "class_names" in metadata["metadata"]
