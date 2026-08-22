from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.services.model_import import (
    ModelImportResult,
    compute_file_hash,
    import_root_model,
    validate_model_file,
    validate_model_real,
)


@pytest.fixture()
def fake_model_file(tmp_path: Path) -> Path:
    model_file = tmp_path / "fake_model.pt"
    model_file.write_bytes(b"\x80\x04" + b"\x00" * 2000)
    return model_file


@pytest.fixture()
def real_model_file(tmp_path: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model_path = tmp_path / "real_model.pt"
    model.save(str(model_path))
    return model_path


class TestModelImportBasic:
    def test_validate_model_file_basic(self, fake_model_file: Path) -> None:
        is_valid, errors = validate_model_file(fake_model_file)
        assert is_valid is True
        assert errors == []

    def test_compute_file_hash(self, fake_model_file: Path) -> None:
        hash1 = compute_file_hash(fake_model_file)
        hash2 = compute_file_hash(fake_model_file)
        assert hash1 == hash2
        assert hash1.startswith("sha256:")


class TestModelImportRealValidation:
    @pytest.mark.slow
    def test_validate_real_model(self, real_model_file: Path) -> None:
        is_valid, errors, metadata = validate_model_real(real_model_file, device="cpu")

        assert is_valid is True
        assert errors == []
        assert metadata["model_type"] == "yolo"
        assert metadata["num_classes"] == 80

    @pytest.mark.slow
    def test_validate_real_model_inference(self, real_model_file: Path) -> None:
        is_valid, errors, metadata = validate_model_real(real_model_file, device="cpu")

        assert is_valid is True
        assert metadata.get("inference_test") == "passed"

    def test_validate_fake_model_real_check(self, fake_model_file: Path) -> None:
        is_valid, errors, metadata = validate_model_real(fake_model_file, device="cpu")

        assert is_valid is False
        assert len(errors) > 0


class TestModelImportRootModel:
    @pytest.mark.slow
    def test_import_real_model_approved(self, real_model_file: Path) -> None:
        from unittest.mock import MagicMock

        session = MagicMock()
        result = import_root_model(
            session,
            real_model_file,
            real_validation=True,
            device="cpu",
        )

        assert isinstance(result, ModelImportResult)
        assert result.status == "approved"
        assert result.validation_errors == []
        assert result.metadata.get("model_type") == "yolo"
        assert result.metadata.get("num_classes") == 80
        session.add.assert_called_once()
        session.flush.assert_called_once()

    def test_import_fake_model_rejected(self, fake_model_file: Path) -> None:
        from unittest.mock import MagicMock

        session = MagicMock()
        result = import_root_model(
            session,
            fake_model_file,
            real_validation=True,
            device="cpu",
        )

        assert result.status == "rejected"
        assert len(result.validation_errors) > 0
        session.add.assert_not_called()

    def test_import_without_real_validation(self, fake_model_file: Path) -> None:
        from unittest.mock import MagicMock

        session = MagicMock()
        result = import_root_model(
            session,
            fake_model_file,
            real_validation=False,
        )

        assert result.status == "approved"
        session.add.assert_called_once()

    @pytest.mark.slow
    def test_import_preserves_metadata(self, real_model_file: Path) -> None:
        from unittest.mock import MagicMock

        session = MagicMock()
        custom_metadata = {"custom_key": "custom_value"}
        result = import_root_model(
            session,
            real_model_file,
            metadata_json=custom_metadata,
            real_validation=True,
            device="cpu",
        )

        assert result.metadata.get("custom_key") == "custom_value"
        assert result.metadata.get("model_type") == "yolo"


class TestModelImportHashConsistency:
    def test_hash_same_file(self, fake_model_file: Path) -> None:
        hash1 = compute_file_hash(fake_model_file)
        hash2 = compute_file_hash(fake_model_file)
        assert hash1 == hash2

    def test_hash_different_files(self, tmp_path: Path) -> None:
        file1 = tmp_path / "model1.pt"
        file1.write_bytes(b"model1" + b"\x00" * 1000)

        file2 = tmp_path / "model2.pt"
        file2.write_bytes(b"model2" + b"\x00" * 1000)

        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)
        assert hash1 != hash2
