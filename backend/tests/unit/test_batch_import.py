from __future__ import annotations

import io
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from backend.app.services.batch_import import (
    BatchImportError,
    BatchImportProgress,
    BatchImportResult,
    BatchImportStatus,
    extract_archive,
    import_dataset_from_archive,
    validate_archive_contents,
)
from backend.app.services.dataset_validation import (
    DatasetValidationError,
    DatasetValidationResult,
)


_image_counter = 0


def _create_valid_image(path: Path, size: tuple[int, int] = (640, 480)) -> None:
    global _image_counter
    _image_counter += 1
    r = (_image_counter * 37) % 256
    g = (_image_counter * 53) % 256
    b = (_image_counter * 71) % 256
    img = Image.new("RGB", size, color=(r, g, b))
    img.save(path)


def _create_valid_label(path: Path, class_id: int = 0) -> None:
    path.write_text(f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def _create_minimal_data_yaml(path: Path, num_classes: int = 2) -> None:
    content = (
        f"nc: {num_classes}\n"
        "names: ['fire', 'smoke']\n"
        "train: ./images/train\n"
        "val: ./images/val\n"
    )
    path.write_text(content, encoding="utf-8")


def _create_valid_yolo_dataset(base: Path) -> None:
    _create_minimal_data_yaml(base / "data.yaml")
    for split in ("train", "val"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(3):
            _create_valid_image(img_dir / f"img_{i:04d}.jpg")
            _create_valid_label(lbl_dir / f"img_{i:04d}.txt")


def _make_zip(dataset_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(dataset_dir.rglob("*")):
            if p.is_file():
                arcname = p.relative_to(dataset_dir).as_posix()
                zf.write(p, arcname)
    buf.seek(0)
    return buf.read()


def _make_tar(dataset_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(dataset_dir.rglob("*")):
            if p.is_file():
                arcname = p.relative_to(dataset_dir).as_posix()
                tf.add(p, arcname)
    buf.seek(0)
    return buf.read()


class TestExtractArchive:
    def test_extract_zip_to_directory(self, tmp_path: Path) -> None:
        dataset = tmp_path / "my_dataset"
        dataset.mkdir()
        _create_valid_yolo_dataset(dataset)

        archive_bytes = _make_zip(dataset)
        archive_path = tmp_path / "dataset.zip"
        archive_path.write_bytes(archive_bytes)

        result = extract_archive(archive_path, tmp_path / "output")

        assert result.extracted_path.exists()
        assert (result.extracted_path / "data.yaml").exists()
        assert (result.extracted_path / "images").is_dir()
        assert result.archive_format == "zip"
        assert result.file_count > 0

    def test_extract_tar_gz_to_directory(self, tmp_path: Path) -> None:
        dataset = tmp_path / "my_dataset"
        dataset.mkdir()
        _create_valid_yolo_dataset(dataset)

        archive_bytes = _make_tar(dataset)
        archive_path = tmp_path / "dataset.tar.gz"
        archive_path.write_bytes(archive_bytes)

        result = extract_archive(archive_path, tmp_path / "output")

        assert result.extracted_path.exists()
        assert (result.extracted_path / "data.yaml").exists()
        assert result.archive_format in ("tar", "tar.gz")

    def test_extract_zip_with_single_subdirectory(self, tmp_path: Path) -> None:
        dataset = tmp_path / "nested" / "my_dataset"
        dataset.mkdir(parents=True)
        _create_valid_yolo_dataset(dataset)

        archive_bytes = _make_zip(dataset)
        archive_path = tmp_path / "dataset.zip"
        archive_path.write_bytes(archive_bytes)

        result = extract_archive(archive_path, tmp_path / "output")

        assert (result.extracted_path / "data.yaml").exists()

    def test_extract_nonexistent_archive_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_archive(tmp_path / "nope.zip", tmp_path / "output")

    def test_extract_invalid_archive_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"not a real zip")
        with pytest.raises(BatchImportError, match="Invalid|unsupported"):
            extract_archive(bad, tmp_path / "output")

    def test_extract_unsupported_format_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "archive.7z"
        bad.write_bytes(b"\x00" * 100)
        with pytest.raises(BatchImportError, match="Unsupported"):
            extract_archive(bad, tmp_path / "output")

    def test_extract_zip_with_path_traversal_is_rejected(self, tmp_path: Path) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../../../etc/passwd", "malicious")
        buf.seek(0)
        archive_path = tmp_path / "evil.zip"
        archive_path.write_bytes(buf.read())

        with pytest.raises(BatchImportError, match="[Pp]ath.?[Tt]raversal|path_traversal"):
            extract_archive(archive_path, tmp_path / "output")

    def test_extract_tar_with_path_traversal_is_rejected(self, tmp_path: Path) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            info = tarfile.TarInfo(name="../../../etc/passwd")
            info.size = 0
            tf.addfile(info)
        buf.seek(0)
        archive_path = tmp_path / "evil.tar"
        archive_path.write_bytes(buf.read())

        with pytest.raises(BatchImportError, match="[Pp]ath.?[Tt]raversal|path_traversal"):
            extract_archive(archive_path, tmp_path / "output")

    def test_extract_empty_zip_raises(self, tmp_path: Path) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            pass
        buf.seek(0)
        archive_path = tmp_path / "empty.zip"
        archive_path.write_bytes(buf.read())

        with pytest.raises(BatchImportError, match="empty"):
            extract_archive(archive_path, tmp_path / "output")


class TestValidateArchiveContents:
    def test_valid_dataset_passes(self, tmp_path: Path) -> None:
        dataset = tmp_path / "ds"
        dataset.mkdir()
        _create_valid_yolo_dataset(dataset)

        result = validate_archive_contents(dataset)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_data_yaml_fails(self, tmp_path: Path) -> None:
        result = validate_archive_contents(tmp_path)
        assert result["valid"] is False
        assert any("data.yaml" in e for e in result["errors"])

    def test_returns_warnings(self, tmp_path: Path) -> None:
        dataset = tmp_path / "ds"
        dataset.mkdir()
        _create_minimal_data_yaml(dataset / "data.yaml")
        for split in ("train", "val"):
            (dataset / "images" / split).mkdir(parents=True)
            (dataset / "labels" / split).mkdir(parents=True)

        result = validate_archive_contents(dataset)
        assert any("test" in w.lower() or "test" in w for w in result["warnings"])


class TestBatchImportProgress:
    def test_default_progress(self) -> None:
        p = BatchImportProgress()
        assert p.status == BatchImportStatus.PENDING
        assert p.total_steps == 0
        assert p.current_step == 0
        assert p.percent == 0.0

    def test_percent_calculation(self) -> None:
        p = BatchImportProgress(total_steps=10, current_step=5)
        assert p.percent == 50.0

    def test_percent_zero_total(self) -> None:
        p = BatchImportProgress(total_steps=0, current_step=0)
        assert p.percent == 0.0

    def test_update_step(self) -> None:
        p = BatchImportProgress(total_steps=4)
        p.update_step(1, "extracting")
        assert p.current_step == 1
        assert p.status == BatchImportStatus.EXTRACTING
        assert p.message == "extracting"

    def test_finish_success(self) -> None:
        p = BatchImportProgress(total_steps=4, current_step=4)
        p.finish_success()
        assert p.status == BatchImportStatus.COMPLETED
        assert p.percent == 100.0

    def test_finish_failure(self) -> None:
        p = BatchImportProgress(total_steps=4, current_step=2)
        p.finish_failure("something broke")
        assert p.status == BatchImportStatus.FAILED
        assert p.error_message == "something broke"


class TestImportDatasetFromArchive:
    def test_full_import_flow(self, tmp_path: Path) -> None:
        dataset = tmp_path / "ds"
        dataset.mkdir()
        _create_valid_yolo_dataset(dataset)

        archive_bytes = _make_zip(dataset)
        archive_path = tmp_path / "dataset.zip"
        archive_path.write_bytes(archive_bytes)

        result = import_dataset_from_archive(
            archive_path=archive_path,
            store_root=tmp_path / "store",
            snapshot_root=tmp_path / "snapshots",
            label_schema_id="00000000-0000-0000-0000-000000000001",
            dataset_id="00000000-0000-0000-0000-000000000002",
        )

        assert result.success is True
        assert result.validation_result is not None
        assert result.progress.status == BatchImportStatus.COMPLETED

    def test_import_nonexistent_archive_fails(self, tmp_path: Path) -> None:
        result = import_dataset_from_archive(
            archive_path=tmp_path / "nope.zip",
            store_root=tmp_path / "store",
            snapshot_root=tmp_path / "snapshots",
            label_schema_id="00000000-0000-0000-0000-000000000001",
            dataset_id="00000000-0000-0000-0000-000000000002",
        )
        assert result.success is False
        assert result.progress.status == BatchImportStatus.FAILED

    def test_import_invalid_dataset_fails(self, tmp_path: Path) -> None:
        dataset = tmp_path / "ds"
        dataset.mkdir()
        (dataset / "junk.txt").write_text("not a dataset")

        archive_bytes = _make_zip(dataset)
        archive_path = tmp_path / "bad.zip"
        archive_path.write_bytes(archive_bytes)

        result = import_dataset_from_archive(
            archive_path=archive_path,
            store_root=tmp_path / "store",
            snapshot_root=tmp_path / "snapshots",
            label_schema_id="00000000-0000-0000-0000-000000000001",
            dataset_id="00000000-0000-0000-0000-000000000002",
        )
        assert result.success is False
        assert result.errors
        assert result.progress.status == BatchImportStatus.FAILED

    def test_import_tracks_progress(self, tmp_path: Path) -> None:
        dataset = tmp_path / "ds"
        dataset.mkdir()
        _create_valid_yolo_dataset(dataset)

        archive_bytes = _make_zip(dataset)
        archive_path = tmp_path / "dataset.zip"
        archive_path.write_bytes(archive_bytes)

        progress_updates: list[BatchImportProgress] = []

        def mock_progress(p: BatchImportProgress) -> None:
            progress_updates.append(p)

        result = import_dataset_from_archive(
            archive_path=archive_path,
            store_root=tmp_path / "store",
            snapshot_root=tmp_path / "snapshots",
            label_schema_id="00000000-0000-0000-0000-000000000001",
            dataset_id="00000000-0000-0000-0000-000000000002",
            on_progress=mock_progress,
        )
        assert len(progress_updates) > 0
        assert progress_updates[-1].status == BatchImportStatus.COMPLETED


class TestBatchImportError:
    def test_error_with_code_and_details(self) -> None:
        err = BatchImportError(
            code="extraction_failed",
            message="Could not extract",
            details={"archive": "test.zip"},
        )
        assert err.code == "extraction_failed"
        assert err.message == "Could not extract"
        assert err.details["archive"] == "test.zip"
        assert str(err) == "Could not extract"
