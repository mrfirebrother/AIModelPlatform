from __future__ import annotations

import hashlib
import io
import struct
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from backend.app.services.dataset_validation import (
    DatasetValidationError,
    DatasetValidationResult,
    DatasetValidationWarning,
    validate_yolo_dataset,
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


def _create_valid_label(path: Path, class_id: int = 0, bbox: list[float] | None = None) -> None:
    if bbox is None:
        bbox = [0.5, 0.5, 0.2, 0.2]
    path.write_text(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n", encoding="utf-8")


def _create_empty_label(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def _create_minimal_data_yaml(path: Path, num_classes: int = 2) -> None:
    content = (
        f"nc: {num_classes}\n"
        "names: ['fire', 'smoke']\n"
        "train: ./images/train\n"
        "val: ./images/val\n"
        "test: ./images/test\n"
    )
    path.write_text(content, encoding="utf-8")


def _create_yolo_dataset(
    root: Path,
    *,
    num_classes: int = 2,
    train_images: int = 3,
    val_images: int = 2,
    test_images: int = 1,
    add_test_split: bool = True,
    corrupt_image: str | None = None,
    missing_label: str | None = None,
    invalid_label: str | None = None,
    out_of_range_class: bool = False,
    empty_negative: bool = False,
) -> Path:
    """Helper to create a minimal valid YOLO dataset."""
    _create_minimal_data_yaml(root / "data.yaml", num_classes=num_classes)

    for split, count in [("train", train_images), ("val", val_images)]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img_name = f"img_{split}_{i:04d}.jpg"
            _create_valid_image(root / "images" / split / img_name)
            if missing_label == f"{split}/{img_name}":
                continue
            if empty_negative and split == "train" and i == 0:
                _create_empty_label(root / "labels" / split / f"img_{split}_{i:04d}.txt")
            elif invalid_label == f"{split}/{img_name}":
                (root / "labels" / split / f"img_{split}_{i:04d}.txt").write_text(
                    "invalid format", encoding="utf-8"
                )
            elif out_of_range_class:
                _create_valid_label(root / "labels" / split / f"img_{split}_{i:04d}.txt", class_id=num_classes)
            else:
                _create_valid_label(root / "labels" / split / f"img_{split}_{i:04d}.txt")

    if add_test_split:
        (root / "images" / "test").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "test").mkdir(parents=True, exist_ok=True)
        for i in range(test_images):
            img_name = f"img_test_{i:04d}.jpg"
            if corrupt_image == "test/" + img_name:
                (root / "images" / "test" / img_name).write_bytes(b"not-an-image")
            else:
                _create_valid_image(root / "images" / "test" / img_name)
            _create_valid_label(root / "labels" / "test" / f"img_test_{i:04d}.txt")

    return root


class TestDatasetValidationResult:
    def test_result_stores_errors_and_warnings(self) -> None:
        result = DatasetValidationResult()
        assert result.errors == []
        assert result.warnings == []
        assert result.is_valid is True

    def test_result_is_invalid_with_errors(self) -> None:
        result = DatasetValidationResult()
        result.errors.append(DatasetValidationError(code="test", message="error"))
        assert result.is_valid is False

    def test_result_can_have_warnings_only(self) -> None:
        result = DatasetValidationResult()
        result.warnings.append(DatasetValidationWarning(code="test", message="warning"))
        assert result.is_valid is True


class TestValidateYoloDataset:
    def test_valid_dataset_passes(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_yolo_dataset(root)

        result = validate_yolo_dataset(root)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_missing_data_yaml_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        (root / "images").mkdir()
        (root / "labels").mkdir()

        result = validate_yolo_dataset(root)
        assert not result.is_valid
        assert any(e.code == "missing_data_yaml" for e in result.errors)

    def test_missing_train_split_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_minimal_data_yaml(root / "data.yaml")
        (root / "images" / "val").mkdir(parents=True)
        (root / "labels" / "val").mkdir(parents=True)

        result = validate_yolo_dataset(root)
        assert not result.is_valid
        assert any(e.code == "missing_split" for e in result.errors)

    def test_missing_val_split_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_minimal_data_yaml(root / "data.yaml")
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)

        result = validate_yolo_dataset(root)
        assert not result.is_valid
        assert any(e.code == "missing_split" for e in result.errors)

    def test_missing_label_for_image_generates_warning(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_yolo_dataset(root, missing_label="train/img_train_0000.jpg")

        result = validate_yolo_dataset(root)
        assert result.is_valid
        assert any(w.code == "orphan_image" for w in result.warnings)

    def test_orphan_label_warns(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_yolo_dataset(root)
        (root / "labels" / "train" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        result = validate_yolo_dataset(root)
        assert result.is_valid
        assert any(w.code == "orphan_label" for w in result.warnings)

    def test_corrupt_image_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_yolo_dataset(root, corrupt_image="test/img_test_0000.jpg")

        result = validate_yolo_dataset(root)
        assert not result.is_valid
        assert any(e.code == "corrupt_image" for e in result.errors)

    def test_invalid_label_format_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_yolo_dataset(root, invalid_label="train/img_train_0000.jpg")

        result = validate_yolo_dataset(root)
        assert not result.is_valid
        assert any(e.code == "invalid_label_format" for e in result.errors)

    def test_out_of_range_class_id_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_yolo_dataset(root, out_of_range_class=True)

        result = validate_yolo_dataset(root)
        assert not result.is_valid
        assert any(e.code == "class_id_out_of_range" for e in result.errors)

    def test_negative_class_id_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_yolo_dataset(root)
        (root / "labels" / "train" / "img_train_0000.txt").write_text(
            "-1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )

        result = validate_yolo_dataset(root)
        assert not result.is_valid
        assert any(e.code == "class_id_out_of_range" for e in result.errors)

    def test_empty_label_is_valid_negative_sample(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_yolo_dataset(root, empty_negative=True)

        result = validate_yolo_dataset(root)
        assert result.is_valid
        assert result.negative_count > 0

    def test_missing_test_split_warns(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_yolo_dataset(root, add_test_split=False)

        result = validate_yolo_dataset(root)
        assert result.is_valid
        assert any(w.code == "missing_test_split" for w in result.warnings)

    def test_duplicate_hash_across_splits_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_minimal_data_yaml(root / "data.yaml")

        duplicate_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        for split in ["train", "val", "test"]:
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            for i in range(2):
                img_name = f"img_{i:04d}.jpg"
                duplicate_img.save(root / "images" / split / img_name)
                _create_valid_label(root / "labels" / split / f"img_{i:04d}.txt")

        result = validate_yolo_dataset(root)
        assert not result.is_valid
        assert any(e.code == "duplicate_hash_across_splits" for e in result.errors)

    def test_same_hash_in_same_split_is_ok(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_minimal_data_yaml(root / "data.yaml")

        for split in ["train", "val", "test"]:
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            for i in range(2):
                img_name = f"img_{i:04d}.jpg"
                _create_valid_image(root / "images" / split / img_name)
                _create_valid_label(root / "labels" / split / f"img_{i:04d}.txt")

        result = validate_yolo_dataset(root)
        assert result.is_valid

    def test_class_imbalance_warns(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_minimal_data_yaml(root / "data.yaml", num_classes=2)

        for split in ["train", "val", "test"]:
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)

        for i in range(10):
            _create_valid_image(root / "images" / "train" / f"img_{i:04d}.jpg")
            _create_valid_label(root / "labels" / "train" / f"img_{i:04d}.txt", class_id=0)

        for i in range(3):
            _create_valid_image(root / "images" / "val" / f"img_{i:04d}.jpg")
            _create_valid_label(root / "labels" / "val" / f"img_{i:04d}.txt", class_id=0)
        for i in range(3, 6):
            _create_valid_image(root / "images" / "val" / f"img_{i:04d}.jpg")
            _create_valid_label(root / "labels" / "val" / f"img_{i:04d}.txt", class_id=1)

        for i in range(2):
            _create_valid_image(root / "images" / "test" / f"img_{i:04d}.jpg")
            _create_valid_label(root / "labels" / "test" / f"img_{i:04d}.txt", class_id=0)

        result = validate_yolo_dataset(root)
        assert result.is_valid
        assert any(w.code == "class_imbalance" for w in result.warnings)

    def test_data_yaml_missing_nc_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        (root / "data.yaml").write_text("names: ['fire']\n", encoding="utf-8")
        (root / "images" / "train").mkdir(parents=True)
        (root / "images" / "val").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "labels" / "val").mkdir(parents=True)

        result = validate_yolo_dataset(root)
        assert not result.is_valid
        assert any(e.code == "data_yaml_error" for e in result.errors)

    def test_data_yaml_missing_names_fails(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        (root / "data.yaml").write_text("nc: 2\n", encoding="utf-8")
        (root / "images" / "train").mkdir(parents=True)
        (root / "images" / "val").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "labels" / "val").mkdir(parents=True)

        result = validate_yolo_dataset(root)
        assert not result.is_valid
        assert any(e.code == "data_yaml_error" for e in result.errors)

    def test_unlabeled_image_not_counted_as_negative(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_minimal_data_yaml(root / "data.yaml")
        for split in ["train", "val", "test"]:
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            _create_valid_image(root / "images" / split / "img.jpg")
            _create_valid_label(root / "labels" / split / "img.txt")

        _create_valid_image(root / "images" / "train" / "unlabeled.jpg")

        result = validate_yolo_dataset(root)
        assert result.is_valid
        assert any(w.code == "orphan_image" for w in result.warnings)

    def test_counts_are_correct(self, tmp_path: Path) -> None:
        root = tmp_path / "dataset"
        root.mkdir()
        _create_minimal_data_yaml(root / "data.yaml")
        for split in ["train", "val", "test"]:
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            for i in range(3):
                _create_valid_image(root / "images" / split / f"img_{i}.jpg")
                _create_valid_label(root / "labels" / split / f"img_{i}.txt")

        result = validate_yolo_dataset(root)
        assert result.train_count == 3
        assert result.val_count == 3
        assert result.test_count == 3
