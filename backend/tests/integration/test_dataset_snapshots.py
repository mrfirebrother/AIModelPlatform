from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from PIL import Image

from backend.app.storage.artifacts import ContentAddressedStore, compute_file_hash
from backend.app.storage.snapshots import SnapshotManifest, SnapshotResult, create_dataset_snapshot
from backend.app.services.dataset_validation import validate_yolo_dataset


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


def _create_yolo_dataset(root: Path, *, num_classes: int = 2) -> Path:
    _create_minimal_data_yaml(root / "data.yaml", num_classes=num_classes)
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(3):
            img_name = f"img_{split}_{i:04d}.jpg"
            _create_valid_image(root / "images" / split / img_name)
            _create_valid_label(root / "labels" / split / f"img_{split}_{i:04d}.txt")
    return root


class TestContentAddressedStore:
    def test_store_file_and_get_path(self, tmp_path: Path) -> None:
        store = ContentAddressedStore(tmp_path / "cas")
        content = b"hello world"
        path = store.store_content(content, extension=".txt")
        assert path.exists()
        assert path.read_bytes() == content

    def test_content_addressing_uses_hash(self, tmp_path: Path) -> None:
        store = ContentAddressedStore(tmp_path / "cas")
        content = b"test data"
        path1 = store.store_content(content, extension=".bin")
        path2 = store.store_content(content, extension=".bin")
        assert path1 == path2

    def test_different_content_different_path(self, tmp_path: Path) -> None:
        store = ContentAddressedStore(tmp_path / "cas")
        path1 = store.store_content(b"content1")
        path2 = store.store_content(b"content2")
        assert path1 != path2

    def test_compute_file_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello")
        h = compute_file_hash(f)
        assert h.startswith("sha256:")

    def test_store_file_returns_hash(self, tmp_path: Path) -> None:
        store = ContentAddressedStore(tmp_path / "cas")
        path = store.store_content(b"data", extension=".txt")
        assert path.exists()


class TestSnapshotManifest:
    def test_manifest_creation(self, tmp_path: Path) -> None:
        manifest = SnapshotManifest(
            label_schema_id=uuid4(),
            train_files=[{"image": "a.jpg", "label": "a.txt", "image_hash": "sha256:aaa", "label_hash": "sha256:bbb"}],
            val_files=[],
            test_files=[],
            train_positive=1,
            val_positive=0,
            test_positive=0,
            train_negative=0,
            val_negative=0,
            test_negative=0,
        )
        assert len(manifest.train_files) == 1

    def test_manifest_to_json(self, tmp_path: Path) -> None:
        manifest = SnapshotManifest(
            label_schema_id=uuid4(),
            train_files=[],
            val_files=[],
            test_files=[],
            train_positive=0,
            val_positive=0,
            test_positive=0,
            train_negative=0,
            val_negative=0,
            test_negative=0,
        )
        data = manifest.to_dict()
        assert "label_schema_id" in data
        assert "train_files" in data


class TestCreateDatasetSnapshot:
    def test_snapshot_reads_train_images_layout(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        (source / "data.yaml").write_text(
            "nc: 1\nnames: ['corrosion']\n"
            "train: train/images\nval: valid/images\ntest: test/images\n",
            encoding="utf-8",
        )
        for split in ["train", "valid", "test"]:
            (source / split / "images").mkdir(parents=True)
            (source / split / "labels").mkdir(parents=True)
            _create_valid_image(source / split / "images" / "sample.jpg")
            _create_valid_label(source / split / "labels" / "sample.txt")

        result = create_dataset_snapshot(
            source_dir=source,
            store_root=tmp_path / "cas",
            snapshot_root=tmp_path / "snapshots",
            label_schema_id=uuid4(),
            dataset_id=uuid4(),
        )

        assert len(result.manifest.train_files) == 1
        assert len(result.manifest.val_files) == 1
        assert len(result.manifest.test_files) == 1

    def test_snapshot_generates_read_only_files(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        _create_yolo_dataset(source)

        store_root = tmp_path / "cas"
        snapshot_root = tmp_path / "snapshots"
        snapshot_root.mkdir()
        label_schema_id = uuid4()

        result = create_dataset_snapshot(
            source_dir=source,
            store_root=store_root,
            snapshot_root=snapshot_root,
            label_schema_id=label_schema_id,
            dataset_id=uuid4(),
        )

        assert result.manifest_path.exists()
        assert result.store_path.exists()

    def test_snapshot_does_not_modify_source(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        _create_yolo_dataset(source)

        source_files_before = sorted(str(f.relative_to(source)) for f in source.rglob("*") if f.is_file())

        store_root = tmp_path / "cas"
        snapshot_root = tmp_path / "snapshots"
        snapshot_root.mkdir()

        create_dataset_snapshot(
            source_dir=source,
            store_root=store_root,
            snapshot_root=snapshot_root,
            label_schema_id=uuid4(),
            dataset_id=uuid4(),
        )

        source_files_after = sorted(str(f.relative_to(source)) for f in source.rglob("*") if f.is_file())
        assert source_files_before == source_files_after

    def test_snapshot_manifest_is_valid_json(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        _create_yolo_dataset(source)

        store_root = tmp_path / "cas"
        snapshot_root = tmp_path / "snapshots"
        snapshot_root.mkdir()

        result = create_dataset_snapshot(
            source_dir=source,
            store_root=store_root,
            snapshot_root=snapshot_root,
            label_schema_id=uuid4(),
            dataset_id=uuid4(),
        )

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert "train_files" in manifest_data
        assert "val_files" in manifest_data
        assert "test_files" in manifest_data
        assert "label_schema_id" in manifest_data

    def test_snapshot_content_is_in_store(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        _create_yolo_dataset(source)

        store_root = tmp_path / "cas"
        snapshot_root = tmp_path / "snapshots"
        snapshot_root.mkdir()

        result = create_dataset_snapshot(
            source_dir=source,
            store_root=store_root,
            snapshot_root=snapshot_root,
            label_schema_id=uuid4(),
            dataset_id=uuid4(),
        )

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        for file_entry in manifest_data["train_files"][:1]:
            stored_path = Path(file_entry["image_stored_path"])
            assert stored_path.exists()

    def test_snapshot_manifest_hash_matches(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        _create_yolo_dataset(source)

        store_root = tmp_path / "cas"
        snapshot_root = tmp_path / "snapshots"
        snapshot_root.mkdir()

        result = create_dataset_snapshot(
            source_dir=source,
            store_root=store_root,
            snapshot_root=snapshot_root,
            label_schema_id=uuid4(),
            dataset_id=uuid4(),
        )

        actual_hash = compute_file_hash(result.manifest_path)
        assert result.manifest_hash == actual_hash

    def test_negative_samples_recorded(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        _create_minimal_data_yaml(source / "data.yaml")
        for split in ["train", "val", "test"]:
            (source / "images" / split).mkdir(parents=True)
            (source / "labels" / split).mkdir(parents=True)
            for i in range(3):
                _create_valid_image(source / "images" / split / f"img_{i}.jpg")
                _create_empty_label(source / "labels" / split / f"img_{i}.txt")

        store_root = tmp_path / "cas"
        snapshot_root = tmp_path / "snapshots"
        snapshot_root.mkdir()

        result = create_dataset_snapshot(
            source_dir=source,
            store_root=store_root,
            snapshot_root=snapshot_root,
            label_schema_id=uuid4(),
            dataset_id=uuid4(),
        )

        manifest_data = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert manifest_data["train_negative"] == 3

    def test_orphan_labels_are_not_counted_as_images(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        _create_minimal_data_yaml(source / "data.yaml")
        for split in ["train", "val", "test"]:
            (source / "images" / split).mkdir(parents=True)
            (source / "labels" / split).mkdir(parents=True)
            _create_valid_image(source / "images" / split / "sample.jpg")
            _create_valid_label(source / "labels" / split / "sample.txt")
        _create_valid_label(source / "labels" / "train" / "orphan.txt")

        result = create_dataset_snapshot(
            source_dir=source,
            store_root=tmp_path / "cas",
            snapshot_root=tmp_path / "snapshots",
            label_schema_id=uuid4(),
            dataset_id=uuid4(),
        )

        assert result.manifest.train_positive == 1
        assert result.manifest.train_negative == 0

    def test_snapshot_with_validation_error_raises(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        _create_minimal_data_yaml(source / "data.yaml")
        (source / "images" / "train").mkdir(parents=True)
        (source / "labels" / "train").mkdir(parents=True)
        (source / "images" / "val").mkdir(parents=True)
        (source / "labels" / "val").mkdir(parents=True)
        (source / "images" / "train" / "bad.jpg").write_bytes(b"not-an-image")

        with pytest.raises(Exception):
            create_dataset_snapshot(
                source_dir=source,
                store_root=tmp_path / "cas",
                snapshot_root=tmp_path / "snapshots",
                label_schema_id=uuid4(),
                dataset_id=uuid4(),
            )
