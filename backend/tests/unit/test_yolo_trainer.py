from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from backend.app.training.yolo_dataset import YoloDataset


@pytest.fixture()
def tmp_output(tmp_path: Path) -> Path:
    return tmp_path / "yolo_data"


class TestYoloDatasetFromManifest:
    def test_creates_images_and_labels_dirs(self, tmp_output: Path) -> None:
        manifest = {
            "train_files": [
                {"image": "images/train/a.jpg", "label": "labels/train/a.txt"},
            ],
            "val_files": [
                {"image": "images/val/b.jpg", "label": "labels/val/b.txt"},
            ],
            "test_files": [],
            "nc": 2,
            "names": ["fire", "smoke"],
        }
        ds = YoloDataset.from_manifest(manifest, tmp_output)
        assert (tmp_output / "images" / "train").is_dir()
        assert (tmp_output / "images" / "val").is_dir()
        assert (tmp_output / "labels" / "train").is_dir()
        assert (tmp_output / "labels" / "val").is_dir()
        assert (tmp_output / "data.yaml").is_file()

    def test_data_yaml_contains_nc_and_names(self, tmp_output: Path) -> None:
        manifest = {
            "train_files": [],
            "val_files": [],
            "test_files": [],
            "nc": 3,
            "names": ["a", "b", "c"],
        }
        ds = YoloDataset.from_manifest(manifest, tmp_output)
        import yaml

        data_yaml = yaml.safe_load((tmp_output / "data.yaml").read_text(encoding="utf-8"))
        assert data_yaml["nc"] == 3
        assert data_yaml["names"] == ["a", "b", "c"]

    def test_data_yaml_has_relative_paths(self, tmp_output: Path) -> None:
        manifest = {
            "train_files": [],
            "val_files": [],
            "test_files": [],
            "nc": 1,
            "names": ["x"],
        }
        ds = YoloDataset.from_manifest(manifest, tmp_output)
        import yaml

        data_yaml = yaml.safe_load((tmp_output / "data.yaml").read_text(encoding="utf-8"))
        assert "train" in data_yaml
        assert "val" in data_yaml
        assert data_yaml["train"].endswith("images/train")
        assert data_yaml["val"].endswith("images/val")

    def test_stores_files_to_correct_split(self, tmp_path: Path, tmp_output: Path) -> None:
        img_a = tmp_path / "001.jpg"
        img_a.write_bytes(b"\xff\xd8\xff\xe0fake")
        lbl_a = tmp_path / "001.txt"
        lbl_a.write_text("0 0.5 0.5 0.1 0.1\n")
        img_b = tmp_path / "002.jpg"
        img_b.write_bytes(b"\xff\xd8\xff\xe0fake")
        lbl_b = tmp_path / "002.txt"
        lbl_b.write_text("0 0.5 0.5 0.1 0.1\n")
        img_c = tmp_path / "100.jpg"
        img_c.write_bytes(b"\xff\xd8\xff\xe0fake")
        lbl_c = tmp_path / "100.txt"
        lbl_c.write_text("0 0.5 0.5 0.1 0.1\n")

        manifest = {
            "train_files": [
                {
                    "image": "images/train/001.jpg",
                    "label": "labels/train/001.txt",
                    "image_stored_path": str(img_a),
                    "label_stored_path": str(lbl_a),
                },
                {
                    "image": "images/train/002.jpg",
                    "label": "labels/train/002.txt",
                    "image_stored_path": str(img_b),
                    "label_stored_path": str(lbl_b),
                },
            ],
            "val_files": [
                {
                    "image": "images/val/100.jpg",
                    "label": "labels/val/100.txt",
                    "image_stored_path": str(img_c),
                    "label_stored_path": str(lbl_c),
                },
            ],
            "test_files": [],
            "nc": 1,
            "names": ["obj"],
        }
        ds = YoloDataset.from_manifest(manifest, tmp_output)
        assert (tmp_output / "images" / "train" / "001.jpg").exists()
        assert (tmp_output / "images" / "train" / "002.jpg").exists()
        assert (tmp_output / "images" / "val" / "100.jpg").exists()

    def test_empty_manifest_creates_dirs_and_yaml(self, tmp_output: Path) -> None:
        manifest = {
            "train_files": [],
            "val_files": [],
            "test_files": [],
            "nc": 0,
            "names": [],
        }
        ds = YoloDataset.from_manifest(manifest, tmp_output)
        assert (tmp_output / "data.yaml").is_file()
        assert (tmp_output / "images" / "train").is_dir()


class TestYoloDatasetWithStoredPaths:
    def test_copies_from_stored_paths(self, tmp_path: Path, tmp_output: Path) -> None:
        source_img = tmp_path / "src_img.jpg"
        source_img.write_bytes(b"\xff\xd8\xff\xe0fake_jpeg")
        source_lbl = tmp_path / "src_label.txt"
        source_lbl.write_text("0 0.5 0.5 0.1 0.1\n")

        manifest = {
            "train_files": [
                {
                    "image": "images/train/001.jpg",
                    "label": "labels/train/001.txt",
                    "image_stored_path": str(source_img),
                    "label_stored_path": str(source_lbl),
                },
            ],
            "val_files": [],
            "test_files": [],
            "nc": 1,
            "names": ["obj"],
        }
        ds = YoloDataset.from_manifest(manifest, tmp_output)
        target_img = tmp_output / "images" / "train" / "001.jpg"
        assert target_img.exists()
        assert target_img.read_bytes() == b"\xff\xd8\xff\xe0fake_jpeg"
        target_lbl = tmp_output / "labels" / "train" / "001.txt"
        assert target_lbl.exists()
        assert target_lbl.read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"


class TestYoloDatasetResolveDevice:
    def test_cpu_fallback_when_no_gpu(self) -> None:
        assert YoloDataset.resolve_device("cpu") == "cpu"

    def test_cpu_requested_returns_cpu(self) -> None:
        assert YoloDataset.resolve_device("cpu") == "cpu"
