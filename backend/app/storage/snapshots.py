from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

from backend.app.services.dataset_validation import (
    DatasetValidationResult,
    validate_yolo_dataset,
)
from .artifacts import ContentAddressedStore, compute_file_hash

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@dataclass
class SnapshotManifest:
    label_schema_id: UUID
    train_files: list[dict[str, str]]
    val_files: list[dict[str, str]]
    test_files: list[dict[str, str]]
    train_positive: int
    val_positive: int
    test_positive: int
    train_negative: int
    val_negative: int
    test_negative: int
    data_yaml: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        return {
            "label_schema_id": str(self.label_schema_id),
            "train_files": self.train_files,
            "val_files": self.val_files,
            "test_files": self.test_files,
            "train_positive": self.train_positive,
            "val_positive": self.val_positive,
            "test_positive": self.test_positive,
            "train_negative": self.train_negative,
            "val_negative": self.val_negative,
            "test_negative": self.test_negative,
            "data_yaml": self.data_yaml,
        }


@dataclass
class SnapshotResult:
    snapshot_id: UUID
    manifest_path: Path
    manifest_hash: str
    store_path: Path
    validation_result: DatasetValidationResult
    manifest: SnapshotManifest


def _build_file_entry(
    img_path: Path,
    label_path: Path,
    store: ContentAddressedStore,
    source_dir: Path,
) -> dict[str, str]:
    img_stored = store.store_file(img_path, extension=img_path.suffix)
    label_stored = store.store_file(label_path, extension=".txt")
    return {
        "image": str(img_path.relative_to(source_dir).as_posix()),
        "label": str(label_path.relative_to(source_dir).as_posix()),
        "image_hash": f"sha256:{img_stored.name.split('.')[0]}",
        "label_hash": f"sha256:{label_stored.name.split('.')[0]}",
        "image_stored_path": str(img_stored),
        "label_stored_path": str(label_stored),
    }


def _count_positive_negative(labels_dir: Path) -> tuple[int, int]:
    positive = 0
    negative = 0
    if not labels_dir.exists():
        return positive, negative
    for label_file in labels_dir.iterdir():
        if label_file.suffix != ".txt":
            continue
        content = label_file.read_text(encoding="utf-8").strip()
        if not content:
            negative += 1
        else:
            positive += 1
    return positive, negative


def create_dataset_snapshot(
    source_dir: Path,
    store_root: Path,
    snapshot_root: Path,
    label_schema_id: UUID,
    dataset_id: UUID,
) -> SnapshotResult:
    validation = validate_yolo_dataset(source_dir)
    if not validation.is_valid:
        error_msgs = "; ".join(e.message for e in validation.errors)
        raise ValueError(f"Dataset validation failed: {error_msgs}")

    store = ContentAddressedStore(store_root)

    manifest_files: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    split_counts: dict[str, dict[str, int]] = {
        "train": {"positive": 0, "negative": 0},
        "val": {"positive": 0, "negative": 0},
        "test": {"positive": 0, "negative": 0},
    }

    for split in ["train", "val", "test"]:
        img_dir = source_dir / "images" / split
        lbl_dir = source_dir / "labels" / split
        if not img_dir.exists():
            continue

        positive, negative = _count_positive_negative(lbl_dir)
        split_counts[split]["positive"] = positive
        split_counts[split]["negative"] = negative

        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_file = lbl_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                continue
            entry = _build_file_entry(img_file, label_file, store, source_dir)
            manifest_files[split].append(entry)

    data_yaml = {}
    data_yaml_path = source_dir / "data.yaml"
    if data_yaml_path.exists():
        data_yaml_content = data_yaml_path.read_text(encoding="utf-8")
        for line in data_yaml_content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key in ("nc", "names", "train", "val", "test"):
                data_yaml[key] = value

    manifest = SnapshotManifest(
        label_schema_id=label_schema_id,
        train_files=manifest_files["train"],
        val_files=manifest_files["val"],
        test_files=manifest_files["test"],
        train_positive=split_counts["train"]["positive"],
        val_positive=split_counts["val"]["positive"],
        test_positive=split_counts["test"]["positive"],
        train_negative=split_counts["train"]["negative"],
        val_negative=split_counts["val"]["negative"],
        test_negative=split_counts["test"]["negative"],
        data_yaml=data_yaml,
    )

    snapshot_id = uuid4()
    snapshot_dir = snapshot_root / str(snapshot_id)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

    manifest_hash = compute_file_hash(manifest_path)

    return SnapshotResult(
        snapshot_id=snapshot_id,
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        store_path=store_root,
        validation_result=validation,
        manifest=manifest,
    )
