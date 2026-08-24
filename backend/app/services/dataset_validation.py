from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class DatasetValidationError:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetValidationWarning:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetValidationResult:
    errors: list[DatasetValidationError] = field(default_factory=list)
    warnings: list[DatasetValidationWarning] = field(default_factory=list)
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    negative_count: int = 0
    all_class_ids: set[int] = field(default_factory=set)
    file_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _compute_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _parse_data_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    result: dict[str, Any] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "nc":
            try:
                result["nc"] = int(value)
            except ValueError:
                result["nc"] = value
        elif key == "names":
            names_str = value.strip("[]")
            result["names"] = [n.strip().strip("'\"") for n in names_str.split(",")]
        elif key in ("train", "val", "test"):
            result[key] = value.strip("'\"")
    return result


def _split_image_paths(data_yaml: dict[str, Any]) -> dict[str, str]:
    return {
        "train": data_yaml.get("train", "images/train"),
        "val": data_yaml.get("val", "images/val"),
        "test": data_yaml.get("test", "images/test"),
    }


def get_yolo_split_paths(dataset_dir: Path) -> dict[str, Path]:
    """Return the image directory for each split defined by data.yaml."""
    dataset_root = dataset_dir.resolve()
    data_yaml = _parse_data_yaml(dataset_dir / "data.yaml")
    paths = {}
    for split, relative_path in _split_image_paths(data_yaml).items():
        path = (dataset_dir / relative_path).resolve()
        if not path.is_relative_to(dataset_root):
            raise ValueError(
                f"Dataset split path escapes dataset root: {split}={relative_path}"
            )
        label_path = _label_dir_for_images_dir(path).resolve()
        if not label_path.is_relative_to(dataset_root):
            raise ValueError(
                f"Dataset label path escapes dataset root: {split}={label_path}"
            )
        paths[split] = path
    return paths


def _label_dir_for_images_dir(img_dir: Path) -> Path:
    if img_dir.name == "images":
        return img_dir.parent / "labels"
    if img_dir.parent.name == "images":
        return img_dir.parent.parent / "labels" / img_dir.name
    return img_dir.parent / "labels"


def _validate_label_format(label_path: Path) -> tuple[bool, list[int], bool]:
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return True, [], True

    class_ids = []
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            return False, [], False
        try:
            class_id = int(parts[0])
            _ = float(parts[1])
            _ = float(parts[2])
            _ = float(parts[3])
            _ = float(parts[4])
        except (ValueError, IndexError):
            return False, [], False
        class_ids.append(class_id)
    return True, class_ids, False


def validate_yolo_dataset(dataset_dir: Path) -> DatasetValidationResult:
    result = DatasetValidationResult()

    data_yaml_path = dataset_dir / "data.yaml"
    data_yaml = _parse_data_yaml(data_yaml_path)

    if not data_yaml:
        result.errors.append(
            DatasetValidationError(code="missing_data_yaml", message="data.yaml not found or empty")
        )
    else:
        if "nc" not in data_yaml:
            result.errors.append(
                DatasetValidationError(code="data_yaml_error", message="data.yaml missing 'nc' field")
            )
        if "names" not in data_yaml:
            result.errors.append(
                DatasetValidationError(code="data_yaml_error", message="data.yaml missing 'names' field")
            )

    splits = ["train", "val", "test"]
    required_splits = ["train", "val"]

    # Read split paths from data.yaml
    try:
        split_img_paths = get_yolo_split_paths(dataset_dir)
    except ValueError as exc:
        result.errors.append(
            DatasetValidationError(code="invalid_split_path", message=str(exc))
        )
        return result

    for split in required_splits:
        img_dir = split_img_paths[split]
        if not img_dir.exists():
            result.errors.append(
                DatasetValidationError(
                    code="missing_split",
                    message=f"Missing {split} split directory: {img_dir}",
                )
            )

    has_test = split_img_paths["test"].exists()
    if not has_test:
        result.warnings.append(
            DatasetValidationWarning(code="missing_test_split", message="No test split found")
        )

    all_hashes: dict[str, list[str]] = {}
    all_class_ids: set[int] = set()

    for split in splits:
        img_dir = split_img_paths[split]
        # Derive label dir from image dir
        lbl_dir = _label_dir_for_images_dir(img_dir)
        if not img_dir.exists():
            continue

        split_images: list[str] = []
        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            img_rel = f"{split}/{img_file.name}"
            img_hash = _compute_hash(img_file)
            split_images.append(img_rel)
            all_hashes.setdefault(img_hash, []).append(img_rel)

            try:
                img = Image.open(img_file)
                img.load()
                img.close()
            except Exception:
                result.errors.append(
                    DatasetValidationError(
                        code="corrupt_image",
                        message=f"Cannot read image: {img_rel}",
                        details={"path": img_rel},
                    )
                )
                continue

            label_file = lbl_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                result.warnings.append(
                    DatasetValidationWarning(
                        code="orphan_image",
                        message=f"Image without label (not negative sample): {img_rel}",
                        details={"image": img_rel},
                    )
                )
                continue

            valid_format, class_ids, is_negative = _validate_label_format(label_file)
            if not valid_format:
                result.errors.append(
                    DatasetValidationError(
                        code="invalid_label_format",
                        message=f"Invalid label format: {split}/{img_file.stem}.txt",
                        details={"label": f"{split}/{img_file.stem}.txt"},
                    )
                )
                continue

            if is_negative:
                result.negative_count += 1
            else:
                for cid in class_ids:
                    if cid < 0 or (data_yaml.get("nc") and cid >= data_yaml["nc"]):
                        result.errors.append(
                            DatasetValidationError(
                                code="class_id_out_of_range",
                                message=f"Class ID {cid} out of range [0, {data_yaml.get('nc', '?')}) in {split}/{img_file.stem}.txt",
                                details={"label": f"{split}/{img_file.stem}.txt", "class_id": cid},
                            )
                        )
                    all_class_ids.add(cid)

        if split == "train":
            result.train_count = len(split_images)
        elif split == "val":
            result.val_count = len(split_images)
        elif split == "test":
            result.test_count = len(split_images)

    for lbl_dir_split in splits:
        img_dir = split_img_paths[lbl_dir_split]
        lbl_dir = _label_dir_for_images_dir(img_dir)
        if not lbl_dir.exists():
            continue
        for lbl_file in sorted(lbl_dir.iterdir()):
            if lbl_file.suffix != ".txt":
                continue
            alt_paths = [
                img_dir / (lbl_file.stem + ext)
                for ext in IMAGE_EXTENSIONS
            ]
            if not any(p.exists() for p in alt_paths):
                result.warnings.append(
                    DatasetValidationWarning(
                        code="orphan_label",
                        message=f"Label without matching image: {lbl_dir_split}/{lbl_file.name}",
                    )
                )

    for img_dir_split in splits:
        img_dir = split_img_paths[img_dir_split]
        if not img_dir.exists():
            continue
        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_file = _label_dir_for_images_dir(img_dir) / (img_file.stem + ".txt")
            if not label_file.exists():
                result.warnings.append(
                    DatasetValidationWarning(
                        code="orphan_image",
                        message=f"Image without label (not negative sample): {img_dir_split}/{img_file.name}",
                    )
                )

    for img_hash, files in all_hashes.items():
        if len(files) > 1:
            splits_involved = set()
            for f in files:
                splits_involved.add(f.split("/")[0])
            if len(splits_involved) > 1:
                result.warnings.append(
                    DatasetValidationWarning(
                        code="duplicate_hash_across_splits",
                        message=f"Duplicate file hash across splits: {', '.join(files)}",
                        details={"hash": img_hash, "files": files},
                    )
                )

    result.all_class_ids = all_class_ids
    result.file_hashes = all_hashes

    if data_yaml.get("nc") and data_yaml.get("names"):
        num_classes = data_yaml["nc"]
        train_val_class_counts: dict[int, int] = {i: 0 for i in range(num_classes)}
        for split in ["train", "val"]:
            lbl_dir = _label_dir_for_images_dir(split_img_paths[split])
            if not lbl_dir.exists():
                continue
            for lbl_file in lbl_dir.iterdir():
                if lbl_file.suffix != ".txt":
                    continue
                content = lbl_file.read_text(encoding="utf-8").strip()
                if not content:
                    continue
                for line in content.splitlines():
                    parts = line.strip().split()
                    if parts:
                        try:
                            cid = int(parts[0])
                            if cid in train_val_class_counts:
                                train_val_class_counts[cid] += 1
                        except ValueError:
                            pass

        total = sum(train_val_class_counts.values())
        if total > 0:
            counts_list = list(train_val_class_counts.values())
            max_count = max(counts_list) if counts_list else 1
            min_count = min(counts_list) if counts_list else 0
            if max_count > 0 and min_count / max_count < 0.3:
                result.warnings.append(
                    DatasetValidationWarning(
                        code="class_imbalance",
                        message=f"Significant class imbalance detected: {dict(train_val_class_counts)}",
                        details={"counts": dict(train_val_class_counts)},
                    )
                )

    return result
