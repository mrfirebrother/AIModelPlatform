"""Build real YOLO dataset and model fixtures for end-to-end tests.

Creates a small YOLO dataset with 12 images (2 classes) and references
the yolov8n pretrained model for actual training validation.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent
REAL_DATASET_DIR = FIXTURES_DIR / "real_yolo_dataset"
ROOT_MODEL_PATH = FIXTURES_DIR.parent.parent / "yolov8n.pt"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RealYoloDataset:
    root_dir: Path
    data_yaml_path: Path
    nc: int
    names: list[str]
    train_images: list[Path]
    val_images: list[Path]
    test_images: list[Path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _generate_test_image(
    path: Path,
    width: int = 640,
    height: int = 480,
    seed: int = 0,
) -> None:
    """Generate a simple colored test image with a rectangle."""
    img = Image.new("RGB", (width, height), color=(30 + seed * 20, 100, 200 - seed * 10))
    draw = ImageDraw.Draw(img)
    x0 = 50 + (seed * 37) % 200
    y0 = 50 + (seed * 53) % 150
    x1 = x0 + 150
    y1 = y0 + 120
    draw.rectangle([x0, y0, x1, y1], fill=(200, 50, 50))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), "JPEG", quality=85)


def _generate_yolo_label(
    path: Path,
    class_id: int,
    x_center: float = 0.5,
    y_center: float = 0.5,
    w: float = 0.25,
    h: float = 0.25,
) -> None:
    """Write a single bounding-box label in YOLO format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{class_id} {x_center} {y_center} {w} {h}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_real_yolo_dataset(
    output_dir: Path | None = None,
    num_train: int = 10,
    num_val: int = 4,
    num_test: int = 2,
) -> RealYoloDataset:
    """Create a small YOLO dataset with real image files.

    Classes: 0 = crack, 1 = spalling
    """
    if output_dir is None:
        output_dir = REAL_DATASET_DIR

    if output_dir.exists():
        shutil.rmtree(output_dir)

    names = ["crack", "spalling"]

    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test,
    }

    all_images: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    seed = 0

    for split, count in splits.items():
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img_name = f"{split}_{i:04d}.jpg"
            lbl_name = f"{split}_{i:04d}.txt"
            img_path = img_dir / img_name
            lbl_path = lbl_dir / lbl_name

            _generate_test_image(img_path, seed=seed)
            class_id = seed % len(names)
            _generate_yolo_label(lbl_path, class_id=class_id)

            all_images[split].append(img_path)
            seed += 1

    import yaml

    data_yaml = {
        "path": str(output_dir),
        "train": str((output_dir / "images" / "train").as_posix()),
        "val": str((output_dir / "images" / "val").as_posix()),
        "nc": len(names),
        "names": names,
    }
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml.dump(data_yaml, default_flow_style=False), encoding="utf-8")

    return RealYoloDataset(
        root_dir=output_dir,
        data_yaml_path=yaml_path,
        nc=len(names),
        names=names,
        train_images=all_images["train"],
        val_images=all_images["val"],
        test_images=all_images["test"],
    )


def get_root_model_path() -> Path:
    """Return path to the root yolov8n.pt model."""
    return ROOT_MODEL_PATH


def ensure_root_model_exists() -> Path:
    """Verify root model exists, raise if not."""
    path = get_root_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Root model not found at {path}. "
            "Download yolov8n.pt first."
        )
    return path


def cleanup_real_dataset(output_dir: Path | None = None) -> None:
    """Remove generated test dataset."""
    target = output_dir or REAL_DATASET_DIR
    if target.exists():
        shutil.rmtree(target)
