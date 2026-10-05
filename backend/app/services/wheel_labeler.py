"""Wheel (车轮) annotation library — ported from 车轴识别/labeler/server.py.

Folder-based YOLO box annotation: open a folder of images, propose wheels with
HoughCircles, save `labels/<rel>.txt` next to the images. Flask-free; the HTTP
layer lives in api/routes/annotate.py.
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

JPEG_CACHE: dict[tuple[str, int], bytes] = {}


class LabelerError(ValueError):
    """400-able labeler error with a user-facing message."""


class Library:
    def __init__(self) -> None:
        self.root: Path | None = None
        self.images: list[Path] = []

    def open(self, folder: Path) -> None:
        folder = folder.expanduser().resolve()
        if not folder.is_dir():
            raise LabelerError(f"不是文件夹：{folder}")
        self.root = folder
        self.images = scan_images(folder)
        classes = folder / "classes.txt"
        if not classes.exists():
            classes.write_text("wheel\n", encoding="utf-8")

    def image_by_rel(self, rel: str) -> Path:
        if self.root is None:
            raise LabelerError("还没有打开文件夹")
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise LabelerError("找不到图片")
        if path.suffix.lower() not in IMAGE_EXTS:
            raise LabelerError("不是支持的图片")
        return path

    def label_path(self, image: Path) -> Path:
        assert self.root is not None
        rel = image.relative_to(self.root)
        return self.root / "labels" / rel.with_suffix(".txt")


LIBRARY = Library()


def scan_images(folder: Path) -> list[Path]:
    found: list[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = path.relative_to(folder).parts
        if "labels" in parts:
            continue
        found.append(path)
    found.sort()
    return found


def browse_dir(path: str) -> dict:
    """服务器端目录浏览（替代原工具的 tkinter 选择器，远程可用）。"""
    raw = (path or "").strip().strip('"')
    if not raw:
        # 盘符列表
        entries: list[dict] = []
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            root = Path(f"{letter}:\\")
            if root.exists():
                entries.append({"name": f"{letter}:\\", "path": f"{letter}:\\"})
        return {"path": "", "entries": entries, "at_root": True}
    base = Path(raw).expanduser()
    if not base.exists():
        raise LabelerError(f"目录不存在：{raw}")
    if base.is_file():
        base = base.parent
    try:
        entries = [
            {"name": p.name, "path": str(p)}
            for p in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            if p.is_dir()
        ]
    except PermissionError:
        entries = []
    return {
        "path": str(base),
        "parent": str(base.parent) if base.parent != base else "",
        "entries": entries,
        "at_root": base.parent == base,
    }


def read_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise LabelerError(f"无法读取图片：{path.name}")
    return image


def jpeg_of(path: Path) -> bytes:
    key = (str(path), path.stat().st_mtime_ns)
    cached = JPEG_CACHE.get(key)
    if cached is not None:
        return cached
    image = read_bgr(path)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise LabelerError("图片编码失败")
    blob = encoded.tobytes()
    if len(JPEG_CACHE) > 12:
        JPEG_CACHE.clear()
    JPEG_CACHE[key] = blob
    return blob


def box_count(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            count += 1
    return count


def read_boxes(path: Path, width: int, height: int) -> list[dict]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = parts[:5]
        xc, yc, bw, bh = (float(xc), float(yc), float(bw), float(bh))
        pw, ph = bw * width, bh * height
        boxes.append(
            {
                "x": xc * width - pw / 2,
                "y": yc * height - ph / 2,
                "w": pw,
                "h": ph,
            }
        )
    return boxes


def write_boxes(path: Path, boxes: list[dict], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        w = max(float(box["w"]), 0.0)
        h = max(float(box["h"]), 0.0)
        if w < 1 or h < 1 or width < 1 or height < 1:
            continue
        xc = (float(box["x"]) + w / 2) / width
        yc = (float(box["y"]) + h / 2) / height
        nw = w / width
        nh = h / height
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def propose_wheels(image: np.ndarray, param2: float, lower_only: bool) -> list[dict]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    min_r = max(8, int(round(height * 0.018)))
    max_r = max(min_r + 6, int(round(height * 0.14)))
    min_dist = max(min_r, int(round(min_r * 1.15)))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=120,
        param2=float(param2),
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []
    found = []
    for x, y, r in np.round(circles[0]).astype(int):
        if lower_only and y < height * 0.38:
            continue
        if r < min_r or r > max_r:
            continue
        found.append((int(x), int(y), int(r)))
    if not found and lower_only:
        return propose_wheels(image, param2, False)
    if not found:
        return []
    radii = np.array([item[2] for item in found], dtype=float)
    median = float(np.median(radii))
    found = [item for item in found if median * 0.62 <= item[2] <= median * 1.45]
    found.sort(key=lambda item: item[2], reverse=True)
    kept: list[tuple[int, int, int]] = []
    for item in found:
        if all(
            (item[0] - other[0]) ** 2 + (item[1] - other[1]) ** 2
            >= (0.75 * max(item[2], other[2])) ** 2
            for other in kept
        ):
            kept.append(item)
    if len(kept) > 20:
        kept = sorted(kept, key=lambda item: item[1], reverse=True)[:20]
    kept.sort(key=lambda item: item[0])
    boxes = []
    for x, y, r in kept:
        pad = int(round(r * 0.12))
        x1 = max(0, x - r - pad)
        y1 = max(0, y - r - pad)
        x2 = min(width, x + r + pad)
        y2 = min(height, y + r + pad)
        boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
    return boxes


def place_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def export_dataset(library: Library) -> dict:
    if library.root is None:
        raise LabelerError("还没有打开文件夹")
    out = library.root.parent / f"{library.root.name}_yolo"
    if out.resolve() == library.root.resolve() or library.root.resolve().is_relative_to(out.resolve()):
        raise LabelerError("导出目录和图片目录重叠")
    images_dir = out / "images"
    labels_dir = out / "labels"
    rows: list[str] = []
    for image in library.images:
        label = library.label_path(image)
        if not label.exists():
            continue
        rel = image.relative_to(library.root)
        dest_image = images_dir / rel
        dest_label = labels_dir / rel.with_suffix(".txt")
        place_file(image, dest_image)
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label, dest_label)
        rows.append(Path("images") / rel)
    if not rows:
        raise LabelerError("还没有已看过的图片")
    random.Random(7).shuffle(rows)
    val_count = 0 if len(rows) < 5 else max(1, int(round(len(rows) * 0.15)))
    val_rows = rows[:val_count]
    train_rows = rows[val_count:] or rows
    (out / "train.txt").write_text(
        "\n".join(row.as_posix() for row in train_rows) + "\n", encoding="utf-8"
    )
    (out / "val.txt").write_text(
        "\n".join(row.as_posix() for row in (val_rows or train_rows)) + "\n",
        encoding="utf-8",
    )
    yaml = (
        f"path: {out.as_posix()}\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "names:\n"
        "  0: wheel\n"
    )
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    return {"dir": str(out), "images": len(rows), "train": len(train_rows), "val": len(val_rows or train_rows)}


def image_record(library: Library, image: Path) -> dict:
    assert library.root is not None
    count = box_count(library.label_path(image))
    return {
        "rel": image.relative_to(library.root).as_posix(),
        "name": image.name,
        "count": count,
    }
