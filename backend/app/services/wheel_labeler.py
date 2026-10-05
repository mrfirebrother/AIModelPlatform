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
        if folder.parent == folder:
            raise LabelerError(f"不要直接打开驱动器根目录（{folder}），请选择具体的图片文件夹")
        self.root = folder
        self.images = scan_images(folder)
        classes = folder / "classes.txt"
        if not classes.exists():
            # 新文件夹给一个占位类别，用户在界面改名/添加；已有文件一律尊重不动
            classes.write_text("object\n", encoding="utf-8")

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


def scan_images(folder: Path, limit: int = 30000) -> list[Path]:
    found: list[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = path.relative_to(folder).parts
        if "labels" in parts:
            continue
        found.append(path)
        if len(found) > limit:
            raise LabelerError(f"图片超过 {limit} 张，请选择更具体的子文件夹，不要打开过大的目录")
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
    is_drive_root = base.parent == base
    return {
        "path": str(base),
        # 盘符根目录没有上一级（.. 不显示，只能用“回到驱动器”跳回盘符列表）
        "parent": "" if is_drive_root else str(base.parent),
        "entries": entries,
        # at_root 只留给真正的盘符列表；盘符根目录标题显示盘符本身（如 D:\）
        "at_root": False,
        "is_drive_root": is_drive_root,
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
        cid, xc, yc, bw, bh = parts[:5]
        try:
            cls = int(float(cid))
        except ValueError:
            cls = 0
        if cls < 0:
            cls = 0
        xc, yc, bw, bh = (float(xc), float(yc), float(bw), float(bh))
        pw, ph = bw * width, bh * height
        boxes.append(
            {
                "x": xc * width - pw / 2,
                "y": yc * height - ph / 2,
                "w": pw,
                "h": ph,
                "cls": cls,
            }
        )
    return boxes


def write_boxes(path: Path, boxes: list[dict], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        try:
            cls = int(box.get("cls", 0))
        except (TypeError, ValueError):
            cls = 0
        if cls < 0:
            cls = 0
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
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
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
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(read_classes(library.root)))
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


#: 客户端上传落盘限制（与数据集上传同一量级）
UPLOAD_MAX_FILES = 5000
UPLOAD_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
UPLOAD_MAX_FILE_BYTES = 200 * 1024 * 1024  # 200 MB


def sanitize_upload_rel(raw: str) -> Path:
    """清洗客户端传来的相对路径（webkitRelativePath），防目录穿越。

    只允许落在上传根目录内部的图片文件；非法一律抛 LabelerError（转 400）。
    """
    text = (raw or "").strip().replace("\\", "/")
    if not text or text.startswith("/") or (len(text) > 1 and text[1] == ":"):
        raise LabelerError(f"非法的文件路径：{raw}")
    parts = [p for p in text.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        raise LabelerError(f"非法的文件路径：{raw}")
    rel = Path(*parts)
    if rel.suffix.lower() not in IMAGE_EXTS:
        raise LabelerError(f"不支持的图片类型：{raw}")
    return rel


def new_upload_root(base_dir: str) -> Path:
    """新建一次上传的临时根目录（<dataset_dir>/annotate_uploads/up_<rand8>/）。"""
    import uuid

    root = Path(base_dir) / "annotate_uploads" / f"up_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def read_classes(folder: Path) -> list[str]:
    """读文件夹的 classes.txt；没有则返回占位类别。"""
    path = folder / "classes.txt"
    if not path.exists():
        return ["object"]
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    return names or ["object"]


def write_classes(folder: Path, names: list[str]) -> list[str]:
    """写 classes.txt；清洗（去空、去重、限 50 个），非法直接报错。"""
    if not isinstance(names, list) or not names:
        raise LabelerError("类别不能为空")
    cleaned = []
    for name in names:
        if not isinstance(name, str):
            raise LabelerError("类别名必须是文字")
        name = name.strip()
        if not name:
            continue
        if name not in cleaned:
            cleaned.append(name)
    if not cleaned:
        raise LabelerError("类别不能为空")
    if len(cleaned) > 50:
        raise LabelerError("类别太多，最多 50 个")
    (folder / "classes.txt").write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    return cleaned


def prune_previous_upload(base_dir: str, old: Path | None, new: Path | None) -> bool:
    """打开新目录后删掉上一次的上传目录，磁盘只留当前这一批。

    只认 annotate_uploads/ 的直接子目录（即 new_upload_root 建出来的 up_*），
    其他路径一律不动；删除失败也不抛错（返回 False），绝不能影响打开新目录。
    """
    try:
        if old is None or new is None:
            return False
        uploads = (Path(base_dir) / "annotate_uploads").resolve()
        old_root = old.resolve()
        if old_root == new.resolve() or old_root.parent != uploads:
            return False
        shutil.rmtree(old_root, ignore_errors=True)
        return not old_root.exists()
    except OSError:
        return False


def resolve_upload_dest(root: Path, rel: Path) -> Path:
    """把清洗过的相对路径定位到 root 内部；重名文件自动加序号，不覆盖。"""
    dest = root / rel
    if not dest.resolve().is_relative_to(root.resolve()):
        raise LabelerError(f"非法的文件路径：{rel.as_posix()}")
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        for i in range(1, 10000):
            cand = dest.parent / f"{stem}_{i}{suffix}"
            if not cand.exists():
                return cand
        raise LabelerError(f"重名文件过多：{rel.as_posix()}")
    return dest
