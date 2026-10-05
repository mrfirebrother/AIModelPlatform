"""Wheel annotation tool API (车轮标注工具) — ported from 车轴识别/labeler.

Folder-based YOLO box annotation with a HoughCircles wheel proposer.
All routes sit under /api/annotate and require X-API-Key (plus the UI
password gate when UI_PASSWORD is configured).
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response

from backend.app.api.dependencies import get_effective_settings, verify_api_key
from backend.app.services.wheel_labeler import (
    LIBRARY,
    UPLOAD_MAX_FILE_BYTES,
    UPLOAD_MAX_FILES,
    UPLOAD_MAX_TOTAL_BYTES,
    LabelerError,
    browse_dir,
    box_count,
    export_dataset,
    image_record,
    jpeg_of,
    new_upload_root,
    prune_previous_upload,
    read_classes,
    read_bgr,
    read_boxes,
    resolve_upload_dest,
    sanitize_upload_rel,
    write_boxes,
    write_classes,
)
from backend.app.services.wheel_labeler import propose_wheels

router = APIRouter(prefix="/api/annotate", tags=["annotate"])


def _err(exc: LabelerError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/state")
def get_state(_key: str = Depends(verify_api_key)) -> Any:
    root = LIBRARY.root
    return {
        "root": str(root) if root else "",
        "images": [image_record(LIBRARY, image) for image in LIBRARY.images] if root else [],
    }


@router.post("/open")
def open_folder(
    payload: dict[str, Any],
    _key: str = Depends(verify_api_key),
    settings: Any = Depends(get_effective_settings),
) -> Any:
    raw = str(payload.get("path") or "").strip().strip('"')
    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请填写文件夹路径")
    old_root = LIBRARY.root
    try:
        LIBRARY.open(Path(raw))
    except LabelerError as exc:
        raise _err(exc)
    # 新目录已接管：删掉上一次的上传临时目录，磁盘只留当前这一批
    prune_previous_upload(settings.dataset_dir, old_root, LIBRARY.root)
    return {
        "root": str(LIBRARY.root),
        "images": [image_record(LIBRARY, image) for image in LIBRARY.images],
    }


@router.get("/classes")
def get_classes(_key: str = Depends(verify_api_key)) -> Any:
    if LIBRARY.root is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="还没有打开文件夹")
    return {"classes": read_classes(LIBRARY.root)}


@router.put("/classes")
def put_classes(payload: dict[str, Any], _key: str = Depends(verify_api_key)) -> Any:
    if LIBRARY.root is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="还没有打开文件夹")
    try:
        cleaned = write_classes(LIBRARY.root, payload.get("classes"))
    except LabelerError as exc:
        raise _err(exc)
    return {"classes": cleaned}


@router.post("/browse")
def browse(payload: dict[str, Any], _key: str = Depends(verify_api_key)) -> Any:
    """服务器端目录浏览（远程可用，替代原工具的 tkinter 选择器）。"""
    try:
        return browse_dir(str(payload.get("path") or ""))
    except LabelerError as exc:
        raise _err(exc)


@router.get("/image")
def get_image(rel: str = "", _key: str = Depends(verify_api_key)) -> Response:
    try:
        path = LIBRARY.image_by_rel(rel)
        blob = jpeg_of(path)
    except LabelerError as exc:
        raise _err(exc)
    return Response(content=blob, media_type="image/jpeg")


@router.get("/labels")
def get_labels(rel: str = "", _key: str = Depends(verify_api_key)) -> Any:
    try:
        path = LIBRARY.image_by_rel(rel)
        image = read_bgr(path)
    except LabelerError as exc:
        raise _err(exc)
    height, width = image.shape[:2]
    return {
        "width": width,
        "height": height,
        "boxes": read_boxes(LIBRARY.label_path(path), width, height),
        "seen": LIBRARY.label_path(path).exists(),
    }


@router.put("/labels")
def put_labels(rel: str = "", payload: dict[str, Any] | None = None, _key: str = Depends(verify_api_key)) -> Any:
    payload = payload or {}
    try:
        path = LIBRARY.image_by_rel(rel)
        image = read_bgr(path)
    except LabelerError as exc:
        raise _err(exc)
    height, width = image.shape[:2]
    boxes = payload.get("boxes") or []
    if not isinstance(boxes, list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="框数据不正确")
    write_boxes(LIBRARY.label_path(path), boxes, width, height)
    return {"count": box_count(LIBRARY.label_path(path))}


@router.delete("/labels")
def delete_labels(rel: str = "", _key: str = Depends(verify_api_key)) -> Any:
    try:
        path = LIBRARY.image_by_rel(rel)
    except LabelerError as exc:
        raise _err(exc)
    label = LIBRARY.label_path(path)
    if label.exists():
        label.unlink()
    return {"count": None}


@router.post("/propose")
def propose(rel: str = "", payload: dict[str, Any] | None = None, _key: str = Depends(verify_api_key)) -> Any:
    payload = payload or {}
    param2 = float(payload.get("param2") or 40)
    param2 = min(max(param2, 12), 90)
    lower_only = bool(payload.get("lowerOnly", True))
    try:
        path = LIBRARY.image_by_rel(rel)
        boxes = propose_wheels(read_bgr(path), param2, lower_only)
    except LabelerError as exc:
        raise _err(exc)
    return {"boxes": boxes}


@router.post("/export")
def export(_key: str = Depends(verify_api_key)) -> Any:
    try:
        return export_dataset(LIBRARY)
    except LabelerError as exc:
        raise _err(exc)


_CHUNK = 1024 * 1024  # 1 MB，与数据集上传一致


@router.post("/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    paths: list[str] = Form(default=[]),
    _key: str = Depends(verify_api_key),
    settings: Any = Depends(get_effective_settings),
) -> Any:
    """接收客户端目录上传（webkitdirectory）：存入服务器临时目录并可直接打开标注。

    `paths` 与 `files` 一一对应，为浏览器端的 webkitRelativePath；缺失时回退用文件名。
    只收图片扩展名；越界文件记入 skipped，不中断整批。
    """
    if len(files) > UPLOAD_MAX_FILES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"一次最多上传 {UPLOAD_MAX_FILES} 个文件",
        )
    if paths and len(paths) != len(files):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件与路径数量不一致",
        )
    try:
        root = new_upload_root(settings.dataset_dir)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"无法创建上传目录：{exc}",
        )
    saved = 0
    skipped = 0
    total = 0
    for index, f in enumerate(files):
        raw = paths[index] if paths else (f.filename or "")
        try:
            rel = sanitize_upload_rel(raw or (f.filename or ""))
        except LabelerError:
            skipped += 1
            continue
        size = 0
        dest = None
        try:
            dest = resolve_upload_dest(root, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as out:
                while True:
                    chunk = await f.read(_CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    total += len(chunk)
                    if size > UPLOAD_MAX_FILE_BYTES or total > UPLOAD_MAX_TOTAL_BYTES:
                        raise LabelerError(
                            f"超出大小限制：{rel.as_posix()}（单文件 200MB / 整批 2GB）"
                        )
                    out.write(chunk)
        except LabelerError:
            if dest is not None and dest.exists():
                try:
                    dest.unlink()
                except OSError:
                    pass
            if total > UPLOAD_MAX_TOTAL_BYTES or size > UPLOAD_MAX_FILE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="超出大小限制（单文件 200MB / 整批 2GB），已停止接收",
                )
            skipped += 1
            continue
        finally:
            await f.close()
        saved += 1
    if saved == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="没有收到可用的图片（仅支持 jpg/jpeg/png/bmp/webp/tif/tiff）",
        )
    return {"root": str(root), "images": saved, "skipped": skipped}
