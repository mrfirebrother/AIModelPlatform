"""Wheel annotation tool API (车轮标注工具) — ported from 车轴识别/labeler.

Folder-based YOLO box annotation with a HoughCircles wheel proposer.
All routes sit under /api/annotate and require X-API-Key (plus the UI
password gate when UI_PASSWORD is configured).
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response

from backend.app.api.dependencies import verify_api_key
from backend.app.services.wheel_labeler import (
    LIBRARY,
    LabelerError,
    browse_dir,
    box_count,
    export_dataset,
    image_record,
    jpeg_of,
    read_bgr,
    read_boxes,
    write_boxes,
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
def open_folder(payload: dict[str, Any], _key: str = Depends(verify_api_key)) -> Any:
    raw = str(payload.get("path") or "").strip().strip('"')
    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请填写文件夹路径")
    try:
        LIBRARY.open(Path(raw))
    except LabelerError as exc:
        raise _err(exc)
    return {
        "root": str(LIBRARY.root),
        "images": [image_record(LIBRARY, image) for image in LIBRARY.images],
    }


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
