from __future__ import annotations

import hashlib
import secrets
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_effective_settings, get_db, verify_api_key
from backend.app.observability.operation_log import log_operation
from backend.app.repositories.model_repository import (
    create_model_node,
    delete_model_node,
    get_model_node,
    list_child_models,
    list_root_models,
)
from backend.app.schemas.model import (
    ModelNodeCreate,
    ModelNodeListResponse,
    ModelNodeResponse,
    ModelNodeStatusUpdate,
)

_MAX_MODEL_BYTES = 500 * 1024 * 1024  # 500 MB

router = APIRouter(prefix="/api/models", tags=["models"])


@router.post("", response_model=ModelNodeResponse, status_code=status.HTTP_201_CREATED)
def create_model(
    payload: ModelNodeCreate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    if payload.parent_id is None:
        existing_roots = list_root_models(db)
        if existing_roots:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only one root model is allowed. Delete the existing root model first.",
            )
    try:
        model = create_model_node(
            db,
            task_type=payload.task_type,
            model_family=payload.model_family,
            artifact_path=payload.artifact_path,
            artifact_hash=payload.artifact_hash,
            name=payload.name,
            parent_id=payload.parent_id,
            label_schema_id=payload.label_schema_id,
            dataset_snapshot_id=payload.dataset_snapshot_id,
            training_attempt_id=payload.training_attempt_id,
            framework=payload.framework,
            artifact_format=payload.artifact_format,
            status=payload.status,
            metadata_json=payload.metadata_json,
        )
        log_operation(
            db,
            operation_type="model.create",
            resource_type="model_node",
            resource_id=model.id,
            status="success",
            summary_json={"model_id": str(model.id)},
        )
        return model
    except Exception as exc:
        log_operation(
            db,
            operation_type="model.create",
            status="error",
            error_summary=str(exc),
        )
        raise


@router.get("", response_model=ModelNodeListResponse)
def list_models(
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    from sqlalchemy import select as _select
    from backend.app.models import ModelNode as _ModelNode
    models = list(db.execute(_select(_ModelNode)).scalars().all())
    return ModelNodeListResponse(models=models, total=len(models))


@router.get("/{model_id}", response_model=ModelNodeResponse)
def get_model(
    model_id: str,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    # Try UUID first, then code lookup
    try:
        uuid_id = UUID(model_id)
        model = get_model_node(db, uuid_id)
    except ValueError:
        # Not a UUID, try code lookup
        from sqlalchemy import select as _select
        from backend.app.models import ModelNode as _ModelNode
        model = db.execute(
            _select(_ModelNode).where(_ModelNode.code == model_id)
        ).scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    return model


@router.patch("/{model_id}/status", response_model=ModelNodeResponse)
def update_model_status(
    model_id: str,
    payload: ModelNodeStatusUpdate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    # Try UUID first, then code lookup
    try:
        uuid_id = UUID(model_id)
        model = get_model_node(db, uuid_id)
    except ValueError:
        from sqlalchemy import select as _select
        from backend.app.models import ModelNode as _ModelNode
        model = db.execute(
            _select(_ModelNode).where(_ModelNode.code == model_id)
        ).scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    try:
        model.status = payload.status
        db.flush()
        log_operation(
            db,
            operation_type="model.update_status",
            resource_type="model_node",
            resource_id=model.id,
            status="success",
            summary_json={"model_id": str(model.id), "new_status": payload.status},
        )
        return model
    except Exception as exc:
        log_operation(
            db,
            operation_type="model.update_status",
            status="error",
            error_summary=str(exc),
        )
        raise


@router.post("/upload")
async def upload_model(
    file: UploadFile,
    _key: str = Depends(verify_api_key),
) -> Any:
    if not file.filename or not file.filename.endswith(".pt"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .pt files are allowed",
        )

    settings = get_effective_settings()
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{secrets.token_hex(8)}_{file.filename}"
    dest = model_dir / safe_name

    sha256 = hashlib.sha256()
    total = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_MODEL_BYTES:
                    break
                f.write(chunk)
                sha256.update(chunk)
        if total > _MAX_MODEL_BYTES:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds {_MAX_MODEL_BYTES // (1024 * 1024)} MB limit",
            )
    except HTTPException:
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {exc}",
        )

    return {
        "filename": file.filename,
        "file_path": str(dest),
        "size": total,
        "sha256": f"sha256:{sha256.hexdigest()}",
    }


@router.delete("/{model_id}")
def delete_model(
    model_id: str,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    # Try UUID first, then code lookup
    try:
        uuid_id = UUID(model_id)
        model = get_model_node(db, uuid_id)
    except ValueError:
        from sqlalchemy import select as _select
        from backend.app.models import ModelNode as _ModelNode
        model = db.execute(
            _select(_ModelNode).where(_ModelNode.code == model_id)
        ).scalar_one_or_none()
        uuid_id = model.id if model else None
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    children = list_child_models(db, uuid_id)
    if children:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete model with children. Delete child models first.",
        )

    # Cascade delete dependent evaluations (FK RESTRICT would otherwise cause 500)
    try:
        from sqlalchemy import select as _select  # local import to avoid cycle
        from backend.app.models import Evaluation as _Evaluation

        evals = list(db.execute(_select(_Evaluation).where(_Evaluation.model_node_id == uuid_id)).scalars().all())
        for ev in evals:
            db.delete(ev)
        if evals:
            db.flush()

        delete_model_node(db, uuid_id)
    except Exception as exc:
        # Translate FK violations into 400 with clear message
        err_msg = str(exc)
        if "ForeignKeyViolation" in err_msg or "violates foreign key" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete model: still referenced by other records (e.g. training tasks or releases).",
            ) from exc
        raise
    log_operation(
        db,
        operation_type="model.delete",
        resource_type="model_node",
        resource_id=uuid_id,
        status="success",
        summary_json={"model_id": str(uuid_id)},
    )
    return {"deleted": True}


@router.post("/{model_id}/infer")
async def infer_with_model(
    model_id: str,
    file: UploadFile,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    # Try UUID first, then code lookup
    try:
        uuid_id = UUID(model_id)
        model = get_model_node(db, uuid_id)
    except ValueError:
        from sqlalchemy import select as _select
        from backend.app.models import ModelNode as _ModelNode
        model = db.execute(
            _select(_ModelNode).where(_ModelNode.code == model_id)
        ).scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    import time
    start = time.time()

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image")

    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_width, image_height = img.size
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image: {exc}")

    try:
        from ultralytics import YOLO
        import base64

        yolo = YOLO(model.artifact_path)
        results = yolo(img, verbose=False)

        detections = []
        if results and len(results) > 0:
            r = results[0]
            class_names = dict(r.names) if hasattr(r, "names") and r.names else {}
            if hasattr(r, "boxes") and r.boxes is not None:
                for i in range(len(r.boxes.xyxy)):
                    box = r.boxes.xyxy[i].tolist()
                    conf = float(r.boxes.conf[i])
                    cls_id = int(r.boxes.cls[i])
                    detections.append({
                        "class_name": class_names.get(cls_id, str(cls_id)),
                        "confidence": round(conf, 4),
                        "bbox": [round(v, 2) for v in box],
                        "class_id": cls_id,
                    })

        # Generate overlay image with detection boxes
        overlay_base64 = None
        try:
            import cv2
            import numpy as np

            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                label = f"{det['class_name']} {det['confidence']*100:.1f}%"
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 102, 173), 2)
                cv2.putText(img_cv, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 102, 173), 1)
            _, buffer = cv2.imencode(".jpg", img_cv, [cv2.IMWRITE_JPEG_QUALITY, 85])
            overlay_base64 = base64.b64encode(buffer).decode("utf-8")
        except Exception:
            pass

        latency = round((time.time() - start) * 1000, 1)
        return {
            "model_code": model.code,
            "model_name": model.name,
            "detections": detections,
            "latency_ms": latency,
            "image_width": image_width,
            "image_height": image_height,
            "overlay_image": overlay_base64,
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {exc}")
