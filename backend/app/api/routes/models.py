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
    models = list_root_models(db)
    return ModelNodeListResponse(models=models, total=len(models))


@router.get("/{model_id}", response_model=ModelNodeResponse)
def get_model(
    model_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    model = get_model_node(db, model_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    return model


@router.patch("/{model_id}/status", response_model=ModelNodeResponse)
def update_model_status(
    model_id: UUID,
    payload: ModelNodeStatusUpdate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    model = get_model_node(db, model_id)
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
    model_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    model = get_model_node(db, model_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    children = list_child_models(db, model_id)
    if children:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete model with children. Delete child models first.",
        )

    delete_model_node(db, model_id)
    log_operation(
        db,
        operation_type="model.delete",
        resource_type="model_node",
        resource_id=model_id,
        status="success",
        summary_json={"model_id": str(model_id)},
    )
    return {"deleted": True}
