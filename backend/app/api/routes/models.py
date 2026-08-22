from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, verify_api_key
from backend.app.observability.operation_log import log_operation
from backend.app.repositories.model_repository import (
    create_model_node,
    get_model_node,
    list_root_models,
)
from backend.app.schemas.model import (
    ModelNodeCreate,
    ModelNodeListResponse,
    ModelNodeResponse,
    ModelNodeStatusUpdate,
)

router = APIRouter(prefix="/api/models", tags=["models"])


@router.post("", response_model=ModelNodeResponse, status_code=status.HTTP_201_CREATED)
def create_model(
    payload: ModelNodeCreate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        model = create_model_node(
            db,
            task_type=payload.task_type,
            model_family=payload.model_family,
            artifact_path=payload.artifact_path,
            artifact_hash=payload.artifact_hash,
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
