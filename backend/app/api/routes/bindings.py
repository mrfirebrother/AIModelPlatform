from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, verify_api_key
from backend.app.observability.operation_log import log_operation
from backend.app.repositories.binding_repository import (
    create_binding,
    create_release,
    get_binding,
    list_bindings,
)
from backend.app.schemas.binding import (
    BindingCreate,
    BindingListResponse,
    BindingResponse,
    ReleaseCreate,
    ReleaseResponse,
    RollbackCreate,
)
from backend.app.services.binding_service import activate_release, create_rollback_release

router = APIRouter(prefix="/api", tags=["bindings"])


@router.post("/bindings", response_model=BindingResponse, status_code=status.HTTP_201_CREATED)
def create_new_binding(
    payload: BindingCreate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        binding = create_binding(db, external_ref=payload.external_ref)
        log_operation(
            db,
            operation_type="binding.create",
            resource_type="model_binding",
            resource_id=binding.id,
            status="success",
            summary_json={"binding_id": str(binding.id)},
        )
        return binding
    except Exception as exc:
        log_operation(
            db,
            operation_type="binding.create",
            status="error",
            error_summary=str(exc),
        )
        raise


@router.get("/bindings", response_model=BindingListResponse)
def list_all_bindings(
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    bindings = list_bindings(db)
    return BindingListResponse(bindings=bindings, total=len(bindings))


@router.get("/bindings/{binding_id}", response_model=BindingResponse)
def get_binding_by_id(
    binding_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    binding = get_binding(db, binding_id)
    if binding is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Binding not found")
    return binding


@router.post("/releases", response_model=ReleaseResponse, status_code=status.HTTP_201_CREATED)
def create_new_release(
    payload: ReleaseCreate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        release = create_release(
            db,
            binding_id=payload.binding_id,
            model_node_id=payload.model_node_id,
            inference_config_json=payload.inference_config_json,
            reason=payload.reason,
        )
        log_operation(
            db,
            operation_type="release.create",
            resource_type="binding_release",
            resource_id=release.id,
            status="success",
            summary_json={
                "release_id": str(release.id),
                "binding_id": str(payload.binding_id),
            },
        )
        return release
    except Exception as exc:
        log_operation(
            db,
            operation_type="release.create",
            status="error",
            error_summary=str(exc),
        )
        raise


@router.post("/releases/{release_id}/activate", response_model=ReleaseResponse)
def activate(
    release_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        release = activate_release(db, release_id)
        log_operation(
            db,
            operation_type="release.activate",
            resource_type="binding_release",
            resource_id=release.id,
            status="success",
            summary_json={"release_id": str(release.id)},
        )
        return release
    except ValueError as exc:
        log_operation(
            db,
            operation_type="release.activate",
            status="error",
            error_summary=str(exc),
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        log_operation(
            db,
            operation_type="release.activate",
            status="error",
            error_summary=str(exc),
        )
        raise


@router.post("/releases/rollback", response_model=ReleaseResponse, status_code=status.HTTP_201_CREATED)
def rollback(
    payload: RollbackCreate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        release = create_rollback_release(
            db,
            binding_id=payload.binding_id,
            target_release_id=payload.target_release_id,
            model_node_id=payload.model_node_id,
            reason=payload.reason,
        )
        log_operation(
            db,
            operation_type="release.rollback",
            resource_type="binding_release",
            resource_id=release.id,
            status="success",
            summary_json={
                "release_id": str(release.id),
                "target_release_id": str(payload.target_release_id),
            },
        )
        return release
    except ValueError as exc:
        log_operation(
            db,
            operation_type="release.rollback",
            status="error",
            error_summary=str(exc),
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        log_operation(
            db,
            operation_type="release.rollback",
            status="error",
            error_summary=str(exc),
        )
        raise
