from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, verify_api_key
from backend.app.models import BindingRelease
from backend.app.observability.operation_log import log_operation
from backend.app.services.release_service import ReleaseService

router = APIRouter(prefix="/api/releases", tags=["releases"])

release_service = ReleaseService()


class ReleaseListResponse(BaseModel):
    releases: list[dict[str, Any]]
    total: int


@router.get("", response_model=ReleaseListResponse)
def list_releases(
    binding_id: UUID | None = None,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    stmt = select(BindingRelease)
    if binding_id is not None:
        stmt = stmt.where(BindingRelease.binding_id == binding_id)
    releases = list(db.execute(stmt).scalars().all())
    return ReleaseListResponse(
        releases=[_release_to_dict(r) for r in releases],
        total=len(releases),
    )


@router.get("/{release_id}")
def get_release(
    release_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    release = db.get(BindingRelease, release_id)
    if release is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Release not found")
    return _release_to_dict(release)


@router.post("/{release_id}/publish")
def publish_release(
    release_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        release = release_service.publish_release(db, release_id)
        log_operation(
            db,
            operation_type="release.publish",
            resource_type="binding_release",
            resource_id=release.id,
            status="success",
            summary_json={"release_id": str(release.id)},
        )
        return _release_to_dict(release)
    except ValueError as exc:
        log_operation(
            db,
            operation_type="release.publish",
            status="error",
            error_summary=str(exc),
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


def _release_to_dict(release: BindingRelease) -> dict[str, Any]:
    return {
        "id": str(release.id),
        "binding_id": str(release.binding_id),
        "revision_no": release.revision_no,
        "model_node_id": str(release.model_node_id),
        "inference_config_json": release.inference_config_json,
        "inference_config_hash": release.inference_config_hash,
        "release_type": release.release_type,
        "rollback_target_release_id": (
            str(release.rollback_target_release_id)
            if release.rollback_target_release_id
            else None
        ),
        "status": release.status,
        "reason": release.reason,
        "created_at": release.created_at.isoformat() if release.created_at else None,
    }
