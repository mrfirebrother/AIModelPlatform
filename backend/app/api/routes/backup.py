from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, verify_api_key
from backend.app.observability.operation_log import log_operation
from backend.app.schemas.backup import (
    BackupCreate,
    BackupListResponse,
    BackupResponse,
    BackupRestoreRequest,
    BackupRestoreResponse,
    BackupVerifyResponse,
)
from backend.app.services.backup_service import BackupService

router = APIRouter(prefix="/api/backup", tags=["backup"])


def get_backup_service() -> BackupService:
    return BackupService()


@router.post("", response_model=BackupResponse, status_code=status.HTTP_201_CREATED)
def create_backup(
    payload: BackupCreate,
    db: Session = Depends(get_db),
    svc: BackupService = Depends(get_backup_service),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        record = svc.create_backup(
            db,
            name=payload.name,
            description=payload.description,
            backup_type=payload.backup_type,
            scope_json=payload.scope_json,
            metadata_json=payload.metadata_json,
        )
        log_operation(
            db,
            operation_type="backup.create",
            resource_type="backup_record",
            resource_id=record.id,
            status="success",
            summary_json={"name": record.name, "backup_type": record.backup_type},
        )
        return BackupResponse.model_validate(record)
    except ValueError as exc:
        log_operation(
            db,
            operation_type="backup.create",
            status="error",
            error_summary=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@router.get("/list", response_model=BackupListResponse)
def list_backups(
    backup_type: str | None = None,
    db: Session = Depends(get_db),
    svc: BackupService = Depends(get_backup_service),
    _key: str = Depends(verify_api_key),
) -> Any:
    records = svc.list_backups(db, backup_type=backup_type)
    return BackupListResponse(
        backups=[BackupResponse.model_validate(r) for r in records],
        total=len(records),
    )


@router.post("/{backup_id}/verify", response_model=BackupVerifyResponse)
def verify_backup(
    backup_id: UUID,
    db: Session = Depends(get_db),
    svc: BackupService = Depends(get_backup_service),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        verified, verify_hash, verified_at = svc.verify_backup(db, backup_id)
        log_operation(
            db,
            operation_type="backup.verify",
            resource_type="backup_record",
            resource_id=backup_id,
            status="success",
            summary_json={"verified": verified, "verify_hash": verify_hash},
        )
        return BackupVerifyResponse(
            id=backup_id,
            verified=verified,
            verify_hash=verify_hash,
            verified_at=verified_at,
        )
    except ValueError as exc:
        log_operation(
            db,
            operation_type="backup.verify",
            status="error",
            error_summary=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@router.post("/restore", response_model=BackupRestoreResponse)
def restore_backup(
    payload: BackupRestoreRequest,
    db: Session = Depends(get_db),
    svc: BackupService = Depends(get_backup_service),
    _key: str = Depends(verify_api_key),
) -> Any:
    backup_id = payload.target_backup_id
    if backup_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="target_backup_id is required",
        )
    try:
        record = svc.restore_backup(
            db,
            backup_id,
            restore_type=payload.restore_type,
            force=payload.force,
        )
        log_operation(
            db,
            operation_type="backup.restore",
            resource_type="backup_record",
            resource_id=record.id,
            status="success",
            summary_json={
                "source_backup_id": str(backup_id),
                "restore_type": payload.restore_type,
            },
        )
        return BackupRestoreResponse(
            id=record.id,
            status=record.status,
            restore_target_id=record.restore_target_id,
            message="Restore completed successfully",
        )
    except ValueError as exc:
        log_operation(
            db,
            operation_type="backup.restore",
            status="error",
            error_summary=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@router.delete("/{backup_id}", response_model=BackupResponse)
def delete_backup(
    backup_id: UUID,
    db: Session = Depends(get_db),
    svc: BackupService = Depends(get_backup_service),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        record = svc.delete_backup(db, backup_id)
        log_operation(
            db,
            operation_type="backup.delete",
            resource_type="backup_record",
            resource_id=backup_id,
            status="success",
            summary_json={"name": record.name},
        )
        return BackupResponse.model_validate(record)
    except ValueError as exc:
        log_operation(
            db,
            operation_type="backup.delete",
            status="error",
            error_summary=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
