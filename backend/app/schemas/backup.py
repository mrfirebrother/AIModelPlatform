from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class BackupCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    backup_type: str = Field(default="full", pattern="^(full|model|dataset|metadata)$")
    scope_json: dict[str, Any] = Field(default_factory=dict)
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class BackupResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    backup_type: str
    scope_json: dict[str, Any]
    artifact_path: str | None
    artifact_hash: str | None
    metadata_json: dict[str, Any]
    status: str
    verified: bool
    verified_at: datetime | None
    verify_hash: str | None
    restore_target_id: UUID | None
    error_summary: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class BackupListResponse(BaseModel):
    backups: list[BackupResponse]
    total: int


class BackupVerifyResponse(BaseModel):
    id: UUID
    verified: bool
    verify_hash: str
    verified_at: datetime


class BackupRestoreRequest(BaseModel):
    target_backup_id: UUID | None = None
    restore_type: str = Field(default="full", pattern="^(full|model|dataset|metadata)$")
    force: bool = False


class BackupRestoreResponse(BaseModel):
    id: UUID
    status: str
    restore_target_id: UUID | None
    message: str
