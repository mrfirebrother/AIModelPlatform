from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class BindingCreate(BaseModel):
    external_ref: str | None = None


class BindingResponse(BaseModel):
    id: UUID
    external_ref: str | None
    current_release_id: UUID | None
    current_runtime_instance_id: UUID | None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ReleaseCreate(BaseModel):
    binding_id: UUID
    model_node_id: UUID
    inference_config_json: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class ReleaseResponse(BaseModel):
    id: UUID
    binding_id: UUID
    revision_no: int
    model_node_id: UUID
    inference_config_json: dict[str, Any]
    inference_config_hash: str
    release_type: str
    rollback_target_release_id: UUID | None
    status: str
    reason: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class RollbackCreate(BaseModel):
    binding_id: UUID
    target_release_id: UUID
    model_node_id: UUID
    reason: str | None = None


class BindingListResponse(BaseModel):
    bindings: list[BindingResponse]
    total: int
