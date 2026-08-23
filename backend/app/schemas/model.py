from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


def _to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


class ModelNodeCreate(BaseModel):
    model_config = ConfigDict(alias_generator=_to_camel, populate_by_name=True)

    task_type: str = Field(..., max_length=64)
    model_family: str = Field(..., max_length=128)
    artifact_path: str = Field(..., max_length=1024)
    artifact_hash: str = Field(default="", max_length=255)
    name: str = Field(default="", max_length=255)
    parent_id: UUID | None = None
    label_schema_id: UUID | None = None
    dataset_snapshot_id: UUID | None = None
    training_attempt_id: UUID | None = None
    framework: str = Field(default="pytorch", max_length=64)
    artifact_format: str = Field(default="pt", max_length=32)
    status: str = Field(default="candidate", max_length=32)
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class ModelNodeResponse(BaseModel):
    id: UUID
    name: str = ""
    parent_id: UUID | None
    label_schema_id: UUID | None
    dataset_snapshot_id: UUID | None
    training_attempt_id: UUID | None
    task_type: str
    model_family: str
    artifact_path: str
    artifact_hash: str
    framework: str
    artifact_format: str
    status: str
    metadata_json: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelNodeStatusUpdate(BaseModel):
    status: str = Field(..., max_length=32)


class ModelNodeListResponse(BaseModel):
    models: list[ModelNodeResponse]
    total: int
