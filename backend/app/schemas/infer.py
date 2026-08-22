from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    binding_id: UUID
    image_base64: str = Field(..., description="Base64-encoded image data")
    image_format: str = Field(default="png", description="Image format (png, jpg)")


class InferModelInfo(BaseModel):
    id: UUID
    task_type: str
    model_family: str
    artifact_hash: str


class InferReleaseInfo(BaseModel):
    id: UUID
    revision_no: int
    inference_config_hash: str


class InferResponse(BaseModel):
    modelBindingId: UUID
    release: InferReleaseInfo
    model: InferModelInfo
    generation: int
    configHash: str
    detections: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"alias_generator": lambda v: v}
