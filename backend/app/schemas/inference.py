from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class InferenceInput(BaseModel):
    image_base64: str = Field(..., alias="image_base64", description="Base64-encoded image data")
    image_format: str = Field(default="png", alias="image_format", description="Image format (png, jpg, jpeg, bmp, webp)")


class InferenceItem(BaseModel):
    modelBindingId: UUID | None = Field(default=None)
    modelNodeId: UUID | None = Field(default=None)
    input: dict[str, Any] = Field(...)

    @model_validator(mode="after")
    def validate_target(self) -> InferenceItem:
        if self.modelBindingId is None and self.modelNodeId is None:
            raise ValueError("Either modelBindingId or modelNodeId must be provided")
        if self.modelBindingId is not None and self.modelNodeId is not None:
            raise ValueError("Only one of modelBindingId or modelNodeId may be provided")
        if "image_base64" not in self.input:
            raise ValueError("input must contain image_base64")
        return self


class InferenceRequest(BaseModel):
    modelBindingId: UUID | None = Field(default=None)
    modelNodeId: UUID | None = Field(default=None)
    input: dict[str, Any] = Field(...)

    @model_validator(mode="after")
    def validate_target(self) -> InferenceRequest:
        if self.modelBindingId is None and self.modelNodeId is None:
            raise ValueError("Either modelBindingId or modelNodeId must be provided")
        if self.modelBindingId is not None and self.modelNodeId is not None:
            raise ValueError("Only one of modelBindingId or modelNodeId may be provided")
        if "image_base64" not in self.input:
            raise ValueError("input must contain image_base64")
        return self


class InferenceBatchRequest(BaseModel):
    items: list[InferenceItem] = Field(..., min_length=1, max_length=32)


class InferenceDetection(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float]
    class_id: int


class InferenceResult(BaseModel):
    detections: list[InferenceDetection] = Field(default_factory=list)
    modelBindingId: UUID | None = None
    modelNodeId: UUID | None = None
    releaseId: UUID | None = None
    generation: int | None = None


class InferenceResponse(BaseModel):
    results: list[InferenceResult] = Field(default_factory=list)
    latency: float = Field(description="Total latency in milliseconds")
    modelNodeId: UUID | None = None
    bindingId: UUID | None = None
    releaseId: UUID | None = None
    generation: int | None = None


class InferenceBatchResponse(BaseModel):
    results: list[InferenceResult] = Field(default_factory=list)
    total_latency: float = Field(description="Total batch latency in milliseconds")
    succeeded: int = 0
    failed: int = 0


class ModelInfoResponse(BaseModel):
    id: UUID
    task_type: str
    model_family: str
    artifact_hash: str
    framework: str
    status: str
