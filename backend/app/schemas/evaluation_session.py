from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class EvaluationSessionCreate(BaseModel):
    model_node_id: UUID
    expires_in_seconds: int = Field(default=3600, ge=60, le=86400)


class EvaluationSessionResponse(BaseModel):
    session_id: str
    model_node_id: UUID
    expires_in_seconds: int
