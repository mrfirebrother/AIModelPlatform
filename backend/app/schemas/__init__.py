from __future__ import annotations

from pydantic import ConfigDict


def _to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


CAMEL_CONFIG = ConfigDict(alias_generator=_to_camel, populate_by_name=True)


from .backup import (
    BackupCreate,
    BackupListResponse,
    BackupResponse,
    BackupRestoreRequest,
    BackupRestoreResponse,
    BackupVerifyResponse,
)
from .binding import (
    BindingCreate,
    BindingListResponse,
    BindingResponse,
    ReleaseCreate,
    ReleaseResponse,
    RollbackCreate,
)
from .inference import (
    InferenceBatchRequest,
    InferenceBatchResponse,
    InferenceRequest,
    InferenceResponse,
    ModelInfoResponse,
)
from .model import ModelNodeCreate, ModelNodeListResponse, ModelNodeResponse, ModelNodeStatusUpdate

__all__ = [
    "BackupCreate",
    "BackupListResponse",
    "BackupResponse",
    "BackupRestoreRequest",
    "BackupRestoreResponse",
    "BackupVerifyResponse",
    "BindingCreate",
    "BindingListResponse",
    "BindingResponse",
    "InferenceBatchRequest",
    "InferenceBatchResponse",
    "InferenceRequest",
    "InferenceResponse",
    "ModelInfoResponse",
    "ModelNodeCreate",
    "ModelNodeListResponse",
    "ModelNodeResponse",
    "ModelNodeStatusUpdate",
    "ReleaseCreate",
    "ReleaseResponse",
    "RollbackCreate",
]
