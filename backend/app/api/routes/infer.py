from __future__ import annotations

import base64
import time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, verify_api_key
from backend.app.models import ModelBinding, RuntimeInstance
from backend.app.observability.inference_log import log_inference_summary
from backend.app.observability.operation_log import log_operation
from backend.app.schemas.infer import InferRequest, InferResponse

router = APIRouter(prefix="/api", tags=["infer"])

ALLOWED_IMAGE_FORMATS = {"png", "jpg", "jpeg", "bmp", "webp"}


def _validate_image_base64(image_base64: str, image_format: str) -> bytes:
    if image_format.lower() not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format: {image_format}",
        )
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image data",
        )
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image data",
        )
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image exceeds maximum size (10MB)",
        )
    return image_bytes


@router.post("/infer", response_model=InferResponse)
def infer(
    payload: InferRequest,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    start_time = time.time()
    binding = db.get(ModelBinding, payload.binding_id)
    if binding is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Binding not found")

    if binding.current_runtime_instance_id is None:
        log_inference_summary(
            binding_id=payload.binding_id,
            model_node_id=UUID(int=0),
            release_id=UUID(int=0),
            generation=0,
            config_hash="",
            image_format=payload.image_format,
            image_size_bytes=0,
            latency_ms=0,
            status="error",
            error="No serving instance",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No serving instance available for this binding",
        )

    instance = db.get(RuntimeInstance, binding.current_runtime_instance_id)
    if instance is None or instance.status != "serving":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Runtime instance is not serving",
        )

    image_bytes = _validate_image_base64(payload.image_base64, payload.image_format)

    from backend.app.models import BindingRelease, ModelNode

    release = db.get(BindingRelease, binding.current_release_id)
    model = db.get(ModelNode, instance.model_node_id) if instance.model_node_id else None

    latency_ms = (time.time() - start_time) * 1000

    log_inference_summary(
        binding_id=payload.binding_id,
        model_node_id=instance.model_node_id,
        release_id=instance.release_id,
        generation=instance.generation,
        config_hash=instance.config_hash,
        image_format=payload.image_format,
        image_size_bytes=len(image_bytes),
        latency_ms=latency_ms,
        status="success",
    )

    log_operation(
        db,
        operation_type="infer",
        resource_type="model_binding",
        resource_id=payload.binding_id,
        status="success",
        summary_json={
            "binding_id": str(payload.binding_id),
            "generation": instance.generation,
        },
    )

    return InferResponse(
        modelBindingId=payload.binding_id,
        release={
            "id": str(release.id) if release else str(instance.release_id),
            "revision_no": release.revision_no if release else 0,
            "inference_config_hash": (
                release.inference_config_hash if release else instance.config_hash
            ),
        },
        model={
            "id": str(model.id) if model else str(instance.model_node_id),
            "task_type": model.task_type if model else "unknown",
            "model_family": model.model_family if model else "unknown",
            "artifact_hash": model.artifact_hash if model else "",
        },
        generation=instance.generation,
        configHash=instance.config_hash,
        detections=[],
    )
