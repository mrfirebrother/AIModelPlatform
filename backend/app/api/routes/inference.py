from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, get_effective_settings, verify_api_key
from backend.app.config import Settings
from backend.app.schemas.inference import (
    InferenceBatchRequest,
    InferenceBatchResponse,
    InferenceRequest,
    InferenceResponse,
    ModelInfoResponse,
)
from backend.app.services.inference_service import InferenceService

logger = logging.getLogger("platform.inference")

router = APIRouter(prefix="/api/v1", tags=["external-inference"])


def _get_inference_service(db: Session = Depends(get_db)) -> InferenceService:
    return InferenceService(db)


@router.post("/infer", response_model=InferenceResponse)
def infer_single(
    payload: InferenceRequest,
    service: InferenceService = Depends(_get_inference_service),
    settings: Settings = Depends(get_effective_settings),
    _key: str = Depends(verify_api_key),
) -> Any:
    start_time = time.time()

    image_base64 = payload.input.get("image_base64", "")
    image_format = payload.input.get("image_format", "png")

    try:
        if payload.modelBindingId is not None:
            resp = service.run_inference(
                binding_id=payload.modelBindingId,
                image_base64=image_base64,
                image_format=image_format,
            )
        else:
            binding = service.resolve_binding(model_node_id=payload.modelNodeId)
            if binding is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No binding found for modelNodeId",
                )
            resp = service.run_inference(
                binding_id=binding.id,
                image_base64=image_base64,
                image_format=image_format,
            )
        return resp
    except ValueError as exc:
        msg = str(exc)
        if "not found" in msg.lower() or "Binding not found" in msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)
        if "not serving" in msg.lower() or "No serving" in msg:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=msg)
        if "Invalid" in msg or "Unsupported" in msg or "Empty" in msg or "exceeds" in msg:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)


@router.post("/infer/batch", response_model=InferenceBatchResponse)
def infer_batch(
    payload: InferenceBatchRequest,
    service: InferenceService = Depends(_get_inference_service),
    settings: Settings = Depends(get_effective_settings),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        resp = service.run_batch_inference(items=payload.items)
        return resp
    except Exception as exc:
        logger.error("Batch inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.get("/infer/models", response_model=list[ModelInfoResponse])
def list_models(
    service: InferenceService = Depends(_get_inference_service),
    _key: str = Depends(verify_api_key),
) -> Any:
    return service.list_available_models()
