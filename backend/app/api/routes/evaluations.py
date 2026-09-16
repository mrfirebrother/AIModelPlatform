from __future__ import annotations

import base64
import uuid as _uuid
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from backend.app.schemas import CAMEL_CONFIG
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, get_effective_settings, verify_api_key
from backend.app.evaluation.policy import EvaluationPolicy
from backend.app.observability.operation_log import log_operation
from backend.app.schemas.evaluation_session import (
    EvaluationSessionCreate,
    EvaluationSessionResponse,
)
from backend.app.services import evaluation_service

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


class EvaluationCreate(BaseModel):
    model_config = CAMEL_CONFIG
    model_node_id: UUID
    dataset_snapshot_id: UUID
    policy: dict[str, Any]


class EvaluationResponse(BaseModel):
    id: UUID
    model_node_id: UUID
    dataset_snapshot_id: UUID
    # 评测流水线状态（pending/running/passed/failed），由 worker 与调度器使用。
    # 注意：这里没有人工复核相关字段 —— 平台不存在审核机制，评测只提供指标。
    auto_status: str
    evaluation_policy_json: dict
    auto_metrics_json: dict
    model_name: str
    dataset_name: str
    metrics: dict[str, Any] | None = None
    test_image_count: int = 0
    created_at: str
    last_error: str | None = None
    model_config = {"from_attributes": True}


class EvaluationListResponse(BaseModel):
    evaluations: list[EvaluationResponse]
    total: int


def _to_evaluation_response(ev: Any) -> EvaluationResponse:
    metrics = ev.auto_metrics_json or {}
    return EvaluationResponse(
        id=ev.id,
        model_node_id=ev.model_node_id,
        dataset_snapshot_id=ev.dataset_snapshot_id,
        auto_status=ev.auto_status,
        evaluation_policy_json=ev.evaluation_policy_json or {},
        auto_metrics_json=metrics,
        model_name=(ev.model_node.name or ev.model_node.model_family),
        dataset_name=ev.dataset_snapshot.dataset.name,
        metrics=metrics or None,
        test_image_count=int(metrics.get("num_images", 0)),
        created_at=ev.created_at.isoformat(),
        last_error=ev.last_error,
    )


@router.post("", response_model=EvaluationResponse, status_code=status.HTTP_201_CREATED)
def submit_evaluation(
    payload: EvaluationCreate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        policy = EvaluationPolicy.from_dict(payload.policy)
        ev = evaluation_service.create_evaluation(
            db,
            model_node_id=payload.model_node_id,
            dataset_snapshot_id=payload.dataset_snapshot_id,
            policy=policy,
        )
        log_operation(
            db,
            operation_type="evaluation.create",
            resource_type="evaluation",
            resource_id=ev.id,
            status="success",
            summary_json={"evaluation_id": str(ev.id)},
        )
        return _to_evaluation_response(ev)
    except ValueError as exc:
        log_operation(
            db,
            operation_type="evaluation.create",
            status="error",
            error_summary=str(exc),
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        log_operation(
            db,
            operation_type="evaluation.create",
            status="error",
            error_summary=str(exc),
        )
        raise


@router.get("", response_model=EvaluationListResponse)
def list_evaluations(
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    from sqlalchemy import select
    from backend.app.models import Evaluation

    evaluations = list(db.execute(select(Evaluation)).scalars().all())
    return EvaluationListResponse(
        evaluations=[_to_evaluation_response(e) for e in evaluations],
        total=len(evaluations),
    )


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
def get_evaluation(
    evaluation_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    ev = evaluation_service.get_evaluation(db, evaluation_id)
    if ev is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation not found")
    return _to_evaluation_response(ev)


class InferResult(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float]
    class_id: int


class InferResponse(BaseModel):
    detections: list[InferResult]
    latency_ms: float
    image_width: int
    image_height: int


@router.post("/{evaluation_id}/infer", response_model=InferResponse)
async def infer_with_model(
    evaluation_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    ev = evaluation_service.get_evaluation(db, evaluation_id)
    if ev is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation not found")

    model_node = ev.model_node
    if model_node is None or not model_node.artifact_path:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Model artifact not found")

    import time
    start = time.time()

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image")

    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_width, image_height = img.size
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image: {exc}")

    try:
        from ultralytics import YOLO
        model = YOLO(model_node.artifact_path)
        results = model(img, verbose=False)

        detections = []
        if results and len(results) > 0:
            r = results[0]
            class_names = dict(r.names) if hasattr(r, "names") and r.names else {}
            if hasattr(r, "boxes") and r.boxes is not None:
                for i in range(len(r.boxes.xyxy)):
                    box = r.boxes.xyxy[i].tolist()
                    conf = float(r.boxes.conf[i])
                    cls_id = int(r.boxes.cls[i])
                    det = InferResult(
                        class_name=class_names.get(cls_id, str(cls_id)),
                        confidence=round(conf, 4),
                        bbox=[round(v, 2) for v in box],
                        class_id=cls_id,
                    )
                    detections.append(det)

        latency = round((time.time() - start) * 1000, 1)
        return InferResponse(
            detections=detections,
            latency_ms=latency,
            image_width=image_width,
            image_height=image_height,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {exc}")


@router.post("/sessions", response_model=EvaluationSessionResponse, status_code=status.HTTP_201_CREATED)
def create_evaluation_session(
    payload: EvaluationSessionCreate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    session_id = str(_uuid.uuid4())
    log_operation(
        db,
        operation_type="evaluation.session.create",
        resource_type="evaluation_session",
        status="success",
        summary_json={"session_id": session_id},
    )
    return EvaluationSessionResponse(
        session_id=session_id,
        model_node_id=payload.model_node_id,
        expires_in_seconds=payload.expires_in_seconds,
    )
