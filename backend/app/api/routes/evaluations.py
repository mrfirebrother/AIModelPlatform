from __future__ import annotations

import uuid as _uuid
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from backend.app.schemas import CAMEL_CONFIG
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, verify_api_key
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
    auto_status: str
    human_status: str
    evaluation_policy_json: dict
    auto_metrics_json: dict
    model_config = {"from_attributes": True}


class EvaluationListResponse(BaseModel):
    evaluations: list[EvaluationResponse]
    total: int


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
        return ev
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
        evaluations=[EvaluationResponse.model_validate(e) for e in evaluations],
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
    return ev


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
