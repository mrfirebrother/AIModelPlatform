from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, verify_api_key
from backend.app.schemas.model import ModelNodeResponse
from backend.app.services.model_diff import (
    DiffReport,
    diff_models,
    get_model_history,
)

router = APIRouter(prefix="/api/models", tags=["model-diff"])


@router.get("/{model_id}/diff")
def compare_models(
    model_id: UUID,
    compare: UUID = Query(..., description="ID of the model to compare against"),
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        report: DiffReport = diff_models(db, model_id, compare)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        )
    return {
        "base_model_id": str(report.base_model_id),
        "compare_model_id": str(report.compare_model_id),
        "is_identical": report.is_identical,
        "config_diffs": [
            {"field": d.field, "old_value": d.old_value, "new_value": d.new_value, "change": d.change}
            for d in report.config_diffs
        ],
        "metrics_diffs": [
            {"field": d.field, "old_value": d.old_value, "new_value": d.new_value, "change": d.change}
            for d in report.metrics_diffs
        ],
        "label_diffs": [
            {"field": d.field, "old_value": d.old_value, "new_value": d.new_value, "change": d.change}
            for d in report.label_diffs
        ],
        "summary": report.summary,
    }


@router.get("/{model_id}/history")
def model_history(
    model_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        chain = get_model_history(db, model_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        )
    return {
        "model_id": str(model_id),
        "history": [
            {
                "id": str(m.id),
                "task_type": m.task_type,
                "model_family": m.model_family,
                "status": m.status,
                "framework": m.framework,
                "artifact_format": m.artifact_format,
                "parent_id": str(m.parent_id) if m.parent_id else None,
            }
            for m in chain
        ],
        "depth": len(chain),
    }
