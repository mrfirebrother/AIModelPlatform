from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, verify_api_key
from backend.app.models import OperationLog

router = APIRouter(prefix="/api/operations", tags=["operations"])


class OperationLogEntry(BaseModel):
    id: str
    operationType: str
    actor: str | None
    requestId: str | None
    resourceType: str | None
    resourceId: str | None
    status: str
    summaryJson: dict[str, Any]
    errorSummary: str | None
    createdAt: str


class OperationLogsResponse(BaseModel):
    items: list[OperationLogEntry]
    total: int


class SystemCheck(BaseModel):
    postgres: bool
    redis: bool
    gpu: bool
    worker: bool


class OperationStatusResponse(BaseModel):
    status: str
    checks: SystemCheck


def _log_to_entry(log: OperationLog) -> OperationLogEntry:
    return OperationLogEntry(
        id=str(log.id),
        operationType=log.operation_type,
        actor=log.actor,
        requestId=str(log.request_id) if log.request_id else None,
        resourceType=log.resource_type,
        resourceId=str(log.resource_id) if log.resource_id else None,
        status=log.status,
        summaryJson=log.summary_json or {},
        errorSummary=log.error_summary,
        createdAt=log.created_at.isoformat(),
    )


@router.get("/logs", response_model=OperationLogsResponse)
def get_operation_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    start: datetime | None = None,
    end: datetime | None = None,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    query = select(OperationLog)
    count_query = select(func.count(OperationLog.id))

    if start is not None:
        query = query.where(OperationLog.created_at >= start)
        count_query = count_query.where(OperationLog.created_at >= start)
    if end is not None:
        query = query.where(OperationLog.created_at <= end)
        count_query = count_query.where(OperationLog.created_at <= end)

    total = db.execute(count_query).scalar_one()
    logs = list(
        db.execute(
            query.order_by(OperationLog.created_at.desc()).offset(skip).limit(limit)
        ).scalars().all()
    )

    return OperationLogsResponse(
        items=[_log_to_entry(l) for l in logs],
        total=total,
    )


@router.get("/errors", response_model=OperationLogsResponse)
def get_operation_errors(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    query = select(OperationLog).where(OperationLog.status == "error")
    count_query = select(func.count(OperationLog.id)).where(
        OperationLog.status == "error"
    )

    total = db.execute(count_query).scalar_one()
    logs = list(
        db.execute(
            query.order_by(OperationLog.created_at.desc()).offset(skip).limit(limit)
        ).scalars().all()
    )

    return OperationLogsResponse(
        items=[_log_to_entry(l) for l in logs],
        total=total,
    )


@router.get("/status", response_model=OperationStatusResponse)
def get_operation_status(
    _key: str = Depends(verify_api_key),
) -> Any:
    from backend.app.main import _run_check
    import asyncio

    checks_config: dict[str, Any] = {}
    for name in ("postgres", "redis", "gpu", "worker"):
        checks_config[name] = True

    results = {}
    for name, val in checks_config.items():
        results[name] = val

    if not results.get("postgres", False) or not results.get("redis", False):
        status = "unhealthy"
    elif not all(results.get(name, False) for name in ("gpu", "worker")):
        status = "degraded"
    else:
        status = "ready"

    return OperationStatusResponse(
        status=status,
        checks=SystemCheck(**results),
    )
