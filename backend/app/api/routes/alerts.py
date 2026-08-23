from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, verify_api_key
from backend.app.models import Alert
from backend.app.observability.alerting import acknowledge_alert, get_alert_stats

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


class AlertEntry(BaseModel):
    id: str
    alertType: str
    severity: str
    title: str
    message: str
    resourceType: str | None
    resourceId: str | None
    status: str
    metadataJson: dict[str, Any]
    acknowledgedAt: str | None
    acknowledgedBy: str | None
    createdAt: str
    updatedAt: str


class AlertsResponse(BaseModel):
    items: list[AlertEntry]
    total: int


class AlertStatsResponse(BaseModel):
    total: int
    active: int
    acknowledged: int
    bySeverity: dict[str, int]
    byType: dict[str, int]


class AcknowledgeRequest(BaseModel):
    acknowledgedBy: str = "user"


class AcknowledgeResponse(BaseModel):
    id: str
    status: str
    acknowledgedAt: str
    acknowledgedBy: str


def _alert_to_entry(alert: Alert) -> AlertEntry:
    return AlertEntry(
        id=str(alert.id),
        alertType=alert.alert_type,
        severity=alert.severity,
        title=alert.title,
        message=alert.message,
        resourceType=alert.resource_type,
        resourceId=alert.resource_id,
        status=alert.status,
        metadataJson=alert.metadata_json or {},
        acknowledgedAt=alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
        acknowledgedBy=alert.acknowledged_by,
        createdAt=alert.created_at.isoformat(),
        updatedAt=alert.updated_at.isoformat(),
    )


@router.get("", response_model=AlertsResponse)
def list_alerts(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    status: str | None = None,
    alert_type: str | None = Query(None, alias="type"),
    severity: str | None = None,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    query = select(Alert)
    count_query = select(func.count(Alert.id))

    if status is not None:
        query = query.where(Alert.status == status)
        count_query = count_query.where(Alert.status == status)
    if alert_type is not None:
        query = query.where(Alert.alert_type == alert_type)
        count_query = count_query.where(Alert.alert_type == alert_type)
    if severity is not None:
        query = query.where(Alert.severity == severity)
        count_query = count_query.where(Alert.severity == severity)

    total = db.execute(count_query).scalar_one()
    alerts = list(
        db.execute(
            query.order_by(Alert.created_at.desc()).offset(skip).limit(limit)
        ).scalars().all()
    )

    return AlertsResponse(
        items=[_alert_to_entry(a) for a in alerts],
        total=total,
    )


@router.post("/{alert_id}/acknowledge", response_model=AcknowledgeResponse)
def acknowledge_alert_endpoint(
    alert_id: UUID,
    body: AcknowledgeRequest | None = None,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    acknowledged_by = body.acknowledgedBy if body else "user"
    alert = acknowledge_alert(db, alert_id, acknowledged_by=acknowledged_by)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found or already acknowledged")
    return AcknowledgeResponse(
        id=str(alert.id),
        status=alert.status,
        acknowledgedAt=alert.acknowledged_at.isoformat(),
        acknowledgedBy=alert.acknowledged_by,
    )


@router.get("/stats", response_model=AlertStatsResponse)
def alert_stats(
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    stats = get_alert_stats(db)
    return AlertStatsResponse(
        total=stats["total"],
        active=stats["active"],
        acknowledged=stats["acknowledged"],
        bySeverity=stats["by_severity"],
        byType=stats["by_type"],
    )
