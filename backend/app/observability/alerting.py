from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models.alert import Alert


@dataclasses.dataclass(frozen=True, slots=True)
class AlertRule:
    alert_type: str
    severity: str
    title_template: str
    message_template: str
    resource_type: str | None = None


GPU_LOW_MEMORY = AlertRule(
    alert_type="gpu_memory_low",
    severity="critical",
    title_template="GPU {device} memory low",
    message_template="GPU {device} has only {free_mb}MB free out of {total_mb}MB ({pct:.1f}% used)",
    resource_type="gpu",
)

TRAINING_FAILED = AlertRule(
    alert_type="training_failed",
    severity="critical",
    title_template="Training task {task_id} failed",
    message_template="Training attempt {attempt_id} failed: {error}",
    resource_type="training_task",
)

INFERENCE_ERROR_RATE = AlertRule(
    alert_type="inference_error_rate",
    severity="warning",
    title_template="High inference error rate",
    message_template="Inference error rate is {error_rate:.1f}% over {window_size} requests (threshold: {threshold:.1f}%)",
    resource_type="inference",
)

DISK_SPACE_LOW = AlertRule(
    alert_type="disk_space_low",
    severity="warning",
    title_template="Disk space low on {path}",
    message_template="Disk usage at {path} is {usage_pct:.1f}% ({free_gb:.1f}GB free)",
    resource_type="storage",
)

SERVICE_UNAVAILABLE = AlertRule(
    alert_type="service_unavailable",
    severity="critical",
    title_template="Service {service} unavailable",
    message_template="Health check for {service} failed: {details}",
    resource_type="service",
)

ALL_RULES: list[AlertRule] = [
    GPU_LOW_MEMORY,
    TRAINING_FAILED,
    INFERENCE_ERROR_RATE,
    DISK_SPACE_LOW,
    SERVICE_UNAVAILABLE,
]


def create_alert(
    session: Session,
    *,
    alert_type: str,
    severity: str,
    title: str,
    message: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Alert:
    alert = Alert(
        alert_type=alert_type,
        severity=severity,
        title=title,
        message=message,
        resource_type=resource_type,
        resource_id=resource_id,
        metadata_json=metadata or {},
    )
    session.add(alert)
    session.flush()
    return alert


def acknowledge_alert(
    session: Session,
    alert_id: UUID,
    *,
    acknowledged_by: str = "system",
) -> Alert | None:
    alert = session.get(Alert, alert_id)
    if alert is None or alert.status != "active":
        return None
    alert.status = "acknowledged"
    alert.acknowledged_at = datetime.now(timezone.utc)
    alert.acknowledged_by = acknowledged_by
    session.flush()
    return alert


def get_active_alerts(
    session: Session,
    *,
    alert_type: str | None = None,
    severity: str | None = None,
) -> list[Alert]:
    query = select(Alert).where(Alert.status == "active")
    if alert_type is not None:
        query = query.where(Alert.alert_type == alert_type)
    if severity is not None:
        query = query.where(Alert.severity == severity)
    query = query.order_by(Alert.created_at.desc())
    return list(session.execute(query).scalars().all())


def get_alert_stats(session: Session) -> dict[str, Any]:
    total = session.execute(select(func.count(Alert.id))).scalar_one()
    active = session.execute(
        select(func.count(Alert.id)).where(Alert.status == "active")
    ).scalar_one()
    acknowledged = session.execute(
        select(func.count(Alert.id)).where(Alert.status == "acknowledged")
    ).scalar_one()

    severity_counts = dict(
        session.execute(
            select(Alert.severity, func.count(Alert.id))
            .where(Alert.status == "active")
            .group_by(Alert.severity)
        ).all()
    )

    type_counts = dict(
        session.execute(
            select(Alert.alert_type, func.count(Alert.id))
            .where(Alert.status == "active")
            .group_by(Alert.alert_type)
        ).all()
    )

    return {
        "total": total,
        "active": active,
        "acknowledged": acknowledged,
        "by_severity": severity_counts,
        "by_type": type_counts,
    }


def check_gpu_memory(
    session: Session,
    *,
    device: str,
    free_mb: int,
    total_mb: int,
    threshold_pct: float = 10.0,
) -> Alert | None:
    used_pct = (total_mb - free_mb) / total_mb * 100 if total_mb > 0 else 0
    if free_mb <= 0 or used_pct >= (100 - threshold_pct):
        title = GPU_LOW_MEMORY.title_template.format(device=device)
        message = GPU_LOW_MEMORY.message_template.format(
            device=device, free_mb=free_mb, total_mb=total_mb, pct=used_pct
        )
        return create_alert(
            session,
            alert_type=GPU_LOW_MEMORY.alert_type,
            severity=GPU_LOW_MEMORY.severity,
            title=title,
            message=message,
            resource_type=GPU_LOW_MEMORY.resource_type,
            resource_id=f"gpu:{device}",
            metadata={"device": device, "free_mb": free_mb, "total_mb": total_mb, "used_pct": used_pct},
        )
    return None


def check_training_failure(
    session: Session,
    *,
    task_id: str,
    attempt_id: str,
    error: str,
) -> Alert:
    title = TRAINING_FAILED.title_template.format(task_id=task_id)
    message = TRAINING_FAILED.message_template.format(
        task_id=task_id, attempt_id=attempt_id, error=error
    )
    return create_alert(
        session,
        alert_type=TRAINING_FAILED.alert_type,
        severity=TRAINING_FAILED.severity,
        title=title,
        message=message,
        resource_type=TRAINING_FAILED.resource_type,
        resource_id=task_id,
        metadata={"task_id": task_id, "attempt_id": attempt_id, "error": error},
    )


def check_inference_error_rate(
    session: Session,
    *,
    error_count: int,
    total_count: int,
    threshold_pct: float = 5.0,
) -> Alert | None:
    if total_count == 0:
        return None
    error_rate = error_count / total_count * 100
    if error_rate >= threshold_pct:
        title = INFERENCE_ERROR_RATE.title_template
        message = INFERENCE_ERROR_RATE.message_template.format(
            error_rate=error_rate, window_size=total_count, threshold=threshold_pct
        )
        return create_alert(
            session,
            alert_type=INFERENCE_ERROR_RATE.alert_type,
            severity=INFERENCE_ERROR_RATE.severity,
            title=title,
            message=message,
            resource_type=INFERENCE_ERROR_RATE.resource_type,
            metadata={
                "error_count": error_count,
                "total_count": total_count,
                "error_rate": error_rate,
                "threshold": threshold_pct,
            },
        )
    return None


def check_disk_space(
    session: Session,
    *,
    path: str,
    free_gb: float,
    total_gb: float,
    threshold_pct: float = 10.0,
) -> Alert | None:
    usage_pct = (total_gb - free_gb) / total_gb * 100 if total_gb > 0 else 0
    free_pct = free_gb / total_gb * 100 if total_gb > 0 else 0
    if free_pct <= threshold_pct:
        title = DISK_SPACE_LOW.title_template.format(path=path)
        message = DISK_SPACE_LOW.message_template.format(
            path=path, usage_pct=usage_pct, free_gb=free_gb
        )
        return create_alert(
            session,
            alert_type=DISK_SPACE_LOW.alert_type,
            severity=DISK_SPACE_LOW.severity,
            title=title,
            message=message,
            resource_type=DISK_SPACE_LOW.resource_type,
            resource_id=f"disk:{path}",
            metadata={"path": path, "free_gb": free_gb, "total_gb": total_gb, "usage_pct": usage_pct},
        )
    return None


def check_service_availability(
    session: Session,
    *,
    service: str,
    is_healthy: bool,
    details: str = "",
) -> Alert | None:
    if not is_healthy:
        title = SERVICE_UNAVAILABLE.title_template.format(service=service)
        message = SERVICE_UNAVAILABLE.message_template.format(
            service=service, details=details or "no details"
        )
        return create_alert(
            session,
            alert_type=SERVICE_UNAVAILABLE.alert_type,
            severity=SERVICE_UNAVAILABLE.severity,
            title=title,
            message=message,
            resource_type=SERVICE_UNAVAILABLE.resource_type,
            resource_id=f"service:{service}",
            metadata={"service": service, "details": details},
        )
    return None
