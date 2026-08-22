from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from backend.app.models import OperationLog


def log_operation(
    session: Session,
    *,
    operation_type: str,
    actor: str | None = None,
    request_id: UUID | None = None,
    resource_type: str | None = None,
    resource_id: UUID | None = None,
    status: str = "success",
    summary_json: dict[str, Any] | None = None,
    error_summary: str | None = None,
) -> OperationLog:
    log = OperationLog(
        operation_type=operation_type,
        actor=actor,
        request_id=request_id,
        resource_type=resource_type,
        resource_id=resource_id,
        status=status,
        summary_json=summary_json or {},
        error_summary=error_summary,
    )
    session.add(log)
    session.flush()
    return log
