from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import TrainingAttempt, TrainingTask
from backend.app.workers.attempt_lease import expire_attempt, find_expired_attempts

from .celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="platform.scheduler.sweep_pending_evaluations")
def sweep_pending_evaluations() -> str:
    """Placeholder for a future idempotent evaluation enqueue sweep."""
    return "pending evaluation sweep not implemented"


@celery_app.task(name="platform.scheduler.reconcile_leases")
def reconcile_leases() -> str:
    """Find expired leases and fail the associated training attempts."""
    from backend.app.workers.db_session import worker_session

    with worker_session() as session:
        expired = find_expired_attempts(session)
        if not expired:
            return "no expired leases found"

        count = 0
        for attempt in expired:
            try:
                expire_attempt(session, attempt.id)
                task = session.get(TrainingTask, attempt.task_id)
                if task is not None and task.status not in ("completed", "cancelled", "failed"):
                    task.status = "failed"
                count += 1
            except Exception:
                logger.exception("Failed to expire attempt %s", attempt.id)

        session.flush()
        return f"expired {count} lease(s)"
