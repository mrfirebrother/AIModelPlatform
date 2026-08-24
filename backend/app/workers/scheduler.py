from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import GPUResourceLease, TrainingAttempt, TrainingTask
from backend.app.workers.attempt_lease import expire_attempt, find_expired_attempts
from backend.app.services.resource_lease import (
    expire_lease as expire_gpu_lease,
    find_expired_leases,
)

from .celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="platform.scheduler.sweep_pending_evaluations")
def sweep_pending_evaluations() -> str:
    """Placeholder for a future idempotent evaluation enqueue sweep."""
    return "pending evaluation sweep not implemented"


@celery_app.task(name="platform.scheduler.reconcile_leases")
def reconcile_leases() -> str:
    """Find expired leases and fail the associated training attempts.

    Handles both TrainingAttempt leases and GPUResourceLease records.
    """
    from backend.app.workers.db_session import worker_session

    with worker_session() as session:
        expired_attempts = find_expired_attempts(session)
        expired_gpu_leases = find_expired_leases(session)

        if not expired_attempts and not expired_gpu_leases:
            return "no expired leases found"

        count = 0
        for attempt in expired_attempts:
            try:
                expire_attempt(session, attempt.id)
                task = session.get(TrainingTask, attempt.task_id)
                if task is not None and task.status not in ("completed", "cancelled", "failed"):
                    task.status = "failed"
                count += 1
            except Exception:
                logger.exception("Failed to expire attempt %s", attempt.id)

        for gpu_lease in expired_gpu_leases:
            try:
                expire_gpu_lease(session, gpu_lease.id)
                count += 1
            except Exception:
                logger.exception("Failed to expire GPU lease %s", gpu_lease.id)

        session.flush()
        return f"expired {count} lease(s)"
