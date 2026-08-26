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
    from backend.app.workers.db_session import worker_session
    from sqlalchemy import select as _select
    from backend.app.models import Evaluation as _Eval
    with worker_session() as session:
        pendings = list(session.execute(_select(_Eval).where(_Eval.auto_status == "pending")).scalars().all())
        if not pendings:
            return "no pending evaluations"
        from backend.app.workers.evaluation_worker import run_evaluation
        dispatched = 0
        for ev in pendings:
            try:
                run_evaluation.delay(str(ev.id))
                dispatched += 1
            except Exception:
                logger.exception("Failed to dispatch evaluation %s", ev.id)
        return f"dispatched {dispatched} pending evaluation(s)"


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
                if (
                    gpu_lease.owner_type == "training_attempt"
                    and gpu_lease.owner_id is not None
                ):
                    attempt = session.get(TrainingAttempt, gpu_lease.owner_id)
                    if attempt is not None and attempt.status in (
                        "running",
                        "recovering",
                    ):
                        expire_attempt(session, attempt.id)
                count += 1
            except Exception:
                logger.exception("Failed to expire GPU lease %s", gpu_lease.id)

        session.flush()
        return f"expired {count} lease(s)"


@celery_app.task(name="platform.scheduler.sweep_queued_tasks")
def sweep_queued_tasks() -> str:
    """Find tasks stuck in 'queued' status and re-dispatch them.

    Handles the crash window where a task was committed but Celery dispatch
    failed or the process crashed before the worker picked up the task.
    """
    from backend.app.workers.db_session import worker_session
    from backend.app.workers.train_worker import run_training

    with worker_session() as session:
        queued_tasks = list(
            session.execute(
                select(TrainingTask).where(
                    TrainingTask.status == "queued",
                    TrainingTask.cancellation_requested == False,
                )
            ).scalars().all()
        )

        if not queued_tasks:
            return "no queued tasks found"

        dispatched = 0
        for task in queued_tasks:
            try:
                run_training.delay(str(task.id))
                dispatched += 1
            except Exception:
                logger.exception("Failed to dispatch queued task %s", task.id)

        return f"dispatched {dispatched} queued task(s)"
