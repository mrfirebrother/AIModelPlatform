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


#: 中断后最多自动重试次数（首次运行 + MAX_AUTO_RETRIES 次重试；耗尽则永久失败）
MAX_AUTO_RETRIES = 2


def _recover_or_fail_interrupted_attempt(session: Session, attempt: TrainingAttempt) -> str:
    """Heartbeat lost (worker killed / machine reboot): recover instead of failing.

    Releases the attempt's orphan GPU lease, closes the old attempt, re-queues
    the task and re-dispatches run_training. After MAX_AUTO_RETRIES the task is
    failed permanently instead of retrying forever.
    """
    from sqlalchemy import func as _func

    from backend.app.workers.train_worker import run_training

    task = session.get(TrainingTask, attempt.task_id, with_for_update=True)
    if task is None or task.status in ("completed", "cancelled"):
        expire_attempt(session, attempt.id)
        return "task terminal"

    attempt = session.get(TrainingAttempt, attempt.id, with_for_update=True)
    if attempt.status not in ("running", "recovering"):
        return "attempt not active"

    total_attempts = session.scalar(
        select(_func.count())
        .select_from(TrainingAttempt)
        .where(TrainingAttempt.task_id == task.id)
    ) or 0
    if total_attempts > MAX_AUTO_RETRIES:
        expire_attempt(session, attempt.id)
        task.status = "failed"
        task.failure_reason = f"自动重试 {MAX_AUTO_RETRIES} 次后仍中断，放弃自动恢复"
        session.flush()
        return "retry budget exhausted"

    # 释放该 attempt 名下的孤儿 GPU 租约（否则重试会因无可用显存立即失败）
    orphan_leases = list(
        session.execute(
            select(GPUResourceLease).where(
                GPUResourceLease.owner_type == "training_attempt",
                GPUResourceLease.owner_id == attempt.id,
                GPUResourceLease.status == "active",
            )
        ).scalars().all()
    )
    for gpu_lease in orphan_leases:
        expire_gpu_lease(session, gpu_lease.id)

    attempt.status = "failed"
    attempt.last_error = "Worker interrupted (lease expired); auto-rescheduling"
    attempt.finished_at = datetime.now(timezone.utc)
    task.status = "queued"
    task.failure_reason = None
    session.flush()

    try:
        run_training.delay(str(task.id))
    except Exception:
        # 无 broker 时（测试环境）任务保持 queued，由 sweep_queued_tasks 兜底
        logger.exception("Failed to re-dispatch task %s; sweep will retry", task.id)
    return "rescheduled"


@celery_app.task(name="platform.scheduler.reconcile_leases")
def reconcile_leases() -> str:
    """Find expired leases and recover (or fail) the associated training attempts.

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
                _recover_or_fail_interrupted_attempt(session, attempt)
                count += 1
            except Exception:
                logger.exception("Failed to recover attempt %s", attempt.id)

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
