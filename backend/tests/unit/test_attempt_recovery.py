"""Tests for _recover_or_fail_interrupted_attempt.

Scenario: the worker process dies (machine reboot) mid-training. The attempt's
heartbeat stops, the lease expires, and the scheduler must reschedule the task
instead of failing it permanently.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.models import (
    Base,
    GPUResource,
    GPUResourceLease,
    TrainingAttempt,
    TrainingTask,
)
from backend.app.workers.db_session import reset_worker_session_factory, set_worker_engine_override
from backend.app.workers.scheduler import MAX_AUTO_RETRIES, _recover_or_fail_interrupted_attempt


@pytest.fixture()
def session():
    eng = create_engine("sqlite:///:memory:", poolclass=StaticPool)

    @event.listens_for(eng, "connect")
    def _enable_fk(dbapi_conn, _record):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(eng)
    set_worker_engine_override(eng)
    db = sessionmaker(bind=eng)()
    try:
        yield db
    finally:
        db.close()
        reset_worker_session_factory()


def _make_interrupted_task(session, *, prior_attempts: int = 0) -> tuple[TrainingTask, TrainingAttempt]:
    task = TrainingTask(
        task_type="instance_segmentation",
        model_family="yolo",
        status="running",
        training_config_json={"epochs": 5},
    )
    session.add(task)
    session.flush()
    for n in range(prior_attempts):
        session.add(
            TrainingAttempt(
                task_id=task.id,
                attempt_no=n + 1,
                status="failed",
                gpu_device="0",
                lease_token=None,
                retry_count=0,
                last_error="interrupted",
                finished_at=datetime.now(timezone.utc),
            )
        )
    attempt = TrainingAttempt(
        task_id=task.id,
        attempt_no=prior_attempts + 1,
        status="running",
        gpu_device="0",
        lease_token=None,
        retry_count=0,
        heartbeat_at=datetime.now(timezone.utc) - timedelta(minutes=60),
        lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=30),
        started_at=datetime.now(timezone.utc) - timedelta(minutes=60),
    )
    session.add(attempt)
    session.flush()
    gpu = GPUResource(gpu_device="0", capacity_memory_mb=4096, reserved_memory_mb=0)
    session.add(gpu)
    session.flush()
    session.add(
        GPUResourceLease(
            gpu_device="0",
            gpu_resource_id=gpu.id,
            owner_type="training_attempt",
            owner_id=attempt.id,
            status="active",
            reserved_memory_mb=4096,
            lease_token=uuid4(),
            fencing_token=1,
            heartbeat_at=datetime.now(timezone.utc) - timedelta(minutes=60),
            lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=30),
        )
    )
    session.commit()
    return task, attempt


def test_interrupted_attempt_reschedules_task(session):
    task, attempt = _make_interrupted_task(session)

    result = _recover_or_fail_interrupted_attempt(session, attempt)

    assert result == "rescheduled"
    session.expire_all()
    assert task.status == "queued"
    assert task.failure_reason is None
    fresh_attempt = session.get(TrainingAttempt, attempt.id)
    assert fresh_attempt.status == "failed"
    assert "rescheduling" in (fresh_attempt.last_error or "")
    gpu_lease = session.query(GPUResourceLease).one()
    assert gpu_lease.status == "expired"


def test_retry_budget_exhausted_fails_task_permanently(session):
    task, attempt = _make_interrupted_task(session, prior_attempts=MAX_AUTO_RETRIES)

    result = _recover_or_fail_interrupted_attempt(session, attempt)

    assert result == "retry budget exhausted"
    session.expire_all()
    assert task.status == "failed"
    assert "放弃" in (task.failure_reason or "")


def test_terminal_task_just_expires_attempt(session):
    task, attempt = _make_interrupted_task(session)
    task.status = "cancelled"
    session.commit()

    result = _recover_or_fail_interrupted_attempt(session, attempt)

    assert result == "task terminal"
    session.expire_all()
    assert task.status == "cancelled"
    assert session.get(TrainingAttempt, attempt.id).status == "failed"
