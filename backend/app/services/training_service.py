from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models import (
    DatasetSnapshot,
    ModelNode,
    TrainingAttempt,
    TrainingTask,
)


def create_task(
    session: Session,
    *,
    parent_model_node_id: UUID | None,
    dataset_snapshot_id: UUID,
    task_type: str,
    model_family: str,
    training_config_json: dict,
    resource_config_json: dict,
    evaluation_policy_json: dict,
    created_by: str | None = None,
    target_binding_id: UUID | None = None,
) -> TrainingTask:
    snapshot = session.get(DatasetSnapshot, dataset_snapshot_id)
    if snapshot is None:
        raise ValueError(f"Dataset snapshot {dataset_snapshot_id} not found")
    if parent_model_node_id is not None:
        parent = session.get(ModelNode, parent_model_node_id)
        if parent is None:
            raise ValueError(f"Parent model node {parent_model_node_id} not found")

    task = TrainingTask(
        parent_model_node_id=parent_model_node_id,
        dataset_snapshot_id=dataset_snapshot_id,
        target_binding_id=target_binding_id,
        task_type=task_type,
        model_family=model_family,
        training_config_json=training_config_json,
        resource_config_json=resource_config_json,
        evaluation_policy_json=evaluation_policy_json,
        status="queued",
        created_by=created_by,
    )
    session.add(task)
    session.flush()
    return task


def get_task(session: Session, task_id: UUID) -> TrainingTask | None:
    return session.get(TrainingTask, task_id)


def list_tasks(session: Session) -> list[TrainingTask]:
    return list(session.execute(select(TrainingTask)).scalars().all())


def start_attempt(session: Session, task_id: UUID) -> TrainingAttempt:
    task = session.get(TrainingTask, task_id, with_for_update=True)
    if task is None:
        raise ValueError(f"Training task {task_id} not found")
    if task.status not in ("queued", "running", "recovering", "failed"):
        raise ValueError(f"Cannot start attempt for task in '{task.status}' status")

    active = session.scalar(
        select(TrainingAttempt).where(
            TrainingAttempt.task_id == task_id,
            TrainingAttempt.status.in_(["pending", "running", "recovering"]),
        )
    )
    if active is not None:
        raise ValueError(
            f"Task {task_id} already has an active attempt "
            f"(attempt {active.attempt_no})"
        )

    last_attempt_no = session.scalar(
        select(func.coalesce(func.max(TrainingAttempt.attempt_no), 0)).where(
            TrainingAttempt.task_id == task_id
        )
    )
    retry_count = session.scalar(
        select(func.count()).where(
            TrainingAttempt.task_id == task_id,
            TrainingAttempt.status.in_(["failed", "cancelled"]),
        )
    )

    now = datetime.now(timezone.utc)
    attempt = TrainingAttempt(
        task=task,
        attempt_no=last_attempt_no + 1,
        status="running",
        lease_token=uuid4(),
        fencing_token=last_attempt_no + 1,
        retry_count=retry_count,
        started_at=now,
        heartbeat_at=now,
        lease_expires_at=now,
    )
    session.add(attempt)

    if task.status in ("queued", "failed", "recovering"):
        task.status = "running"

    session.flush()
    return attempt


def complete_attempt(
    session: Session,
    attempt_id: UUID,
    *,
    artifact_path: str,
    artifact_hash: str,
    framework: str = "pytorch",
    artifact_format: str = "pt",
    metadata_json: dict | None = None,
) -> ModelNode:
    attempt = session.get(TrainingAttempt, attempt_id, with_for_update=True)
    if attempt is None:
        raise ValueError(f"Training attempt {attempt_id} not found")
    if attempt.status not in ("pending", "running", "recovering"):
        raise ValueError(f"Cannot complete attempt in '{attempt.status}' status")

    attempt.status = "completed"
    attempt.finished_at = datetime.now(timezone.utc)

    task = session.get(TrainingTask, attempt.task_id, with_for_update=True)
    task.status = "completed"

    model = ModelNode(
        parent_id=task.parent_model_node_id,
        dataset_snapshot_id=task.dataset_snapshot_id,
        training_attempt_id=attempt.id,
        task_type=task.task_type,
        model_family=task.model_family,
        artifact_path=artifact_path,
        artifact_hash=artifact_hash,
        framework=framework,
        artifact_format=artifact_format,
        status="candidate",
        metadata_json=metadata_json or {},
    )
    session.add(model)
    session.flush()
    return model


def fail_attempt(
    session: Session,
    attempt_id: UUID,
    error: str,
) -> TrainingAttempt:
    attempt = session.get(TrainingAttempt, attempt_id, with_for_update=True)
    if attempt is None:
        raise ValueError(f"Training attempt {attempt_id} not found")
    if attempt.status not in ("pending", "running", "recovering"):
        raise ValueError(f"Cannot fail attempt in '{attempt.status}' status")

    attempt.status = "failed"
    attempt.last_error = error
    attempt.finished_at = datetime.now(timezone.utc)

    running = session.scalar(
        select(TrainingAttempt).where(
            TrainingAttempt.task_id == attempt.task_id,
            TrainingAttempt.id != attempt.id,
            TrainingAttempt.status.in_(["running", "recovering", "pending"]),
        )
    )
    if running is None:
        task = session.get(TrainingTask, attempt.task_id, with_for_update=True)
        if task is not None and task.status not in ("completed", "cancelled", "failed"):
            task.status = "failed"

    session.flush()
    return attempt


def cancel_task(session: Session, task_id: UUID) -> TrainingTask:
    task = session.get(TrainingTask, task_id, with_for_update=True)
    if task is None:
        raise ValueError(f"Training task {task_id} not found")
    if task.status in ("completed", "cancelled"):
        return task

    task.status = "cancelled"
    running_attempts = session.execute(
        select(TrainingAttempt).where(
            TrainingAttempt.task_id == task_id,
            TrainingAttempt.status.in_(["pending", "running", "recovering"]),
        )
    ).scalars().all()
    for attempt in running_attempts:
        attempt.status = "cancelled"
        attempt.finished_at = datetime.now(timezone.utc)
    session.flush()
    return task


def get_active_attempt(session: Session, task_id: UUID) -> TrainingAttempt | None:
    return session.scalar(
        select(TrainingAttempt).where(
            TrainingAttempt.task_id == task_id,
            TrainingAttempt.status.in_(["running", "recovering"]),
        )
    )
