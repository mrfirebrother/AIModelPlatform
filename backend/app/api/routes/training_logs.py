from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, verify_api_key
from backend.app.models import Checkpoint, TrainingAttempt, TrainingTask

router = APIRouter(prefix="/api/training", tags=["training-logs"])


class AttemptLogEntry(BaseModel):
    attempt_no: int
    status: str
    current_epoch: int
    log_path: str | None
    started_at: str | None
    finished_at: str | None
    last_error: str | None
    model_config = {"from_attributes": True}


class TrainingLogsResponse(BaseModel):
    task_id: str
    status: str
    current_epoch: int
    total_epochs: int
    attempts: list[AttemptLogEntry]
    model_config = {"from_attributes": True}


class EpochMetrics(BaseModel):
    epoch: int
    loss: float
    lr: float | None = None
    extra: dict[str, Any] = {}


class TrainingMetricsResponse(BaseModel):
    task_id: str
    epochs: list[EpochMetrics]
    best_loss: float | None
    model_config = {"from_attributes": True}


class CheckpointInfo(BaseModel):
    epoch: int
    artifact_path: str
    artifact_hash: str
    metrics: dict[str, Any]
    created_at: str
    model_config = {"from_attributes": True}


class TrainingCheckpointResponse(BaseModel):
    task_id: str
    attempt_no: int | None
    latest_checkpoint: CheckpointInfo | None
    model_config = {"from_attributes": True}


def _get_task_or_404(db: Session, task_id: UUID) -> TrainingTask:
    task = db.get(TrainingTask, task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Training task not found"
        )
    return task


@router.get("/{task_id}/logs", response_model=TrainingLogsResponse)
def get_training_logs(
    task_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    task = _get_task_or_404(db, task_id)

    training_config = task.training_config_json or {}
    total_epochs = training_config.get("epochs", 0)

    attempts = list(
        db.execute(
            select(TrainingAttempt).where(TrainingAttempt.task_id == task_id)
        ).scalars().all()
    )

    current_epoch = 0
    for a in attempts:
        if a.current_epoch > current_epoch:
            current_epoch = a.current_epoch
    if current_epoch == 0 and task.status == "completed":
        current_epoch = total_epochs

    attempt_entries = [
        AttemptLogEntry(
            attempt_no=a.attempt_no,
            status=a.status,
            current_epoch=a.current_epoch,
            log_path=a.log_path,
            started_at=a.started_at.isoformat() if a.started_at else None,
            finished_at=a.finished_at.isoformat() if a.finished_at else None,
            last_error=a.last_error,
        )
        for a in sorted(attempts, key=lambda x: x.attempt_no)
    ]

    return TrainingLogsResponse(
        task_id=str(task.id),
        status=task.status,
        current_epoch=current_epoch,
        total_epochs=total_epochs,
        attempts=attempt_entries,
    )


@router.get("/{task_id}/metrics", response_model=TrainingMetricsResponse)
def get_training_metrics(
    task_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    _get_task_or_404(db, task_id)

    attempt_ids = [
        a.id
        for a in db.execute(
            select(TrainingAttempt).where(TrainingAttempt.task_id == task_id)
        ).scalars().all()
    ]

    checkpoints = list(
        db.execute(
            select(Checkpoint).where(Checkpoint.attempt_id.in_(attempt_ids))
        ).scalars().all()
    )

    epochs = []
    best_loss: float | None = None
    for ckpt in sorted(checkpoints, key=lambda c: c.epoch):
        metrics = ckpt.metrics_json or {}
        loss = metrics.get("loss")
        lr = metrics.get("lr")
        extra = {k: v for k, v in metrics.items() if k not in ("loss", "lr")}

        epochs.append(
            EpochMetrics(
                epoch=ckpt.epoch,
                loss=loss if loss is not None else 0.0,
                lr=lr,
                extra=extra,
            )
        )

        if loss is not None and (best_loss is None or loss < best_loss):
            best_loss = loss

    return TrainingMetricsResponse(
        task_id=str(task_id),
        epochs=epochs,
        best_loss=best_loss,
    )


@router.get("/{task_id}/checkpoint", response_model=TrainingCheckpointResponse)
def get_training_checkpoint(
    task_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    _get_task_or_404(db, task_id)

    attempts = list(
        db.execute(
            select(TrainingAttempt).where(TrainingAttempt.task_id == task_id)
        ).scalars().all()
    )

    if not attempts:
        return TrainingCheckpointResponse(
            task_id=str(task_id),
            attempt_no=None,
            latest_checkpoint=None,
        )

    latest_attempt = max(attempts, key=lambda a: a.attempt_no)

    checkpoints = list(
        db.execute(
            select(Checkpoint).where(Checkpoint.attempt_id == latest_attempt.id)
        ).scalars().all()
    )

    if not checkpoints:
        return TrainingCheckpointResponse(
            task_id=str(task_id),
            attempt_no=latest_attempt.attempt_no,
            latest_checkpoint=None,
        )

    latest_ckpt = max(checkpoints, key=lambda c: c.epoch)

    return TrainingCheckpointResponse(
        task_id=str(task_id),
        attempt_no=latest_attempt.attempt_no,
        latest_checkpoint=CheckpointInfo(
            epoch=latest_ckpt.epoch,
            artifact_path=latest_ckpt.artifact_path,
            artifact_hash=latest_ckpt.artifact_hash,
            metrics=latest_ckpt.metrics_json or {},
            created_at=latest_ckpt.created_at.isoformat(),
        ),
    )
