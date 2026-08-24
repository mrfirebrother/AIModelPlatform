from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.schemas import CAMEL_CONFIG
from backend.app.api.dependencies import get_db, verify_api_key
from backend.app.observability.operation_log import log_operation
from backend.app.services import training_service

router = APIRouter(prefix="/api/training", tags=["training"])


class TrainingTaskCreate(BaseModel):
    model_config = CAMEL_CONFIG
    dataset_snapshot_id: UUID
    datasetSnapshotId: UUID | None = None
    parent_model_node_id: UUID | None = None
    parentModelNodeId: UUID | None = None
    task_type: str
    model_family: str
    training_config_json: dict = {}
    trainingConfigJson: dict | None = None
    resource_config_json: dict = {}
    evaluation_policy_json: dict = {}
    created_by: str | None = None
    target_binding_id: UUID | None = None
    targetBindingId: UUID | None = None
    epochs: int | None = None

    def get_snapshot_id(self) -> UUID:
        return self.dataset_snapshot_id or self.datasetSnapshotId

    def get_parent_id(self) -> UUID | None:
        return self.parent_model_node_id or self.parentModelNodeId

    def get_config(self) -> dict:
        return self.training_config_json or self.trainingConfigJson or {}


class TrainingTaskResponse(BaseModel):
    id: UUID
    parent_model_node_id: UUID | None
    dataset_snapshot_id: UUID
    target_binding_id: UUID | None
    task_type: str
    model_family: str
    training_config_json: dict
    resource_config_json: dict
    evaluation_policy_json: dict
    status: str
    created_by: str | None
    model_config = {"from_attributes": True}

    @property
    def epochs(self) -> int:
        return self.training_config_json.get("epochs", 0)

    @property
    def parent_model_name(self) -> str | None:
        return None  # Would need to query DB

    @property
    def dataset_name(self) -> str | None:
        return None  # Would need to query DB


class TrainingTaskListResponse(BaseModel):
    tasks: list[TrainingTaskResponse]
    total: int


@router.post("/tasks", response_model=TrainingTaskResponse, status_code=status.HTTP_201_CREATED)
def create_training_task(
    payload: TrainingTaskCreate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        task = training_service.create_task(
            db,
            parent_model_node_id=payload.parent_model_node_id,
            dataset_snapshot_id=payload.dataset_snapshot_id,
            task_type=payload.task_type,
            model_family=payload.model_family,
            training_config_json=payload.training_config_json,
            resource_config_json=payload.resource_config_json,
            evaluation_policy_json=payload.evaluation_policy_json,
            created_by=payload.created_by,
            target_binding_id=payload.target_binding_id,
        )
        log_operation(
            db,
            operation_type="training.task.create",
            resource_type="training_task",
            resource_id=task.id,
            status="success",
            summary_json={"task_id": str(task.id)},
        )
        # Dispatch Celery task
        from backend.app.workers.train_worker import run_training
        run_training.delay(str(task.id))
        return task
    except ValueError as exc:
        log_operation(
            db,
            operation_type="training.task.create",
            status="error",
            error_summary=str(exc),
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        log_operation(
            db,
            operation_type="training.task.create",
            status="error",
            error_summary=str(exc),
        )
        raise


@router.get("/tasks", response_model=TrainingTaskListResponse)
def list_training_tasks(
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    tasks = training_service.list_tasks(db)
    return TrainingTaskListResponse(
        tasks=[TrainingTaskResponse.model_validate(t) for t in tasks],
        total=len(tasks),
    )


@router.get("/tasks/{task_id}", response_model=TrainingTaskResponse)
def get_training_task(
    task_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    task = training_service.get_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return task


@router.post("/tasks/{task_id}/cancel", response_model=TrainingTaskResponse)
def cancel_training_task(
    task_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        task = training_service.cancel_task(db, task_id)
        log_operation(
            db,
            operation_type="training.task.cancel",
            resource_type="training_task",
            resource_id=task.id,
            status="success",
            summary_json={"task_id": str(task.id)},
        )
        return task
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
