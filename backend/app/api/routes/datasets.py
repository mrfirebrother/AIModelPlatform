from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, verify_api_key
from backend.app.models import Dataset, DatasetSnapshot, LabelSchema
from backend.app.observability.operation_log import log_operation

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class DatasetCreate(BaseModel):
    name: str
    description: str | None = None


class DatasetResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    model_config = {"from_attributes": True}


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse]
    total: int


class SnapshotCreate(BaseModel):
    dataset_id: UUID
    label_schema_id: UUID
    source_path: str | None = None


class SnapshotResponse(BaseModel):
    id: UUID
    dataset_id: UUID
    label_schema_id: UUID
    manifest_path: str
    manifest_hash: str
    train_positive_count: int
    val_positive_count: int
    test_positive_count: int
    source_path: str | None
    model_config = {"from_attributes": True}


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(
    payload: DatasetCreate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        dataset = Dataset(name=payload.name, description=payload.description)
        db.add(dataset)
        db.flush()
        log_operation(
            db,
            operation_type="dataset.create",
            resource_type="dataset",
            resource_id=dataset.id,
            status="success",
            summary_json={"dataset_id": str(dataset.id), "name": payload.name},
        )
        return dataset
    except Exception as exc:
        log_operation(
            db,
            operation_type="dataset.create",
            status="error",
            error_summary=str(exc),
        )
        raise


@router.get("", response_model=DatasetListResponse)
def list_datasets(
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    datasets = list(db.execute(select(Dataset)).scalars().all())
    return DatasetListResponse(
        datasets=[DatasetResponse.model_validate(d) for d in datasets],
        total=len(datasets),
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    dataset_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return dataset


@router.post("/snapshots", response_model=SnapshotResponse, status_code=status.HTTP_201_CREATED)
def create_snapshot(
    payload: SnapshotCreate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    dataset = db.get(Dataset, payload.dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found"
        )
    schema = db.get(LabelSchema, payload.label_schema_id)
    if schema is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Label schema not found"
        )
    try:
        snapshot = DatasetSnapshot(
            dataset_id=payload.dataset_id,
            label_schema_id=payload.label_schema_id,
            manifest_path=f"/data/snapshots/{payload.dataset_id}/manifest.json",
            manifest_hash="sha256:placeholder",
            train_manifest_json=[],
            val_manifest_json=[],
            test_manifest_json=[],
            source_path=payload.source_path,
        )
        db.add(snapshot)
        db.flush()
        log_operation(
            db,
            operation_type="dataset.snapshot.create",
            resource_type="dataset_snapshot",
            resource_id=snapshot.id,
            status="success",
            summary_json={"snapshot_id": str(snapshot.id)},
        )
        return snapshot
    except Exception as exc:
        log_operation(
            db,
            operation_type="dataset.snapshot.create",
            status="error",
            error_summary=str(exc),
        )
        raise
