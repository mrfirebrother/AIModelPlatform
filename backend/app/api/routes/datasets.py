from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.api.dependencies import get_effective_settings, get_db, verify_api_key
from backend.app.models import Dataset, DatasetSnapshot, LabelSchema, LabelSchemaClass
from backend.app.schemas import CAMEL_CONFIG
from backend.app.observability.operation_log import log_operation

_MAX_DATASET_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class DatasetCreate(BaseModel):
    model_config = CAMEL_CONFIG
    name: str
    description: str | None = None
    source_path: str | None = None
    sourcePath: str | None = None

    def get_source_path(self) -> str | None:
        return self.source_path or self.sourcePath


class DatasetResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    model_config = {"from_attributes": True}


class DatasetListItemResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    label_schema_name: str | None = None
    image_count: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    latest_snapshot_id: UUID | None = None
    source: str | None = None
    source_path: str | None = None
    validation_status: str = "-"
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class DatasetListResponse(BaseModel):
    datasets: list[DatasetListItemResponse]
    total: int


class SnapshotCreate(BaseModel):
    model_config = CAMEL_CONFIG
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


def _normalise_class_names(names: Any) -> list[tuple[int, str]]:
    if isinstance(names, list):
        return [(class_id, str(name)) for class_id, name in enumerate(names)]
    if isinstance(names, dict):
        return [(int(class_id), str(name)) for class_id, name in names.items()]
    return []


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(
    payload: DatasetCreate,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
    settings: Any = Depends(get_effective_settings),
) -> Any:
    try:
        name = payload.name
        suffix = 1
        while db.query(Dataset).filter(Dataset.name == name).first():
            name = f"{payload.name}_{suffix}"
            suffix += 1
        dataset = Dataset(name=name, description=payload.description, source_path=payload.get_source_path())
        db.add(dataset)
        db.flush()

        # If source_path is provided, validate and create snapshot
        source_path_str = payload.get_source_path()
        if source_path_str:
            source_path = Path(source_path_str)
            # If it's a zip file, extract it first
            if source_path.exists() and source_path.is_file():
                import zipfile
                extract_dir = Path(settings.dataset_dir) / "extracted" / str(dataset.id)
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(source_path, 'r') as zf:
                    zf.extractall(extract_dir)
                # Find the actual dataset directory (might be nested)
                data_yaml = extract_dir / "data.yaml"
                if not data_yaml.exists():
                    # Check subdirectories
                    for sub in extract_dir.iterdir():
                        if sub.is_dir() and (sub / "data.yaml").exists():
                            extract_dir = sub
                            break
                source_dir = extract_dir
            elif source_path.exists() and source_path.is_dir():
                source_dir = source_path
            else:
                source_dir = None

            if source_dir and source_dir.exists():
                from backend.app.services.dataset_validation import validate_yolo_dataset
                validation = validate_yolo_dataset(source_dir)
                if validation.is_valid:
                    # Create a default label schema from data.yaml
                    data_yaml = {}
                    data_yaml_path = source_dir / "data.yaml"
                    if data_yaml_path.exists():
                        import yaml
                        with open(data_yaml_path) as f:
                            data_yaml = yaml.safe_load(f) or {}
                    names = _normalise_class_names(data_yaml.get("names", {}))
                    schema = LabelSchema(name=f"{name}_schema")
                    db.add(schema)
                    db.flush()
                    for cid, cname in names:
                        db.add(LabelSchemaClass(schema_id=schema.id, class_id=cid, semantic_key=cname, display_name=cname))
                    db.flush()

                    from backend.app.storage.snapshots import create_dataset_snapshot

                    snapshot_result = create_dataset_snapshot(
                        source_dir=source_dir,
                        store_root=Path(settings.dataset_dir) / "store",
                        snapshot_root=Path(settings.dataset_dir) / "snapshots",
                        label_schema_id=schema.id,
                        dataset_id=dataset.id,
                    )
                    snapshot = DatasetSnapshot(
                        dataset_id=dataset.id,
                        label_schema_id=schema.id,
                        manifest_path=str(snapshot_result.manifest_path),
                        manifest_hash=snapshot_result.manifest_hash,
                        train_manifest_json=snapshot_result.manifest.train_files,
                        val_manifest_json=snapshot_result.manifest.val_files,
                        test_manifest_json=snapshot_result.manifest.test_files,
                        source_path=str(source_dir),
                        train_positive_count=snapshot_result.manifest.train_positive,
                        val_positive_count=snapshot_result.manifest.val_positive,
                        test_positive_count=snapshot_result.manifest.test_positive,
                        train_negative_count=snapshot_result.manifest.train_negative,
                        val_negative_count=snapshot_result.manifest.val_negative,
                        test_negative_count=snapshot_result.manifest.test_negative,
                        quality_json={
                            "valid": True,
                            "warnings": [w.message for w in validation.warnings],
                        },
                    )
                    db.add(snapshot)
                    db.flush()

        log_operation(
            db,
            operation_type="dataset.create",
            resource_type="dataset",
            resource_id=dataset.id,
            status="success",
            summary_json={"dataset_id": str(dataset.id), "name": name},
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
    datasets = list(
        db.execute(
            select(Dataset).options(
                selectinload(Dataset.snapshots).selectinload(DatasetSnapshot.label_schema)
            )
        )
        .scalars()
        .all()
    )
    items: list[DatasetListItemResponse] = []
    for dataset in datasets:
        snapshot = max(
            dataset.snapshots,
            key=lambda value: value.created_at or datetime.min,
            default=None,
        )

        train_count = val_count = test_count = 0
        label_schema_name = None
        latest_snapshot_id = None
        validation_status = "-"
        warnings: list[str] = []
        if snapshot is not None:
            train_manifest = snapshot.train_manifest_json or []
            val_manifest = snapshot.val_manifest_json or []
            test_manifest = snapshot.test_manifest_json or []
            train_count = len(train_manifest) or (
                snapshot.train_positive_count + snapshot.train_negative_count
            )
            val_count = len(val_manifest) or (
                snapshot.val_positive_count + snapshot.val_negative_count
            )
            test_count = len(test_manifest) or (
                snapshot.test_positive_count + snapshot.test_negative_count
            )
            label_schema_name = snapshot.label_schema.name if snapshot.label_schema else None
            latest_snapshot_id = snapshot.id
            quality = snapshot.quality_json or {}
            warnings = quality.get("warnings", [])
            if snapshot.manifest_hash != "sha256:placeholder":
                validation_status = "通过"

        items.append(
            DatasetListItemResponse(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                label_schema_name=label_schema_name,
                image_count=train_count + val_count + test_count,
                train_count=train_count,
                val_count=val_count,
                test_count=test_count,
                latest_snapshot_id=latest_snapshot_id,
                source=Path(dataset.source_path).name if dataset.source_path else None,
                source_path=dataset.source_path,
                validation_status=validation_status,
                warnings=warnings,
                created_at=dataset.created_at,
            )
        )
    return DatasetListResponse(
        datasets=items,
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


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> None:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    try:
        # Delete snapshots first (foreign key constraint)
        snapshots = db.execute(select(DatasetSnapshot).where(DatasetSnapshot.dataset_id == dataset_id)).scalars().all()
        for snapshot in snapshots:
            db.delete(snapshot)
        db.flush()

        log_operation(
            db,
            operation_type="dataset.delete",
            resource_type="dataset",
            resource_id=dataset.id,
            status="success",
            summary_json={"dataset_id": str(dataset.id), "name": dataset.name},
        )
        db.delete(dataset)
        db.flush()
    except Exception as exc:
        log_operation(
            db,
            operation_type="dataset.delete",
            status="error",
            error_summary=str(exc),
        )
        raise


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


@router.post("/upload")
async def upload_dataset(
    file: UploadFile,
    _key: str = Depends(verify_api_key),
) -> Any:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )
    lower = file.filename.lower()
    if not (lower.endswith(".zip") or lower.endswith(".tar.gz") or lower.endswith(".tgz")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .zip or .tar.gz files are allowed",
        )

    settings = get_effective_settings()
    dataset_dir = Path(settings.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{secrets.token_hex(8)}_{file.filename}"
    dest = dataset_dir / safe_name

    sha256 = hashlib.sha256()
    total = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_DATASET_BYTES:
                    break
                f.write(chunk)
                sha256.update(chunk)
        if total > _MAX_DATASET_BYTES:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds {_MAX_DATASET_BYTES // (1024 * 1024 * 1024)} GB limit",
            )
    except HTTPException:
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {exc}",
        )

    return {
        "filename": file.filename,
        "file_path": str(dest),
        "size": total,
        "sha256": f"sha256:{sha256.hexdigest()}",
    }
