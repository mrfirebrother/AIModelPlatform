from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File as FastAPIFile, HTTPException, UploadFile, status
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
    classes: list[str] | None = None

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


def _build_snapshot_background(dataset_id: UUID, source_path_str: str, dataset_dir_str: str) -> None:
    """Background task: validate + create snapshot for dataset (runs after HTTP response)."""
    from backend.app.workers.db_session import worker_session as _worker_session
    from backend.app.models import Dataset as _Dataset

    try:
        with _worker_session() as db:
            dataset = db.get(_Dataset, dataset_id)
            if dataset is None:
                return
            source_path = Path(source_path_str)
            dataset_dir = Path(dataset_dir_str)
            source_dir = None
            if source_path.exists() and source_path.is_file():
                import zipfile

                extract_dir = dataset_dir / "extracted" / str(dataset.id)
                extract_dir.mkdir(parents=True, exist_ok=True)
                if not any(extract_dir.iterdir()):
                    with zipfile.ZipFile(source_path, "r") as zf:
                        zf.extractall(extract_dir)
                from backend.app.services.dataset_validation import _locate_yaml

                yaml_path = _locate_yaml(extract_dir)
                if yaml_path is not None:
                    extract_dir = yaml_path.parent
                source_dir = extract_dir
            elif source_path.exists() and source_path.is_dir():
                source_dir = source_path
            if source_dir is None or not source_dir.exists():
                return
            from backend.app.services.dataset_validation import _locate_yaml, _parse_data_yaml

            data_yaml = {}
            yaml_path = _locate_yaml(source_dir)
            if yaml_path is not None:
                data_yaml = _parse_data_yaml(yaml_path)
                if not data_yaml.get("names"):
                    try:
                        import yaml

                        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
                        if raw.get("names"):
                            data_yaml["names"] = raw["names"]
                    except Exception:
                        pass
            names = _normalise_class_names(data_yaml.get("names", {}))
            if not names:
                names = [(0, "object")]
            schema = LabelSchema(name=f"{dataset.name}_schema")
            db.add(schema)
            db.flush()
            for cid, cname in names:
                db.add(LabelSchemaClass(schema_id=schema.id, class_id=cid, semantic_key=cname, display_name=cname))
            db.flush()
            from backend.app.storage.snapshots import create_dataset_snapshot

            try:
                snapshot_result = create_dataset_snapshot(
                    source_dir=source_dir,
                    store_root=dataset_dir / "store",
                    snapshot_root=dataset_dir / "snapshots",
                    label_schema_id=schema.id,
                    dataset_id=dataset.id,
                )
            except ValueError as ve:
                from backend.app.observability.operation_log import log_operation as _log

                _log(
                    db,
                    operation_type="dataset.create",
                    resource_type="dataset",
                    resource_id=dataset.id,
                    status="error",
                    error_summary=str(ve),
                )
                return
            # Retrieve validation result from snapshot for warnings
            validation = snapshot_result.validation_result
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
            from backend.app.observability.operation_log import log_operation as _log2

            _log2(
                db,
                operation_type="dataset.snapshot.create",
                resource_type="dataset_snapshot",
                resource_id=snapshot.id,
                status="success",
                summary_json={"dataset_id": str(dataset.id)},
            )
    except Exception as exc:
        try:
            from backend.app.workers.db_session import worker_session as _ws2
            from backend.app.observability.operation_log import log_operation as _log3

            with _ws2() as db2:
                _log3(
                    db2,
                    operation_type="dataset.create",
                    status="error",
                    error_summary=str(exc),
                )
        except Exception:
            pass


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(
    payload: DatasetCreate,
    background_tasks: BackgroundTasks,
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

        # Create label schema with provided classes
        class_list = payload.classes or ["object"]
        schema = LabelSchema(name=f"{name}_schema")
        db.add(schema)
        db.flush()
        for cid, cname in enumerate(class_list):
            db.add(LabelSchemaClass(schema_id=schema.id, class_id=cid, semantic_key=cname, display_name=cname))
        db.flush()

        # Link dataset to schema via a snapshot (so label_classes endpoint can find it)
        snap = DatasetSnapshot(
            dataset_id=dataset.id,
            label_schema_id=schema.id,
            manifest_path="",
            manifest_hash="",
            train_manifest_json=[],
            val_manifest_json=[],
            test_manifest_json=[],
            source_path=str(Path(settings.dataset_dir) / "extracted" / str(dataset.id)),
        )
        db.add(snap)
        db.flush()

        log_operation(
            db,
            operation_type="dataset.create",
            resource_type="dataset",
            resource_id=dataset.id,
            status="success",
            summary_json={"dataset_id": str(dataset.id), "name": name},
        )
        # Commit before the background task reads the dataset through another session.
        db.commit()
        # Schedule heavy snapshot creation in background to avoid 504 for large archives
        source_path_str = payload.get_source_path()
        if source_path_str:
            background_tasks.add_task(
                _build_snapshot_background, dataset.id, source_path_str, str(settings.dataset_dir)
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


@router.post("/{dataset_id}/images", status_code=status.HTTP_201_CREATED)
async def upload_dataset_images(
    dataset_id: UUID,
    files: list[UploadFile] = FastAPIFile(...),
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
    settings: Any = Depends(get_effective_settings),
) -> Any:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    saved = 0
    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        dest_dir = Path(settings.dataset_dir) / "extracted" / str(dataset_id) / "images" / "train"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f.filename
        with open(dest, "wb") as out:
            while True:
                chunk = await f.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        saved += 1
    return {"count": saved}


from fastapi.responses import FileResponse


class AnnotationPayload(BaseModel):
    model_config = CAMEL_CONFIG
    annotations: list[dict[str, Any]] = Field(default_factory=list)


@router.get("/{dataset_id}/annotations/{image_id}")
def get_annotations(
    dataset_id: UUID,
    image_id: str,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
    settings: Any = Depends(get_effective_settings),
) -> Any:
    """Load saved annotations for an image."""
    base = Path(settings.dataset_dir) / "extracted" / str(dataset_id)
    if not base.exists():
        return {"annotations": []}
    # Find label file
    for split in ["train", "val", "test"]:
        lbl_dir = base / "labels" / split
        if lbl_dir.exists():
            for f in lbl_dir.iterdir():
                if f.suffix == ".txt" and f.stem == image_id:
                    content = f.read_text(encoding="utf-8").strip()
                    if not content:
                        return {"annotations": []}
                    annotations = []
                    for line in content.splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            nums = list(map(float, parts[1:]))
                            # Heuristic: if exactly 5 numbers → bbox (cid x y w h), else polygon
                            if len(nums) == 4:
                                annotations.append({"class_id": cid, "bbox": nums})
                            else:
                                # polygon: x1 y1 x2 y2 ... (pairs)
                                pts = []
                                for i in range(0, len(nums) - 1, 2):
                                    pts.append([nums[i], nums[i + 1]])
                                annotations.append({"class_id": cid, "polygon": pts})
                    return {"annotations": annotations}
    return {"annotations": []}


@router.delete("/{dataset_id}/annotations/{image_id}")
def delete_annotations(
    dataset_id: UUID,
    image_id: str,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
    settings: Any = Depends(get_effective_settings),
) -> Any:
    """Delete all annotations for an image (clears label file)."""
    base = Path(settings.dataset_dir) / "extracted" / str(dataset_id)
    if not base.exists():
        return {"deleted": True}
    for split in ["train", "val", "test"]:
        lbl_dir = base / "labels" / split
        if lbl_dir.exists():
            for f in lbl_dir.iterdir():
                if f.suffix == ".txt" and f.stem == image_id:
                    f.write_text("", encoding="utf-8")
                    return {"deleted": True}
    return {"deleted": True}


@router.get("/{dataset_id}/images/{image_id}")
def get_dataset_image(
    dataset_id: UUID,
    image_id: str,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
    settings: Any = Depends(get_effective_settings),
) -> Any:
    """Serve an image file from dataset."""
    base = Path(settings.dataset_dir) / "extracted" / str(dataset_id)
    if not base.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    # Search in images/{split}/{image_id}.{ext}
    for split in ["train", "val", "test"]:
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            cand = base / "images" / split / f"{image_id}{ext}"
            if cand.exists():
                return FileResponse(cand, media_type="image/jpeg")
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")


@router.get("/{dataset_id}/image_list")
def list_dataset_images(
    dataset_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
    settings: Any = Depends(get_effective_settings),
) -> Any:
    """List image filenames in dataset."""
    base = Path(settings.dataset_dir) / "extracted" / str(dataset_id)
    if not base.exists():
        return {"images": [], "count": 0}
    images = []
    for split in ["train", "val", "test"]:
        img_dir = base / "images" / split
        if img_dir.exists():
            for f in sorted(img_dir.iterdir()):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    images.append(f.stem)  # stem = filename without ext
    return {"images": images, "count": len(images)}


@router.get("/{dataset_id}/label_classes")
def list_label_classes(
    dataset_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    """List class names from the dataset's label schema."""
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        return {"classes": []}
    # Try snapshot first, fall back to LabelSchema name convention
    from sqlalchemy import select as _sel
    from backend.app.models import DatasetSnapshot as _DS
    schema = None
    snap = db.execute(_sel(_DS).where(_DS.dataset_id == dataset_id).order_by(_DS.created_at.desc())).scalars().first()
    if snap is not None and snap.label_schema_id is not None:
        schema = db.get(LabelSchema, snap.label_schema_id)
    if schema is None:
        schema = db.execute(select(LabelSchema).where(LabelSchema.name == f"{dataset.name}_schema")).scalars().first()
    if schema is None:
        return {"classes": []}
    classes = db.execute(
        select(LabelSchemaClass).where(LabelSchemaClass.schema_id == schema.id).order_by(LabelSchemaClass.class_id)
    ).scalars().all()
    return {"classes": [c.semantic_key for c in classes]}


@router.put("/{dataset_id}/annotations/{image_id}")
def save_annotation(
    dataset_id: UUID,
    image_id: str,
    payload: AnnotationPayload,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
    settings: Any = Depends(get_effective_settings),
) -> Any:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    # Resolve image path and write labels
    base = Path(settings.dataset_dir) / "extracted" / str(dataset_id)
    # Find image file
    img_path = None
    for split in ["train", "val", "test"]:
        cand = base / "images" / split / f"{image_id}.jpg"
        if cand.exists():
            img_path = cand
            break
        cand = base / "images" / split / f"{image_id}.png"
        if cand.exists():
            img_path = cand
            break
    if img_path is None:
        # Fallback: search
        for p in base.rglob(f"{image_id}.*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                img_path = p
                break
    if img_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    # Derive label path
    from backend.app.services.dataset_validation import _label_dir_for_images_dir
    lbl_dir = _label_dir_for_images_dir(img_path.parent)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    lbl_path = lbl_dir / f"{Path(image_id).stem}.txt"
    lines = []
    for ann in payload.annotations:
        cid = ann.get("class_id", 0)
        if "bbox" in ann:
            x, y, w, h = ann["bbox"]
            lines.append(f"{cid} {x} {y} {w} {h}")
        elif "polygon" in ann:
            pts = ann["polygon"]
            flat = " ".join(f"{v:.6f}" for pt in pts for v in pt)
            lines.append(f"{cid} {flat}")
    lbl_path.write_text("\n".join(lines), encoding="utf-8")
    return {"saved": True, "count": len(lines)}


class BatchPayload(BaseModel):
    model_config = CAMEL_CONFIG
    operation: str
    image_ids: list[str] = Field(default_factory=list)
    target_class_id: int | None = None
    source_class_id: int | None = None


@router.post("/{dataset_id}/batch")
def batch_operation(
    dataset_id: UUID,
    payload: BatchPayload,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
    settings: Any = Depends(get_effective_settings),
) -> Any:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    base = Path(settings.dataset_dir) / "extracted" / str(dataset_id)
    if payload.operation == "delete":
        for iid in payload.image_ids:
            for p in base.rglob(f"{iid}.*"):
                try:
                    p.unlink()
                except OSError:
                    pass
            # Also delete label
            for p in base.rglob(f"{Path(iid).stem}.txt"):
                try:
                    p.unlink()
                except OSError:
                    pass
    elif payload.operation == "relabel" and payload.target_class_id is not None:
        for iid in payload.image_ids:
            for p in base.rglob(f"{Path(iid).stem}.txt"):
                try:
                    content = p.read_text(encoding="utf-8")
                    new_lines = []
                    for line in content.splitlines():
                        parts = line.strip().split()
                        if not parts:
                            continue
                        parts[0] = str(payload.target_class_id)
                        new_lines.append(" ".join(parts))
                    p.write_text("\n".join(new_lines), encoding="utf-8")
                except OSError:
                    pass
    return {"success": True}


@router.post("/{dataset_id}/snapshot", status_code=status.HTTP_201_CREATED)
def create_snapshot_for_dataset(
    dataset_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
    settings: Any = Depends(get_effective_settings),
) -> Any:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    # Reuse background task for snapshot (extracted dir is already source)
    base = Path(settings.dataset_dir) / "extracted" / str(dataset_id)
    if not base.exists():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No data to snapshot")
    background_tasks.add_task(_build_snapshot_background, dataset.id, str(base), str(settings.dataset_dir))
    return {"scheduled": True}


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
