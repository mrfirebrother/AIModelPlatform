from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from backend.app.services.dataset_validation import (
    DatasetValidationResult,
    DatasetValidationError,
    DatasetValidationWarning,
    validate_yolo_dataset,
)
from backend.app.storage.snapshots import SnapshotResult, create_dataset_snapshot


@dataclass
class DatasetImportRequest:
    source_dir: Path
    label_schema_id: UUID
    dataset_id: UUID
    name: str | None = None


@dataclass
class DatasetImportResult:
    source_dir: Path
    snapshot_result: SnapshotResult
    validation_result: DatasetValidationResult


def import_dataset_from_directory(
    request: DatasetImportRequest,
    store_root: Path,
    snapshot_root: Path,
) -> DatasetImportResult:
    source_dir = request.source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    validation = validate_yolo_dataset(source_dir)
    if not validation.is_valid:
        error_msgs = "; ".join(e.message for e in validation.errors)
        raise ValueError(f"Dataset validation failed: {error_msgs}")

    snapshot_result = create_dataset_snapshot(
        source_dir=source_dir,
        store_root=store_root,
        snapshot_root=snapshot_root,
        label_schema_id=request.label_schema_id,
        dataset_id=request.dataset_id,
    )

    return DatasetImportResult(
        source_dir=source_dir,
        snapshot_result=snapshot_result,
        validation_result=validation,
    )


@dataclass
class ArchiveImportRequest:
    archive_path: Path
    label_schema_id: UUID
    dataset_id: UUID
    store_root: Path
    snapshot_root: Path
    name: str | None = None


def import_dataset_from_archive(
    request: ArchiveImportRequest,
) -> DatasetImportResult:
    from backend.app.services.batch_import import (
        import_dataset_from_archive as batch_import,
    )

    result = batch_import(
        archive_path=request.archive_path,
        store_root=request.store_root,
        snapshot_root=request.snapshot_root,
        label_schema_id=request.label_schema_id,
        dataset_id=request.dataset_id,
        name=request.name,
    )

    if not result.success:
        error_msgs = "; ".join(result.errors)
        raise ValueError(f"Archive import failed: {error_msgs}")

    from backend.app.services.batch_import import BatchImportResult

    dataset_dir = result.extracted_path
    if dataset_dir is None:
        raise ValueError("Import completed but extracted path is missing")

    validation = validate_yolo_dataset(dataset_dir)
    snapshot_result = create_dataset_snapshot(
        source_dir=dataset_dir,
        store_root=request.store_root,
        snapshot_root=request.snapshot_root,
        label_schema_id=request.label_schema_id,
        dataset_id=request.dataset_id,
    )

    return DatasetImportResult(
        source_dir=dataset_dir,
        snapshot_result=snapshot_result,
        validation_result=validation,
    )
