from __future__ import annotations

import io
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from backend.app.services.dataset_validation import (
    DatasetValidationResult,
    validate_yolo_dataset,
)


class BatchImportStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    SNAPPING = "snapping"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchImportError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


@dataclass
class BatchImportProgress:
    status: BatchImportStatus = BatchImportStatus.PENDING
    total_steps: int = 0
    current_step: int = 0
    message: str = ""
    error_message: str = ""

    @property
    def percent(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return round(self.current_step / self.total_steps * 100, 1)

    def update_step(self, step: int, message: str = "") -> None:
        self.current_step = step
        self.message = message
        if step > 0 and self.status == BatchImportStatus.PENDING:
            self.status = BatchImportStatus.EXTRACTING

    def finish_success(self) -> None:
        self.current_step = self.total_steps
        self.status = BatchImportStatus.COMPLETED
        self.message = "Done"

    def finish_failure(self, error_message: str) -> None:
        self.status = BatchImportStatus.FAILED
        self.error_message = error_message


@dataclass
class ArchiveExtractionResult:
    extracted_path: Path
    archive_format: str
    file_count: int


@dataclass
class BatchImportResult:
    success: bool
    progress: BatchImportProgress
    validation_result: DatasetValidationResult | None = None
    extracted_path: Path | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


SUPPORTED_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz"}
ARCHIVE_MAGIC: dict[str, bytes] = {
    "zip": b"PK",
    "tar_gz": b"\x1f\x8b",
}


def _detect_archive_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".zip":
        return "zip"
    if ext in (".tar", ".gz", ".tgz"):
        if ext == ".gz" and path.stem.lower().endswith(".tar"):
            return "tar.gz"
        if ext == ".tgz":
            return "tar.gz"
        return "tar"
    with open(path, "rb") as f:
        header = f.read(4)
    if header[:2] == b"PK":
        return "zip"
    if header[:2] == b"\x1f\x8b":
        return "tar.gz"
    raise BatchImportError(
        code="unsupported_format",
        message=f"Unsupported archive format: {path.suffix}",
    )


def _is_safe_path(target: Path, base: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _check_path_traversal(member_name: str, base: Path) -> None:
    cleaned = member_name.replace("\\", "/")
    if ".." in cleaned.split("/"):
        raise BatchImportError(
            code="path_traversal",
            message=f"Path traversal detected in archive member: {member_name}",
            details={"member": member_name},
        )
    resolved = (base / cleaned).resolve()
    if not _is_safe_path(resolved, base):
        raise BatchImportError(
            code="path_traversal",
            message=f"Path traversal detected in archive member: {member_name}",
            details={"member": member_name},
        )


def _extract_zip(archive_path: Path, output_dir: Path) -> ArchiveExtractionResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_count = 0
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = zf.namelist()
            if not names:
                raise BatchImportError(
                    code="empty_archive",
                    message="Archive is empty",
                )
            for name in names:
                _check_path_traversal(name, output_dir)
            zf.extractall(output_dir)
            file_count = sum(1 for n in names if not n.endswith("/"))
    except zipfile.BadZipFile:
        raise BatchImportError(
            code="invalid_archive",
            message="Invalid or corrupt zip archive",
        )
    return ArchiveExtractionResult(
        extracted_path=output_dir,
        archive_format="zip",
        file_count=file_count,
    )


def _extract_tar(archive_path: Path, output_dir: Path) -> ArchiveExtractionResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_count = 0
    fmt = _detect_archive_format(archive_path)
    mode = "r:gz" if fmt == "tar.gz" else "r"

    try:
        with tarfile.open(archive_path, mode) as tf:
            members = tf.getmembers()
            if not members:
                raise BatchImportError(
                    code="empty_archive",
                    message="Archive is empty",
                )
            for member in members:
                _check_path_traversal(member.name, output_dir)
            tf.extractall(output_dir)
            file_count = sum(1 for m in members if m.isfile())
    except (tarfile.TarError, OSError) as e:
        raise BatchImportError(
            code="invalid_archive",
            message=f"Invalid or corrupt tar archive: {e}",
        )
    return ArchiveExtractionResult(
        extracted_path=output_dir,
        archive_format="tar",
        file_count=file_count,
    )


def extract_archive(archive_path: Path, output_dir: Path) -> ArchiveExtractionResult:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    fmt = _detect_archive_format(archive_path)
    if fmt == "zip":
        return _extract_zip(archive_path, output_dir)
    if fmt in ("tar", "tar.gz"):
        return _extract_tar(archive_path, output_dir)
    raise BatchImportError(
        code="unsupported_format",
        message=f"Unsupported archive format: {fmt}",
    )


def _find_dataset_root(extracted: Path) -> Path:
    if (extracted / "data.yaml").exists():
        return extracted
    for child in sorted(extracted.iterdir()):
        if child.is_dir() and (child / "data.yaml").exists():
            return child
    return extracted


def validate_archive_contents(dataset_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    data_yaml_path = dataset_dir / "data.yaml"
    if not data_yaml_path.exists():
        errors.append("data.yaml not found in extracted archive")

    result = validate_yolo_dataset(dataset_dir)
    for err in result.errors:
        errors.append(err.message)
    for warn in result.warnings:
        warnings.append(warn.message)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "train_count": result.train_count,
        "val_count": result.val_count,
        "test_count": result.test_count,
    }


on_progress_type = Callable[[BatchImportProgress], None] | None


def import_dataset_from_archive(
    archive_path: Path,
    store_root: Path,
    snapshot_root: Path,
    label_schema_id: UUID | str,
    dataset_id: UUID | str,
    name: str | None = None,
    on_progress: on_progress_type = None,
) -> BatchImportResult:
    total_steps = 4
    progress = BatchImportProgress(total_steps=total_steps)

    def _notify() -> None:
        if on_progress is not None:
            on_progress(progress)

    try:
        if not archive_path.exists():
            progress.finish_failure(f"Archive not found: {archive_path}")
            _notify()
            return BatchImportResult(
                success=False,
                progress=progress,
                errors=[f"Archive not found: {archive_path}"],
            )

        progress.update_step(1, "Extracting archive...")
        progress.status = BatchImportStatus.EXTRACTING
        _notify()

        tmp_dir = Path(store_root) / "_tmp_import" / f"import_{id(archive_path)}"
        try:
            extraction = extract_archive(archive_path, tmp_dir)
            dataset_dir = _find_dataset_root(extraction.extracted_path)

            progress.update_step(2, "Validating dataset...")
            progress.status = BatchImportStatus.VALIDATING
            _notify()

            validation = validate_yolo_dataset(dataset_dir)
            if not validation.is_valid:
                error_msgs = "; ".join(e.message for e in validation.errors)
                progress.finish_failure(f"Validation failed: {error_msgs}")
                _notify()
                return BatchImportResult(
                    success=False,
                    progress=progress,
                    errors=[e.message for e in validation.errors],
                    warnings=[w.message for w in validation.warnings],
                    extracted_path=extraction.extracted_path,
                )

            progress.update_step(3, "Creating snapshot...")
            progress.status = BatchImportStatus.SNAPPING
            _notify()

            from backend.app.storage.snapshots import create_dataset_snapshot

            uuid_label = UUID(label_schema_id) if isinstance(label_schema_id, str) else label_schema_id
            uuid_dataset = UUID(dataset_id) if isinstance(dataset_id, str) else dataset_id

            snapshot_result = create_dataset_snapshot(
                source_dir=dataset_dir,
                store_root=Path(store_root),
                snapshot_root=Path(snapshot_root),
                label_schema_id=uuid_label,
                dataset_id=uuid_dataset,
            )

            progress.update_step(4, "Finalizing...")
            _notify()

            progress.finish_success()
            _notify()

            return BatchImportResult(
                success=True,
                progress=progress,
                validation_result=validation,
                extracted_path=extraction.extracted_path,
                warnings=[w.message for w in validation.warnings],
            )
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    except (BatchImportError, FileNotFoundError) as e:
        progress.finish_failure(str(e))
        _notify()
        return BatchImportResult(
            success=False,
            progress=progress,
            errors=[str(e)],
        )
    except Exception as e:
        progress.finish_failure(f"Unexpected error: {e}")
        _notify()
        return BatchImportResult(
            success=False,
            progress=progress,
            errors=[f"Unexpected error: {e}"],
        )
