from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from backend.app.models import ModelNode
from sqlalchemy.orm import Session


@dataclass
class ModelImportResult:
    model_id: UUID
    artifact_path: str
    artifact_hash: str
    framework: str
    artifact_format: str
    status: str
    validation_errors: list[str]


def validate_model_file(file_path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not file_path.exists():
        errors.append(f"File does not exist: {file_path}")
        return False, errors
    if not file_path.is_file():
        errors.append(f"Path is not a file: {file_path}")
        return False, errors
    suffix = file_path.suffix.lower()
    if suffix not in (".pt", ".pth", ".onnx", ".bin", ".safetensors"):
        errors.append(f"Unsupported model format: {suffix}")
        return False, errors
    try:
        file_size = file_path.stat().st_size
        if file_size == 0:
            errors.append("Model file is empty")
            return False, errors
        if file_size < 1024:
            errors.append(f"Model file suspiciously small: {file_size} bytes")
            return False, errors
    except OSError as e:
        errors.append(f"Cannot read model file: {e}")
        return False, errors
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
            if suffix in (".pt", ".pth"):
                magic = header[:4]
                if magic != b"PK\x03\x04" and not header[0:2] == b"\x80\x04":
                    errors.append("File does not appear to be a valid PyTorch model")
                    return False, errors
    except (OSError, PermissionError) as e:
        errors.append(f"Cannot read model header: {e}")
        return False, errors
    return True, errors


def compute_file_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def import_root_model(
    session: Session,
    file_path: Path,
    *,
    task_type: str = "object_detection",
    model_family: str = "yolo",
    framework: str = "pytorch",
    metadata_json: dict[str, Any] | None = None,
) -> ModelImportResult:
    is_valid, validation_errors = validate_model_file(file_path)
    if not is_valid:
        return ModelImportResult(
            model_id=uuid4(),
            artifact_path=str(file_path),
            artifact_hash="",
            framework=framework,
            artifact_format=file_path.suffix.lstrip("."),
            status="rejected",
            validation_errors=validation_errors,
        )
    artifact_hash = compute_file_hash(file_path)
    model = ModelNode(
        task_type=task_type,
        model_family=model_family,
        artifact_path=str(file_path),
        artifact_hash=artifact_hash,
        framework=framework,
        artifact_format=file_path.suffix.lstrip("."),
        status="approved",
        metadata_json=metadata_json or {},
    )
    session.add(model)
    session.flush()
    return ModelImportResult(
        model_id=model.id,
        artifact_path=str(file_path),
        artifact_hash=artifact_hash,
        framework=framework,
        artifact_format=file_path.suffix.lstrip("."),
        status="approved",
        validation_errors=[],
    )
