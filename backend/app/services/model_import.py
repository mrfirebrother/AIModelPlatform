from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from backend.app.models import ModelNode
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class ModelImportResult:
    model_id: UUID
    artifact_path: str
    artifact_hash: str
    framework: str
    artifact_format: str
    status: str
    validation_errors: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


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


def validate_model_real(file_path: Path, device: str = "cpu") -> tuple[bool, list[str], dict[str, Any]]:
    from backend.app.training.yolo_validator import YoloValidator

    errors: list[str] = []
    metadata: dict[str, Any] = {}

    is_valid_base, base_errors = validate_model_file(file_path)
    if not is_valid_base:
        return False, base_errors, metadata

    try:
        validator = YoloValidator(device=device)
        result = validator.validate(file_path)

        if not result.is_valid:
            errors.extend(result.errors)
            return False, errors, metadata

        metadata["model_type"] = result.model_type
        metadata["num_classes"] = result.num_classes
        metadata["input_size"] = result.input_size
        metadata["framework"] = result.framework
        metadata.update(result.metadata)

        if result.warnings:
            metadata["warnings"] = result.warnings
            logger.warning("Model validation warnings: %s", result.warnings)

        return True, errors, metadata

    except ImportError:
        logger.warning("ultralytics not installed, skipping real validation")
        return True, [], metadata
    except Exception as exc:
        logger.exception("Real model validation failed")
        errors.append(f"Real validation failed: {exc}")
        return False, errors, metadata


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
    real_validation: bool = True,
    device: str = "cpu",
) -> ModelImportResult:
    if real_validation:
        is_valid, validation_errors, real_metadata = validate_model_real(file_path, device=device)
        if not is_valid:
            return ModelImportResult(
                model_id=uuid4(),
                artifact_path=str(file_path),
                artifact_hash="",
                framework=framework,
                artifact_format=file_path.suffix.lstrip("."),
                status="rejected",
                validation_errors=validation_errors,
                metadata=real_metadata,
            )
        merged_metadata = {**(metadata_json or {}), **real_metadata}
    else:
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
        merged_metadata = metadata_json or {}

    artifact_hash = compute_file_hash(file_path)
    from backend.app.repositories.model_repository import _generate_model_code
    model = ModelNode(
        code=_generate_model_code(session),
        task_type=task_type,
        model_family=model_family,
        artifact_path=str(file_path),
        artifact_hash=artifact_hash,
        framework=framework,
        artifact_format=file_path.suffix.lstrip("."),
        status="approved",
        metadata_json=merged_metadata,
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
        metadata=merged_metadata,
    )
