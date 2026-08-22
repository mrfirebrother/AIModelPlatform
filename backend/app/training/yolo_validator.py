from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    model_type: str
    num_classes: int | None = None
    input_size: tuple[int, int] | None = None
    framework: str = "pytorch"
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class YoloValidator:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def validate(self, model_path: Path) -> ValidationResult:
        if not model_path.exists():
            return ValidationResult(
                is_valid=False,
                model_type="unknown",
                errors=[f"Model file does not exist: {model_path}"],
            )

        if not model_path.is_file():
            return ValidationResult(
                is_valid=False,
                model_type="unknown",
                errors=[f"Path is not a file: {model_path}"],
            )

        suffix = model_path.suffix.lower()
        if suffix not in (".pt", ".pth"):
            return ValidationResult(
                is_valid=False,
                model_type="unknown",
                errors=[f"Unsupported model format: {suffix}. Only .pt and .pth are supported."],
            )

        try:
            return self._validate_pytorch_model(model_path)
        except ImportError:
            return ValidationResult(
                is_valid=False,
                model_type="unknown",
                errors=["PyTorch or ultralytics not installed. Cannot validate model."],
            )
        except Exception as exc:
            logger.exception("Failed to validate model: %s", model_path)
            return ValidationResult(
                is_valid=False,
                model_type="unknown",
                errors=[f"Validation failed: {exc}"],
            )

    def _validate_pytorch_model(self, model_path: Path) -> ValidationResult:
        from ultralytics import YOLO

        model = YOLO(str(model_path))

        metadata: dict[str, Any] = {}
        num_classes: int | None = None
        input_size: tuple[int, int] | None = None
        model_type = "yolo"
        errors: list[str] = []
        warnings: list[str] = []

        if hasattr(model, "names") and model.names:
            num_classes = len(model.names)
            metadata["class_names"] = list(model.names.values())

        if hasattr(model, "model") and hasattr(model.model, "yaml"):
            yaml_data = model.model.yaml
            if isinstance(yaml_data, dict):
                metadata["yaml_config"] = yaml_data
                if "nc" in yaml_data:
                    num_classes = yaml_data["nc"]

        try:
            dummy_input = self._create_dummy_input(model)
            if dummy_input is not None:
                import torch

                with torch.no_grad():
                    _ = model(dummy_input)
                input_size = (640, 640)
                metadata["inference_test"] = "passed"
        except Exception as exc:
            warnings.append(f"Inference test failed: {exc}")
            metadata["inference_test"] = "failed"

        metadata["model_path"] = str(model_path)
        metadata["device"] = self.device

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            model_type=model_type,
            num_classes=num_classes,
            input_size=input_size,
            framework="pytorch",
            metadata=metadata,
            errors=errors,
            warnings=warnings,
        )

    def _create_dummy_input(self, model: Any) -> Any:
        import torch

        try:
            if hasattr(model, "predict"):
                imgsz = 640
                if hasattr(model, "model") and hasattr(model.model, "yaml"):
                    yaml_data = model.model.yaml
                    if isinstance(yaml_data, dict) and "imgsz" in yaml_data:
                        imgsz = yaml_data["imgsz"]
                return torch.randn(1, 3, imgsz, imgsz).to(self.device)
        except Exception:
            pass
        return None

    def extract_metadata(self, model_path: Path) -> dict[str, Any]:
        result = self.validate(model_path)
        return {
            "is_valid": result.is_valid,
            "model_type": result.model_type,
            "num_classes": result.num_classes,
            "input_size": result.input_size,
            "framework": result.framework,
            "metadata": result.metadata,
            "errors": result.errors,
            "warnings": result.warnings,
        }
