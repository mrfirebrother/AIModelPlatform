from __future__ import annotations

import base64
import logging
import time
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.models import BindingRelease, ModelBinding, ModelNode, RuntimeInstance
from backend.app.schemas.inference import (
    InferenceBatchRequest,
    InferenceBatchResponse,
    InferenceDetection,
    InferenceItem,
    InferenceRequest,
    InferenceResponse,
    InferenceResult,
    ModelInfoResponse,
)

logger = logging.getLogger("platform.inference")

ALLOWED_IMAGE_FORMATS = {"png", "jpg", "jpeg", "bmp", "webp"}
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024


class InferenceService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def resolve_binding(
        self,
        binding_id: UUID | None = None,
        model_node_id: UUID | None = None,
    ) -> ModelBinding | None:
        if binding_id is not None:
            return self._session.get(ModelBinding, binding_id)
        if model_node_id is not None:
            binding = (
                self._session.query(ModelBinding)
                .join(RuntimeInstance, RuntimeInstance.binding_id == ModelBinding.id)
                .filter(
                    RuntimeInstance.model_node_id == model_node_id,
                    RuntimeInstance.status == "serving",
                )
                .first()
            )
            return binding
        return None

    def validate_image(
        self,
        image_base64: str,
        image_format: str,
        max_size_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    ) -> bytes:
        if image_format.lower() not in ALLOWED_IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format: {image_format}")
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except Exception:
            raise ValueError("Invalid base64 image data")
        if len(image_bytes) == 0:
            raise ValueError("Empty image data")
        if len(image_bytes) > max_size_bytes:
            raise ValueError(
                f"Image exceeds maximum size ({max_size_bytes} bytes)"
            )
        return image_bytes

    def _resolve_runtime(
        self, binding: ModelBinding
    ) -> tuple[RuntimeInstance, BindingRelease | None, ModelNode | None]:
        if binding.current_runtime_instance_id is None:
            raise ValueError("No serving instance available for this binding")
        instance = self._session.get(RuntimeInstance, binding.current_runtime_instance_id)
        if instance is None or instance.status != "serving":
            raise ValueError("Runtime instance is not serving")
        release = self._session.get(BindingRelease, binding.current_release_id)
        model = self._session.get(ModelNode, instance.model_node_id) if instance.model_node_id else None
        return instance, release, model

    def run_inference(
        self,
        binding_id: UUID,
        image_base64: str,
        image_format: str,
    ) -> InferenceResponse:
        start_time = time.time()

        binding = self.resolve_binding(binding_id=binding_id)
        if binding is None:
            raise ValueError("Binding not found")

        image_bytes = self.validate_image(image_base64, image_format)
        instance, release, model = self._resolve_runtime(binding)

        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            "Inference completed: binding=%s model=%s latency=%.2fms",
            binding.id,
            instance.model_node_id,
            latency_ms,
        )

        return InferenceResponse(
            results=[
                InferenceResult(
                    detections=[],
                    modelBindingId=binding.id,
                    modelNodeId=instance.model_node_id,
                    releaseId=instance.release_id,
                    generation=instance.generation,
                )
            ],
            latency=latency_ms,
            modelNodeId=instance.model_node_id,
            bindingId=binding.id,
            releaseId=instance.release_id,
            generation=instance.generation,
        )

    def run_batch_inference(self, items: list[InferenceItem]) -> InferenceBatchResponse:
        start_time = time.time()
        results: list[InferenceResult] = []
        succeeded = 0
        failed = 0

        for item in items:
            try:
                binding_id = item.modelBindingId
                if binding_id is None and item.modelNodeId is not None:
                    binding = self.resolve_binding(model_node_id=item.modelNodeId)
                    if binding is None:
                        raise ValueError("No binding found for modelNodeId")
                    binding_id = binding.id

                image_base64 = item.input.get("image_base64", "")
                image_format = item.input.get("image_format", "png")

                resp = self.run_inference(
                    binding_id=binding_id,
                    image_base64=image_base64,
                    image_format=image_format,
                )
                if resp.results:
                    results.append(resp.results[0])
                succeeded += 1
            except Exception as exc:
                failed += 1
                logger.warning("Batch item failed: %s", exc)
                results.append(
                    InferenceResult(
                        modelBindingId=item.modelBindingId,
                        modelNodeId=item.modelNodeId,
                        detections=[],
                    )
                )

        total_latency = (time.time() - start_time) * 1000

        return InferenceBatchResponse(
            results=results,
            total_latency=total_latency,
            succeeded=succeeded,
            failed=failed,
        )

    def list_available_models(self) -> list[ModelInfoResponse]:
        models = (
            self._session.query(ModelNode)
            .filter(ModelNode.status == "approved")
            .all()
        )
        return [
            ModelInfoResponse(
                id=m.id,
                task_type=m.task_type,
                model_family=m.model_family,
                artifact_hash=m.artifact_hash,
                framework=m.framework,
                status=m.status,
            )
            for m in models
        ]
