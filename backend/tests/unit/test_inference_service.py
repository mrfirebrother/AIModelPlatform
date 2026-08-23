from __future__ import annotations

import base64
import io
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from backend.app.models import Base, BindingRelease, ModelBinding, ModelNode, RuntimeInstance
from backend.app.schemas.inference import (
    InferenceRequest,
    InferenceBatchRequest,
    InferenceItem,
    InferenceResponse,
    InferenceBatchResponse,
    ModelInfoResponse,
)


def _make_session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    event.listen(engine, "connect", lambda c, _: c.execute("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    return Session(engine)


def _make_test_image_base64() -> str:
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return base64.b64encode(png_data).decode("utf-8")


def _seed_binding(session: Session, *, status: str = "bound", has_runtime: bool = True):
    binding = ModelBinding(external_ref="ref-001", status=status)
    model = ModelNode(
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/test.pt",
        artifact_hash="sha256:abc123",
        status="approved",
    )
    session.add(binding)
    session.add(model)
    session.flush()

    release = BindingRelease(
        binding_id=binding.id,
        revision_no=1,
        model_node_id=model.id,
        inference_config_json={"confidence": 0.5},
        inference_config_hash="sha256:config123",
        release_type="normal",
        status="active",
    )
    session.add(release)
    session.flush()
    binding.current_release_id = release.id
    session.flush()

    instance = None
    if has_runtime:
        instance = RuntimeInstance(
            binding_id=binding.id,
            release_id=release.id,
            model_node_id=model.id,
            config_hash="sha256:config123",
            generation=1,
            fencing_token=1,
            status="serving",
            gpu_device="0",
            reserved_memory_mb=4096,
        )
        session.add(instance)
        session.flush()
        binding.current_runtime_instance_id = instance.id
        session.flush()

    return binding, release, model, instance


class TestInferenceRequestValidation:
    def test_valid_request_with_binding_id(self) -> None:
        req = InferenceRequest(
            modelBindingId=uuid4(),
            input={"image_base64": _make_test_image_base64(), "image_format": "png"},
        )
        assert req.modelBindingId is not None
        assert req.modelNodeId is None

    def test_valid_request_with_model_node_id(self) -> None:
        req = InferenceRequest(
            modelNodeId=uuid4(),
            input={"image_base64": _make_test_image_base64(), "image_format": "png"},
        )
        assert req.modelNodeId is not None
        assert req.modelBindingId is None

    def test_request_requires_either_binding_or_node(self) -> None:
        with pytest.raises(Exception):
            InferenceRequest(
                input={"image_base64": _make_test_image_base64(), "image_format": "png"},
            )

    def test_request_rejects_both_binding_and_node(self) -> None:
        with pytest.raises(Exception):
            InferenceRequest(
                modelBindingId=uuid4(),
                modelNodeId=uuid4(),
                input={"image_base64": _make_test_image_base64(), "image_format": "png"},
            )


class TestInferenceResponse:
    def test_response_fields(self) -> None:
        resp = InferenceResponse(
            results=[],
            latency=123.45,
            modelNodeId=uuid4(),
            bindingId=uuid4(),
            releaseId=uuid4(),
            generation=1,
        )
        assert resp.latency == 123.45
        assert resp.generation == 1
        assert resp.results == []


class TestInferenceService:
    def test_resolve_binding_by_id(self) -> None:
        from backend.app.services.inference_service import InferenceService

        session = _make_session()
        binding, release, model, instance = _seed_binding(session)
        service = InferenceService(session)

        resolved = service.resolve_binding(binding_id=binding.id)
        assert resolved is not None
        assert resolved.id == binding.id

    def test_resolve_binding_not_found(self) -> None:
        from backend.app.services.inference_service import InferenceService

        session = _make_session()
        service = InferenceService(session)
        resolved = service.resolve_binding(binding_id=uuid4())
        assert resolved is None

    def test_resolve_binding_by_model_node_id(self) -> None:
        from backend.app.services.inference_service import InferenceService

        session = _make_session()
        binding, release, model, instance = _seed_binding(session)
        service = InferenceService(session)

        resolved = service.resolve_binding(model_node_id=model.id)
        assert resolved is not None
        assert resolved.id == binding.id

    def test_validate_image_valid(self) -> None:
        from backend.app.services.inference_service import InferenceService

        session = _make_session()
        service = InferenceService(session)
        image_bytes = service.validate_image(
            _make_test_image_base64(), "png", max_size_bytes=10 * 1024 * 1024
        )
        assert len(image_bytes) > 0

    def test_validate_image_invalid_base64(self) -> None:
        from backend.app.services.inference_service import InferenceService

        session = _make_session()
        service = InferenceService(session)
        with pytest.raises(ValueError, match="Invalid base64"):
            service.validate_image("not-valid", "png", max_size_bytes=10 * 1024 * 1024)

    def test_validate_image_unsupported_format(self) -> None:
        from backend.app.services.inference_service import InferenceService

        session = _make_session()
        service = InferenceService(session)
        with pytest.raises(ValueError, match="Unsupported image format"):
            service.validate_image(_make_test_image_base64(), "gif", max_size_bytes=10 * 1024 * 1024)

    def test_validate_image_exceeds_size(self) -> None:
        from backend.app.services.inference_service import InferenceService

        session = _make_session()
        service = InferenceService(session)
        with pytest.raises(ValueError, match="exceeds maximum size"):
            service.validate_image(_make_test_image_base64(), "png", max_size_bytes=1)

    def test_run_inference_success(self) -> None:
        from backend.app.services.inference_service import InferenceService

        session = _make_session()
        binding, release, model, instance = _seed_binding(session)
        service = InferenceService(session)

        resp = service.run_inference(
            binding_id=binding.id,
            image_base64=_make_test_image_base64(),
            image_format="png",
        )
        assert resp.bindingId == binding.id
        assert resp.modelNodeId == model.id
        assert resp.releaseId == release.id
        assert resp.generation == 1
        assert resp.latency >= 0

    def test_run_inference_no_binding(self) -> None:
        from backend.app.services.inference_service import InferenceService

        session = _make_session()
        service = InferenceService(session)
        with pytest.raises(ValueError, match="Binding not found"):
            service.run_inference(
                binding_id=uuid4(),
                image_base64=_make_test_image_base64(),
                image_format="png",
            )

    def test_run_inference_no_serving_instance(self) -> None:
        from backend.app.services.inference_service import InferenceService

        session = _make_session()
        binding, release, model, instance = _seed_binding(session, has_runtime=False)
        service = InferenceService(session)
        with pytest.raises(ValueError, match="No serving instance"):
            service.run_inference(
                binding_id=binding.id,
                image_base64=_make_test_image_base64(),
                image_format="png",
            )

    def test_run_inference_instance_not_serving(self) -> None:
        from backend.app.services.inference_service import InferenceService

        session = _make_session()
        binding, release, model, instance = _seed_binding(session)
        instance.status = "loading"
        session.flush()
        service = InferenceService(session)
        with pytest.raises(ValueError, match="not serving"):
            service.run_inference(
                binding_id=binding.id,
                image_base64=_make_test_image_base64(),
                image_format="png",
            )

    def test_list_available_models(self) -> None:
        from backend.app.services.inference_service import InferenceService

        session = _make_session()
        _seed_binding(session)
        service = InferenceService(session)
        models = service.list_available_models()
        assert len(models) >= 1
        assert any(m.task_type == "object_detection" for m in models)

    def test_run_batch_inference(self) -> None:
        from backend.app.services.inference_service import InferenceService

        session = _make_session()
        binding, release, model, instance = _seed_binding(session)
        service = InferenceService(session)

        batch_resp = service.run_batch_inference(
            items=[
                InferenceItem(
                    modelBindingId=binding.id,
                    input={"image_base64": _make_test_image_base64(), "image_format": "png"},
                ),
                InferenceItem(
                    modelBindingId=binding.id,
                    input={"image_base64": _make_test_image_base64(), "image_format": "png"},
                ),
            ],
        )
        assert len(batch_resp.results) == 2
        assert batch_resp.total_latency >= 0
