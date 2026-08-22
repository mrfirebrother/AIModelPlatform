from __future__ import annotations

import base64

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.models import BindingRelease, ModelBinding, ModelNode, RuntimeInstance


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-api-key-123"}


@pytest.fixture()
def serving_binding(session):
    binding = ModelBinding(external_ref="ref-001", status="bound")
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


def _make_test_image_base64() -> str:
    png_data = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return base64.b64encode(png_data).decode("utf-8")


@pytest.mark.anyio
async def test_infer_returns_placeholder_response(app, session, headers, serving_binding):
    binding, release, model, instance = serving_binding

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/infer",
            json={
                "binding_id": str(binding.id),
                "image_base64": _make_test_image_base64(),
                "image_format": "png",
            },
            headers=headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["modelBindingId"] == str(binding.id)
    assert body["release"]["id"] == str(release.id)
    assert body["model"]["id"] == str(model.id)
    assert body["model"]["task_type"] == "object_detection"
    assert body["generation"] == 1
    assert body["configHash"] == "sha256:config123"
    assert body["detections"] == []


@pytest.mark.anyio
async def test_infer_missing_binding_returns_404(app, session, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/infer",
            json={
                "binding_id": "00000000-0000-0000-0000-000000000000",
                "image_base64": _make_test_image_base64(),
                "image_format": "png",
            },
            headers=headers,
        )
    assert response.status_code == 404


@pytest.mark.anyio
async def test_infer_no_serving_instance_returns_503(app, session, headers):
    binding = ModelBinding(external_ref="ref-noserve", status="unbound")
    session.add(binding)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/infer",
            json={
                "binding_id": str(binding.id),
                "image_base64": _make_test_image_base64(),
                "image_format": "png",
            },
            headers=headers,
        )
    assert response.status_code == 503


@pytest.mark.anyio
async def test_infer_invalid_image_returns_400(app, session, headers, serving_binding):
    binding, _, _, _ = serving_binding

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/infer",
            json={
                "binding_id": str(binding.id),
                "image_base64": "not-a-valid-image",
                "image_format": "png",
            },
            headers=headers,
        )
    assert response.status_code == 400
