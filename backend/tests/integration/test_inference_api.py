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
    binding = ModelBinding(external_ref="ref-ext-001", status="bound")
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
async def test_infer_by_binding_id(app, session, headers, serving_binding):
    binding, release, model, instance = serving_binding

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/infer",
            json={
                "modelBindingId": str(binding.id),
                "input": {
                    "image_base64": _make_test_image_base64(),
                    "image_format": "png",
                },
            },
            headers=headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["bindingId"] == str(binding.id)
    assert body["modelNodeId"] == str(model.id)
    assert body["releaseId"] == str(release.id)
    assert body["generation"] == 1
    assert body["latency"] >= 0
    assert len(body["results"]) == 1


@pytest.mark.anyio
async def test_infer_by_model_node_id(app, session, headers, serving_binding):
    binding, release, model, instance = serving_binding

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/infer",
            json={
                "modelNodeId": str(model.id),
                "input": {
                    "image_base64": _make_test_image_base64(),
                    "image_format": "png",
                },
            },
            headers=headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["bindingId"] == str(binding.id)
    assert body["modelNodeId"] == str(model.id)


@pytest.mark.anyio
async def test_infer_missing_binding_returns_404(app, session, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/infer",
            json={
                "modelBindingId": "00000000-0000-0000-0000-000000000000",
                "input": {
                    "image_base64": _make_test_image_base64(),
                    "image_format": "png",
                },
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
            "/api/v1/infer",
            json={
                "modelBindingId": str(binding.id),
                "input": {
                    "image_base64": _make_test_image_base64(),
                    "image_format": "png",
                },
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
            "/api/v1/infer",
            json={
                "modelBindingId": str(binding.id),
                "input": {
                    "image_base64": "not-a-valid-image",
                    "image_format": "png",
                },
            },
            headers=headers,
        )
    assert response.status_code == 400


@pytest.mark.anyio
async def test_infer_unauthorized_returns_401(app, session, serving_binding):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/infer",
            json={
                "modelBindingId": str(serving_binding[0].id),
                "input": {
                    "image_base64": _make_test_image_base64(),
                    "image_format": "png",
                },
            },
            headers={"X-API-Key": "wrong-key"},
        )
    assert response.status_code == 401


@pytest.mark.anyio
async def test_infer_batch(app, session, headers, serving_binding):
    binding, release, model, instance = serving_binding

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/infer/batch",
            json={
                "items": [
                    {
                        "modelBindingId": str(binding.id),
                        "input": {
                            "image_base64": _make_test_image_base64(),
                            "image_format": "png",
                        },
                    },
                    {
                        "modelBindingId": str(binding.id),
                        "input": {
                            "image_base64": _make_test_image_base64(),
                            "image_format": "png",
                        },
                    },
                ],
            },
            headers=headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["succeeded"] == 2
    assert body["failed"] == 0
    assert body["total_latency"] >= 0


@pytest.mark.anyio
async def test_infer_batch_partial_failure(app, session, headers, serving_binding):
    binding, release, model, instance = serving_binding
    fake_id = "00000000-0000-0000-0000-000000000000"

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/infer/batch",
            json={
                "items": [
                    {
                        "modelBindingId": str(binding.id),
                        "input": {
                            "image_base64": _make_test_image_base64(),
                            "image_format": "png",
                        },
                    },
                    {
                        "modelBindingId": fake_id,
                        "input": {
                            "image_base64": _make_test_image_base64(),
                            "image_format": "png",
                        },
                    },
                ],
            },
            headers=headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["succeeded"] == 1
    assert body["failed"] == 1


@pytest.mark.anyio
async def test_list_models(app, session, headers, serving_binding):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/infer/models", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) >= 1
    assert body[0]["task_type"] == "object_detection"
    assert body[0]["model_family"] == "yolo"
