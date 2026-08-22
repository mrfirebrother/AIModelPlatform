from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.models import BindingRelease, ModelBinding, ModelNode


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-api-key-123"}


@pytest.mark.anyio
async def test_create_binding(app, session, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/bindings",
            json={"external_ref": "ref-001"},
            headers=headers,
        )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "unbound"
    assert body["external_ref"] == "ref-001"


@pytest.mark.anyio
async def test_list_bindings(app, session, headers):
    binding = ModelBinding(external_ref="ref-001", status="unbound")
    session.add(binding)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/bindings", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1


@pytest.mark.anyio
async def test_create_release(app, session, headers):
    binding = ModelBinding(external_ref="ref-001", status="unbound")
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

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/releases",
            json={
                "binding_id": str(binding.id),
                "model_node_id": str(model.id),
                "inference_config_json": {"confidence": 0.5},
            },
            headers=headers,
        )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    assert body["release_type"] == "normal"


@pytest.mark.anyio
async def test_rollback_release(app, session, headers):
    binding = ModelBinding(external_ref="ref-001", status="unbound")
    model1 = ModelNode(
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/test1.pt",
        artifact_hash="sha256:abc123",
        status="approved",
    )
    model2 = ModelNode(
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/test2.pt",
        artifact_hash="sha256:def456",
        status="approved",
    )
    session.add(binding)
    session.add(model1)
    session.add(model2)
    session.flush()

    release1 = BindingRelease(
        binding_id=binding.id,
        revision_no=1,
        model_node_id=model1.id,
        inference_config_json={},
        inference_config_hash="sha256:empty",
        release_type="normal",
        status="active",
    )
    session.add(release1)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/releases/rollback",
            json={
                "binding_id": str(binding.id),
                "target_release_id": str(release1.id),
                "model_node_id": str(model2.id),
                "reason": "Performance regression",
            },
            headers=headers,
        )
    assert response.status_code == 201
    body = response.json()
    assert body["release_type"] == "rollback"
    assert body["rollback_target_release_id"] == str(release1.id)
