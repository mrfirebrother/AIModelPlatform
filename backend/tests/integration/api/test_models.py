from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from backend.app.models import LabelSchema, ModelNode


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-api-key-123"}


@pytest.mark.anyio
async def test_create_model_node(app, session, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/models",
            json={
                "task_type": "object_detection",
                "model_family": "yolo",
                "artifact_path": "/data/models/test.pt",
                "artifact_hash": "sha256:abc123",
            },
            headers=headers,
        )
    assert response.status_code == 201
    body = response.json()
    assert body["task_type"] == "object_detection"
    assert body["model_family"] == "yolo"
    assert body["status"] == "candidate"


@pytest.mark.anyio
async def test_create_multiple_root_models(app, session, headers):
    """Multi-tree design: a second root model is allowed.

    Each application (rust segmentation, crack detection, bridge anomaly)
    imports its own pre-trained base and keeps its own lineage tree.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await client.post(
            "/api/models",
            json={
                "task_type": "object_detection",
                "model_family": "yolo",
                "artifact_path": "/data/models/detect.pt",
                "artifact_hash": "sha256:detect",
            },
            headers=headers,
        )
        second = await client.post(
            "/api/models",
            json={
                "task_type": "instance_segmentation",
                "model_family": "yolo",
                "artifact_path": "/data/models/segment.pt",
                "artifact_hash": "sha256:segment",
            },
            headers=headers,
        )
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["task_type"] == "instance_segmentation"

    roots = list(
        session.execute(
            select(ModelNode).where(ModelNode.parent_id.is_(None))
        ).scalars().all()
    )
    assert len(roots) == 2


@pytest.mark.anyio
async def test_list_model_nodes(app, session, headers):
    model = ModelNode(
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/test.pt",
        artifact_hash="sha256:abc123",
        status="candidate",
    )
    session.add(model)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/models", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1


@pytest.mark.anyio
async def test_get_model_node(app, session, headers):
    model = ModelNode(
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/test.pt",
        artifact_hash="sha256:abc123",
        status="candidate",
    )
    session.add(model)
    session.flush()
    model_id = str(model.id)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/models/{model_id}", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == model_id


@pytest.mark.anyio
async def test_update_model_status(app, session, headers):
    model = ModelNode(
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/test.pt",
        artifact_hash="sha256:abc123",
        status="candidate",
    )
    session.add(model)
    session.flush()
    model_id = str(model.id)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "approved"},
            headers=headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"


@pytest.mark.anyio
async def test_unauthorized_without_api_key(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/models")
    assert response.status_code == 401
