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


@pytest.mark.anyio
async def test_delete_binding_tears_down_releases_and_instances(app, session, headers):
    """解绑下线：删除绑定会停服务实例、删发布记录、删绑定行。"""
    from backend.app.models import RuntimeInstance

    binding = ModelBinding(external_ref="ref-teardown", status="unbound")
    model = ModelNode(
        code="TX-BIND",
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/bind.pt",
        artifact_hash="sha256:bind",
        status="approved",
    )
    session.add(binding)
    session.add(model)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        rel = await client.post(
            "/api/releases",
            json={
                "binding_id": str(binding.id),
                "model_node_id": str(model.id),
                "inference_config_json": {"confidence_threshold": 0.1},
                "reason": "teardown test",
            },
            headers=headers,
        )
        assert rel.status_code == 201
        act = await client.post(
            f"/api/releases/{rel.json()['id']}/activate", headers=headers
        )
        assert act.status_code == 200

        # activation stands up a serving instance (binding_service.activate_release)
        session.expire_all()
        assert (
            session.execute(
                RuntimeInstance.__table__.select().where(
                    RuntimeInstance.__table__.c.binding_id == binding.id
                )
            ).scalars().first()
            is not None
        )

        resp = await client.delete(f"/api/bindings/{binding.id}", headers=headers)
        assert resp.status_code == 204

    session.expire_all()
    from sqlalchemy import select as _select

    assert session.get(ModelBinding, binding.id) is None
    releases = list(
        session.execute(
            _select(BindingRelease).where(BindingRelease.binding_id == binding.id)
        ).scalars().all()
    )
    assert releases == []
    instances = list(
        session.execute(
            _select(RuntimeInstance).where(RuntimeInstance.binding_id == binding.id)
        ).scalars().all()
    )
    assert instances == []
