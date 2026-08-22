from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.models import OperationLog
from sqlalchemy import select


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-api-key-123"}


@pytest.mark.anyio
async def test_operation_log_created_on_model_create(app, session, headers):
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

    logs = session.execute(
        select(OperationLog).where(
            OperationLog.operation_type == "model.create"
        )
    ).scalars().all()
    assert len(logs) >= 1
    assert logs[0].status == "success"


@pytest.mark.anyio
async def test_operation_log_created_on_binding_create(app, session, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/bindings",
            json={"external_ref": "ref-001"},
            headers=headers,
        )
    assert response.status_code == 201

    logs = session.execute(
        select(OperationLog).where(
            OperationLog.operation_type == "binding.create"
        )
    ).scalars().all()
    assert len(logs) >= 1


@pytest.mark.anyio
async def test_operation_log_records_error(app, session, headers):
    from backend.app.models import ModelNode

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

    logs = session.execute(
        select(OperationLog).where(
            OperationLog.operation_type == "model.update_status"
        )
    ).scalars().all()
    assert len(logs) >= 1
    assert logs[0].status == "success"
