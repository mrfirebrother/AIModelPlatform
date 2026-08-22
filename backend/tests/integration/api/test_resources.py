from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.models import GPUResource


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-api-key-123"}


@pytest.mark.anyio
async def test_list_gpu_resources(app, session, headers):
    resource = GPUResource(
        gpu_device="0",
        capacity_memory_mb=8192,
        reserved_memory_mb=0,
    )
    session.add(resource)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/resources/gpu", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert len(body["resources"]) >= 1
    assert body["resources"][0]["gpu_device"] == "0"


@pytest.mark.anyio
async def test_get_gpu_resource(app, session, headers):
    resource = GPUResource(
        gpu_device="0",
        capacity_memory_mb=8192,
        reserved_memory_mb=0,
    )
    session.add(resource)
    session.flush()
    resource_id = str(resource.id)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/resources/gpu/{resource_id}", headers=headers
        )
    assert response.status_code == 200
    body = response.json()
    assert body["gpu_device"] == "0"
    assert body["capacity_memory_mb"] == 8192


@pytest.mark.anyio
async def test_list_residency_plans(app, session, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/resources/residency", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert "plans" in body
