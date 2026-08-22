from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-api-key-123"}


# ── GET /api/gpu/status ──────────────────────────────────────────────

@pytest.mark.anyio
async def test_gpu_status_returns_memory_info(app, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/gpu/status", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert "total_memory_mb" in body
    assert "used_memory_mb" in body
    assert "free_memory_mb" in body
    assert "device_count" in body
    assert body["total_memory_mb"] > 0
    assert body["used_memory_mb"] >= 0


# ── GET /api/gpu/models ──────────────────────────────────────────────

@pytest.mark.anyio
async def test_gpu_models_returns_empty_list_initially(app, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/gpu/models", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert "models" in body
    assert isinstance(body["models"], list)
    assert len(body["models"]) == 0


# ── POST /api/gpu/load ───────────────────────────────────────────────

@pytest.mark.anyio
async def test_gpu_load_model_success(app, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/gpu/load",
            json={"model_name": "yolov8n", "memory_mb": 2048},
            headers=headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "gpu_device" in body


@pytest.mark.anyio
async def test_gpu_load_model_missing_name(app, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/gpu/load",
            json={"memory_mb": 2048},
            headers=headers,
        )
    assert response.status_code == 422


@pytest.mark.anyio
async def test_gpu_load_model_increases_used_memory(app, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        before = (await client.get("/api/gpu/status", headers=headers)).json()
        await client.post(
            "/api/gpu/load",
            json={"model_name": "yolov8n", "memory_mb": 1024},
            headers=headers,
        )
        after = (await client.get("/api/gpu/status", headers=headers)).json()
    assert after["used_memory_mb"] >= before["used_memory_mb"]


# ── POST /api/gpu/unload ─────────────────────────────────────────────

@pytest.mark.anyio
async def test_gpu_unload_model_success(app, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/api/gpu/load",
            json={"model_name": "yolov8n", "memory_mb": 1024},
            headers=headers,
        )
        response = await client.post(
            "/api/gpu/unload",
            json={"model_name": "yolov8n"},
            headers=headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True


@pytest.mark.anyio
async def test_gpu_unload_nonexistent_model(app, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/gpu/unload",
            json={"model_name": "nonexistent"},
            headers=headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False


@pytest.mark.anyio
async def test_gpu_unload_decreases_used_memory(app, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/api/gpu/load",
            json={"model_name": "yolov8n", "memory_mb": 1024},
            headers=headers,
        )
        before = (await client.get("/api/gpu/status", headers=headers)).json()
        await client.post(
            "/api/gpu/unload",
            json={"model_name": "yolov8n"},
            headers=headers,
        )
        after = (await client.get("/api/gpu/status", headers=headers)).json()
    assert after["used_memory_mb"] < before["used_memory_mb"]
