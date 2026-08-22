from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import create_app


@pytest.mark.anyio
async def test_health_endpoint():
    app = create_app(
        checks={
            "postgres": lambda: True,
            "redis": lambda: True,
            "gpu": lambda: True,
            "worker": lambda: True,
        }
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"


@pytest.mark.anyio
async def test_liveness_endpoint():
    app = create_app(checks={})
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
