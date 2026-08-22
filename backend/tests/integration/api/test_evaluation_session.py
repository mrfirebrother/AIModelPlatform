from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-api-key-123"}


@pytest.mark.anyio
async def test_create_evaluation_session(app, session, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/evaluations/sessions",
            json={
                "model_node_id": "00000000-0000-0000-0000-000000000001",
                "expires_in_seconds": 3600,
            },
            headers=headers,
        )
    assert response.status_code == 201
    body = response.json()
    assert "session_id" in body
    assert body["expires_in_seconds"] == 3600


@pytest.mark.anyio
async def test_create_evaluation_session_default_expiry(app, session, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/evaluations/sessions",
            json={
                "model_node_id": "00000000-0000-0000-0000-000000000001",
            },
            headers=headers,
        )
    assert response.status_code == 201
    body = response.json()
    assert body["expires_in_seconds"] == 3600
