"""Tests for the single-password UI gate middleware (main.py: ui_password_gate)."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.config import Settings
from backend.app.main import create_app


@pytest.fixture()
def gated_app():
    return create_app(settings=Settings(ui_password="secret"), checks={})


@pytest.mark.anyio
async def test_api_requires_ui_password(gated_app):
    async with AsyncClient(
        transport=ASGITransport(app=gated_app), base_url="http://test"
    ) as client:
        no_header = await client.get("/api/models")
        wrong = await client.get(
            "/api/models", headers={"X-UI-Password": "nope"}
        )
    assert no_header.status_code == 401
    assert "password" in no_header.json()["detail"].lower()
    assert wrong.status_code == 401


@pytest.mark.anyio
async def test_health_routes_are_exempt(gated_app):
    async with AsyncClient(
        transport=ASGITransport(app=gated_app), base_url="http://test"
    ) as client:
        live = await client.get("/health/live")
        health = await client.get("/health")
    # 看门狗和部署脚本探活不能带密码头
    assert live.status_code == 200
    assert health.status_code == 200


@pytest.mark.anyio
async def test_v1_service_api_uses_api_key_not_ui_password(gated_app):
    async with AsyncClient(
        transport=ASGITransport(app=gated_app), base_url="http://test"
    ) as client:
        # 无 X-API-Key → 401 API key（密码门放行，说明 v1 豁免生效）
        resp = await client.post(
            "/api/v1/infer",
            json={"modelCode": "M001", "input": {"image_base64": "x", "image_format": "png"}},
        )
    assert resp.status_code == 401
    assert "API key" in resp.json()["detail"]


@pytest.mark.anyio
async def test_gate_disabled_when_no_password_configured():
    app = create_app(settings=Settings(ui_password=None), checks={})
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health/live")
        # 未配置密码时门禁关闭；/api/models 的 401 只会来自 API key 校验
        unauthenticated = await client.get("/api/models")
    assert resp.status_code == 200
    assert unauthenticated.status_code == 401
    assert "API key" in unauthenticated.json()["detail"]
