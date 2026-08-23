from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.api.dependencies import set_db_override
from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.models import Base, OperationLog


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def _override_db(db_session):
    set_db_override(lambda: iter([db_session]))
    yield
    set_db_override(None)


def _seed_logs(session, count: int = 5, offset_days: int = 0) -> list[OperationLog]:
    logs = []
    for i in range(count):
        log = OperationLog(
            operation_type="training.create" if i % 2 == 0 else "binding.update",
            actor="admin",
            request_id=uuid4(),
            resource_type="training_task" if i % 2 == 0 else "binding",
            resource_id=uuid4(),
            status="success" if i % 3 != 0 else "error",
            summary_json={"detail": f"test log {i}"},
            error_summary="connection timeout" if i % 3 == 0 else None,
            created_at=datetime(2026, 8, 20, 10, i, 0, tzinfo=timezone.utc)
            + timedelta(days=offset_days),
        )
        session.add(log)
        logs.append(log)
    session.commit()
    return logs


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_get_operation_logs_returns_paginated_list(db_session) -> None:
    _seed_logs(db_session, count=3)
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/operations/logs",
            headers={"X-API-Key": Settings().platform_api_key},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert "total" in body
    assert body["total"] == 3
    assert len(body["items"]) == 3


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_get_operation_logs_supports_pagination(db_session) -> None:
    _seed_logs(db_session, count=5)
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/operations/logs?skip=2&limit=2",
            headers={"X-API-Key": Settings().platform_api_key},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 5
    assert len(body["items"]) == 2


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_get_operation_logs_filters_by_time_range(db_session) -> None:
    _seed_logs(db_session, count=3, offset_days=0)
    _seed_logs(db_session, count=2, offset_days=10)
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/operations/logs?start=2026-08-21T00:00:00Z",
            headers={"X-API-Key": Settings().platform_api_key},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_get_operation_logs_empty_when_no_data(db_session) -> None:
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/operations/logs",
            headers={"X-API-Key": Settings().platform_api_key},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["items"] == []


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_get_operation_errors_returns_only_errors(db_session) -> None:
    _seed_logs(db_session, count=5)
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/operations/errors",
            headers={"X-API-Key": Settings().platform_api_key},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    for item in body["items"]:
        assert item["status"] == "error"
        assert item["errorSummary"] is not None


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_get_operation_status_returns_system_health(db_session) -> None:
    app = create_app(
        checks={
            "postgres": lambda: True,
            "redis": lambda: True,
            "gpu": lambda: True,
            "worker": lambda: True,
        }
    )
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/operations/status",
            headers={"X-API-Key": Settings().platform_api_key},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert "checks" in body
    assert body["checks"]["postgres"] is True


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_get_operation_status_requires_api_key(db_session) -> None:
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/api/operations/status")
    assert resp.status_code == 401
