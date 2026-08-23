from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.api.dependencies import set_db_override
from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.models import Alert, Base
from backend.app.observability.alerting import (
    GPU_LOW_MEMORY,
    TRAINING_FAILED,
    INFERENCE_ERROR_RATE,
    DISK_SPACE_LOW,
    SERVICE_UNAVAILABLE,
    ALL_RULES,
    acknowledge_alert,
    check_disk_space,
    check_gpu_memory,
    check_inference_error_rate,
    check_service_availability,
    check_training_failure,
    create_alert,
    get_active_alerts,
    get_alert_stats,
)


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


def _seed_alerts(session, count: int = 3, status: str = "active") -> list[Alert]:
    alerts = []
    for i in range(count):
        alert = Alert(
            alert_type=["gpu_memory_low", "training_failed", "inference_error_rate"][i % 3],
            severity=["critical", "warning"][i % 2],
            title=f"Test alert {i}",
            message=f"Test message {i}",
            resource_type="gpu",
            resource_id=f"gpu:{i}",
            status=status,
            metadata_json={"index": i},
            created_at=datetime(2026, 8, 20, 10, i, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 8, 20, 10, i, 0, tzinfo=timezone.utc),
        )
        session.add(alert)
        alerts.append(alert)
    session.commit()
    return alerts


# ── AlertRule constants ──

class TestAlertRules:
    def test_all_rules_list(self) -> None:
        assert len(ALL_RULES) == 5
        assert GPU_LOW_MEMORY in ALL_RULES
        assert TRAINING_FAILED in ALL_RULES
        assert INFERENCE_ERROR_RATE in ALL_RULES
        assert DISK_SPACE_LOW in ALL_RULES
        assert SERVICE_UNAVAILABLE in ALL_RULES

    def test_gpu_low_memory_rule(self) -> None:
        assert GPU_LOW_MEMORY.alert_type == "gpu_memory_low"
        assert GPU_LOW_MEMORY.severity == "critical"
        assert GPU_LOW_MEMORY.resource_type == "gpu"

    def test_training_failed_rule(self) -> None:
        assert TRAINING_FAILED.alert_type == "training_failed"
        assert TRAINING_FAILED.severity == "critical"

    def test_inference_error_rate_rule(self) -> None:
        assert INFERENCE_ERROR_RATE.alert_type == "inference_error_rate"
        assert INFERENCE_ERROR_RATE.severity == "warning"

    def test_disk_space_low_rule(self) -> None:
        assert DISK_SPACE_LOW.alert_type == "disk_space_low"
        assert DISK_SPACE_LOW.severity == "warning"

    def test_service_unavailable_rule(self) -> None:
        assert SERVICE_UNAVAILABLE.alert_type == "service_unavailable"
        assert SERVICE_UNAVAILABLE.severity == "critical"


# ── create_alert ──

class TestCreateAlert:
    def test_creates_alert_with_all_fields(self, db_session) -> None:
        alert = create_alert(
            db_session,
            alert_type="test",
            severity="warning",
            title="Test title",
            message="Test message",
            resource_type="gpu",
            resource_id="gpu:0",
            metadata={"key": "value"},
        )
        db_session.commit()
        assert alert.id is not None
        assert alert.alert_type == "test"
        assert alert.severity == "warning"
        assert alert.status == "active"
        assert alert.metadata_json == {"key": "value"}

    def test_creates_alert_minimal(self, db_session) -> None:
        alert = create_alert(
            db_session,
            alert_type="test",
            severity="info",
            title="Title",
            message="Message",
        )
        db_session.commit()
        assert alert.resource_type is None
        assert alert.resource_id is None
        assert alert.metadata_json == {}


# ── acknowledge_alert ──

class TestAcknowledgeAlert:
    def test_acknowledge_active_alert(self, db_session) -> None:
        alert = create_alert(
            db_session,
            alert_type="test",
            severity="critical",
            title="Title",
            message="Message",
        )
        db_session.commit()
        result = acknowledge_alert(db_session, alert.id, acknowledged_by="admin")
        assert result is not None
        assert result.status == "acknowledged"
        assert result.acknowledged_by == "admin"
        assert result.acknowledged_at is not None

    def test_acknowledge_nonexistent_returns_none(self, db_session) -> None:
        result = acknowledge_alert(db_session, uuid4())
        assert result is None

    def test_acknowledge_already_acknowledged_returns_none(self, db_session) -> None:
        alert = create_alert(
            db_session,
            alert_type="test",
            severity="critical",
            title="Title",
            message="Message",
        )
        db_session.commit()
        acknowledge_alert(db_session, alert.id)
        db_session.commit()
        result = acknowledge_alert(db_session, alert.id)
        assert result is None


# ── get_active_alerts ──

class TestGetActiveAlerts:
    def test_returns_active_only(self, db_session) -> None:
        _seed_alerts(db_session, count=5, status="active")
        _seed_alerts(db_session, count=2, status="acknowledged")
        active = get_active_alerts(db_session)
        assert len(active) == 5
        assert all(a.status == "active" for a in active)

    def test_filters_by_alert_type(self, db_session) -> None:
        _seed_alerts(db_session, count=3)
        gpu_alerts = get_active_alerts(db_session, alert_type="gpu_memory_low")
        assert all(a.alert_type == "gpu_memory_low" for a in gpu_alerts)

    def test_filters_by_severity(self, db_session) -> None:
        _seed_alerts(db_session, count=3)
        critical = get_active_alerts(db_session, severity="critical")
        assert all(a.severity == "critical" for a in critical)

    def test_empty_when_no_alerts(self, db_session) -> None:
        assert get_active_alerts(db_session) == []


# ── get_alert_stats ──

class TestGetAlertStats:
    def test_stats_empty(self, db_session) -> None:
        stats = get_alert_stats(db_session)
        assert stats["total"] == 0
        assert stats["active"] == 0
        assert stats["acknowledged"] == 0
        assert stats["by_severity"] == {}
        assert stats["by_type"] == {}

    def test_stats_with_alerts(self, db_session) -> None:
        _seed_alerts(db_session, count=3, status="active")
        _seed_alerts(db_session, count=2, status="acknowledged")
        stats = get_alert_stats(db_session)
        assert stats["total"] == 5
        assert stats["active"] == 3
        assert stats["acknowledged"] == 2


# ── check functions ──

class TestCheckGpuMemory:
    def test_alert_when_free_zero(self, db_session) -> None:
        alert = check_gpu_memory(db_session, device="0", free_mb=0, total_mb=8192)
        assert alert is not None
        assert alert.alert_type == "gpu_memory_low"
        assert alert.severity == "critical"

    def test_alert_when_low_free(self, db_session) -> None:
        alert = check_gpu_memory(db_session, device="0", free_mb=100, total_mb=8192, threshold_pct=10.0)
        assert alert is not None

    def test_no_alert_when_enough_memory(self, db_session) -> None:
        alert = check_gpu_memory(db_session, device="0", free_mb=4096, total_mb=8192)
        assert alert is None


class TestCheckTrainingFailure:
    def test_creates_failure_alert(self, db_session) -> None:
        alert = check_training_failure(
            db_session, task_id="t1", attempt_id="a1", error="CUDA OOM"
        )
        assert alert is not None
        assert alert.alert_type == "training_failed"
        assert alert.severity == "critical"
        assert "a1" in alert.message
        assert "CUDA OOM" in alert.message
        assert alert.resource_id == "t1"


class TestCheckInferenceErrorRate:
    def test_alert_when_above_threshold(self, db_session) -> None:
        alert = check_inference_error_rate(
            db_session, error_count=10, total_count=100, threshold_pct=5.0
        )
        assert alert is not None
        assert alert.alert_type == "inference_error_rate"

    def test_no_alert_when_below_threshold(self, db_session) -> None:
        alert = check_inference_error_rate(
            db_session, error_count=1, total_count=100, threshold_pct=5.0
        )
        assert alert is None

    def test_no_alert_when_no_requests(self, db_session) -> None:
        alert = check_inference_error_rate(db_session, error_count=0, total_count=0)
        assert alert is None


class TestCheckDiskSpace:
    def test_alert_when_low(self, db_session) -> None:
        alert = check_disk_space(
            db_session, path="/data", free_gb=1.0, total_gb=100.0, threshold_pct=10.0
        )
        assert alert is not None
        assert alert.alert_type == "disk_space_low"

    def test_no_alert_when_enough(self, db_session) -> None:
        alert = check_disk_space(
            db_session, path="/data", free_gb=50.0, total_gb=100.0, threshold_pct=10.0
        )
        assert alert is None


class TestCheckServiceAvailability:
    def test_alert_when_unhealthy(self, db_session) -> None:
        alert = check_service_availability(
            db_session, service="redis", is_healthy=False, details="timeout"
        )
        assert alert is not None
        assert alert.alert_type == "service_unavailable"

    def test_no_alert_when_healthy(self, db_session) -> None:
        alert = check_service_availability(
            db_session, service="redis", is_healthy=True
        )
        assert alert is None


# ── API endpoints ──

@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_list_alerts_returns_paginated(db_session) -> None:
    _seed_alerts(db_session, count=3)
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/alerts",
            headers={"X-API-Key": Settings().platform_api_key},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert len(body["items"]) == 3


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_list_alerts_requires_api_key(db_session) -> None:
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/api/alerts")
    assert resp.status_code == 401


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_list_alerts_filters_by_status(db_session) -> None:
    _seed_alerts(db_session, count=3, status="active")
    _seed_alerts(db_session, count=2, status="acknowledged")
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/alerts?status=acknowledged",
            headers={"X-API-Key": Settings().platform_api_key},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert all(item["status"] == "acknowledged" for item in body["items"])


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_list_alerts_filters_by_type(db_session) -> None:
    _seed_alerts(db_session, count=3)
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/alerts?type=gpu_memory_low",
            headers={"X-API-Key": Settings().platform_api_key},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert all(item["alertType"] == "gpu_memory_low" for item in body["items"])


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_acknowledge_alert_success(db_session) -> None:
    alerts = _seed_alerts(db_session, count=1, status="active")
    alert_id = str(alerts[0].id)
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post(
            f"/api/alerts/{alert_id}/acknowledge",
            headers={"X-API-Key": Settings().platform_api_key},
            json={"acknowledgedBy": "admin"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "acknowledged"
    assert body["acknowledgedBy"] == "admin"


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_acknowledge_alert_not_found(db_session) -> None:
    fake_id = str(uuid4())
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post(
            f"/api/alerts/{fake_id}/acknowledge",
            headers={"X-API-Key": Settings().platform_api_key},
        )
    assert resp.status_code == 404


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_acknowledge_already_acknowledged(db_session) -> None:
    alerts = _seed_alerts(db_session, count=1, status="acknowledged")
    alert_id = str(alerts[0].id)
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.post(
            f"/api/alerts/{alert_id}/acknowledge",
            headers={"X-API-Key": Settings().platform_api_key},
        )
    assert resp.status_code == 404


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_alert_stats_empty(db_session) -> None:
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/alerts/stats",
            headers={"X-API-Key": Settings().platform_api_key},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["active"] == 0
    assert body["acknowledged"] == 0


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_alert_stats_with_data(db_session) -> None:
    _seed_alerts(db_session, count=3, status="active")
    _seed_alerts(db_session, count=2, status="acknowledged")
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get(
            "/api/alerts/stats",
            headers={"X-API-Key": Settings().platform_api_key},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 5
    assert body["active"] == 3
    assert body["acknowledged"] == 2
    assert isinstance(body["bySeverity"], dict)
    assert isinstance(body["byType"], dict)


@pytest.mark.anyio
@pytest.mark.usefixtures("_override_db")
async def test_alert_stats_requires_api_key(db_session) -> None:
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        resp = await client.get("/api/alerts/stats")
    assert resp.status_code == 401
