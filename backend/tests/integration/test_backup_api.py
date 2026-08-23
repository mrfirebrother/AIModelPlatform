from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from backend.app.models import Base
from backend.app.main import create_app
from backend.app.api.dependencies import get_db
from backend.app.api.routes.backup import get_backup_service
from backend.app.services.backup_service import BackupService


@pytest.fixture()
def backup_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def client(session: Session, backup_dir: Path) -> TestClient:
    application = create_app(checks={})

    def _get_db_override():
        yield session

    from backend.app.api.dependencies import verify_api_key
    from fastapi import Header, HTTPException, status as http_status

    def _verify(api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
        if api_key is None or api_key != "test-api-key-123":
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return api_key

    def _get_backup_svc() -> BackupService:
        return BackupService(backup_dir=backup_dir)

    application.dependency_overrides[get_db] = _get_db_override
    application.dependency_overrides[verify_api_key] = _verify
    application.dependency_overrides[get_backup_service] = _get_backup_svc
    return TestClient(application, raise_server_exceptions=True)


HEADERS = {"X-API-Key": "test-api-key-123"}


class TestBackupCreateAPI:
    def test_create_backup_returns_201(self, client: TestClient) -> None:
        resp = client.post(
            "/api/backup",
            json={"name": "api-backup-1", "backup_type": "full"},
            headers=HEADERS,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "api-backup-1"
        assert data["status"] == "completed"

    def test_create_backup_duplicate_name_returns_400(self, client: TestClient) -> None:
        client.post(
            "/api/backup",
            json={"name": "dup-api"},
            headers=HEADERS,
        )
        resp = client.post(
            "/api/backup",
            json={"name": "dup-api"},
            headers=HEADERS,
        )
        assert resp.status_code == 400
        assert "already exists" in resp.json()["detail"]

    def test_create_backup_unauthorized_returns_401(self, client: TestClient) -> None:
        resp = client.post(
            "/api/backup",
            json={"name": "no-auth"},
            headers={},
        )
        assert resp.status_code == 401


class TestBackupListAPI:
    def test_list_backups(self, client: TestClient) -> None:
        client.post("/api/backup", json={"name": "list-1"}, headers=HEADERS)
        client.post("/api/backup", json={"name": "list-2"}, headers=HEADERS)
        resp = client.get("/api/backup/list", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["backups"]) == 2


class TestBackupVerifyAPI:
    def test_verify_backup(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/backup", json={"name": "verify-api"}, headers=HEADERS
        )
        backup_id = create_resp.json()["id"]
        resp = client.post(f"/api/backup/{backup_id}/verify", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["verified"] is True

    def test_verify_nonexistent_returns_400(self, client: TestClient) -> None:
        fake_id = str(uuid4())
        resp = client.post(f"/api/backup/{fake_id}/verify", headers=HEADERS)
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"]


class TestBackupRestoreAPI:
    def test_restore_verified_backup(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/backup", json={"name": "restore-api"}, headers=HEADERS
        )
        backup_id = create_resp.json()["id"]
        client.post(f"/api/backup/{backup_id}/verify", headers=HEADERS)
        resp = client.post(
            "/api/backup/restore",
            json={"target_backup_id": backup_id, "restore_type": "full"},
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    def test_restore_unverified_without_force_returns_400(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/backup", json={"name": "restore-unverified-api"}, headers=HEADERS
        )
        backup_id = create_resp.json()["id"]
        resp = client.post(
            "/api/backup/restore",
            json={"target_backup_id": backup_id, "force": False},
            headers=HEADERS,
        )
        assert resp.status_code == 400
        assert "verified" in resp.json()["detail"]


class TestBackupDeleteAPI:
    def test_delete_backup(self, client: TestClient) -> None:
        create_resp = client.post(
            "/api/backup", json={"name": "del-api"}, headers=HEADERS
        )
        backup_id = create_resp.json()["id"]
        resp = client.delete(f"/api/backup/{backup_id}", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["name"] == "del-api"

    def test_delete_nonexistent_returns_404(self, client: TestClient) -> None:
        fake_id = str(uuid4())
        resp = client.delete(f"/api/backup/{fake_id}", headers=HEADERS)
        assert resp.status_code == 404
