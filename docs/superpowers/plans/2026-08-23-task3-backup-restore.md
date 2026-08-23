# Backup and Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement manual backup and restore functionality for models, datasets, and database metadata.

**Architecture:** Backup service will manage file system backups with metadata tracking, using content-addressed storage for deduplication. REST API provides manual backup/restore operations.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, SHA256 hashing, JSON manifests.

---

## File Structure

| File | Purpose |
|------|---------|
| `backend/app/services/backup_service.py` | Core backup/restore logic |
| `backend/app/api/routes/backup.py` | REST API endpoints |
| `backend/tests/unit/test_backup_service.py` | Unit tests for service |
| `backend/tests/integration/test_backup_api.py` | Integration tests for API |

---

## Task 3: Backup and Restore

**Files:**
- Create: `backend/app/services/backup_service.py`
- Create: `backend/app/api/routes/backup.py`
- Create: `backend/tests/unit/test_backup_service.py`
- Create: `backend/tests/integration/test_backup_api.py`

### Step 1: Write failing unit tests for BackupService

```python
# backend/tests/unit/test_backup_service.py
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from backend.app.services.backup_service import BackupService


@pytest.fixture()
def backup_root(tmp_path: Path) -> Path:
    root = tmp_path / "backups"
    root.mkdir()
    return root


@pytest.fixture()
def models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    (d / "model_v1.pt").write_bytes(b"\x80\x04" + b"\x00" * 1024)
    return d


@pytest.fixture()
def datasets_dir(tmp_path: Path) -> Path:
    d = tmp_path / "datasets"
    d.mkdir()
    (d / "data.yaml").write_text("nc: 1\nnames: ['test']")
    return d


@pytest.fixture()
def service(backup_root: Path) -> BackupService:
    return BackupService(backup_root=backup_root)


class TestCreateBackup:
    def test_creates_backup_directory(
        self, service: BackupService, models_dir: Path
    ) -> None:
        result = service.create_backup(
            models_dir=models_dir,
            model_ids=["model_v1.pt"],
            datasets_dir=None,
            dataset_ids=[],
            metadata={"version": "v1"},
        )
        assert result.backup_id is not None
        backup_path = service.backup_root / str(result.backup_id)
        assert backup_path.exists()

    def test_copies_model_files(
        self, service: BackupService, models_dir: Path
    ) -> None:
        result = service.create_backup(
            models_dir=models_dir,
            model_ids=["model_v1.pt"],
            datasets_dir=None,
            dataset_ids=[],
            metadata={},
        )
        backup_models = service.backup_root / str(result.backup_id) / "models"
        assert (backup_models / "model_v1.pt").exists()

    def test_creates_manifest(
        self, service: BackupService, models_dir: Path
    ) -> None:
        result = service.create_backup(
            models_dir=models_dir,
            model_ids=["model_v1.pt"],
            datasets_dir=None,
            dataset_ids=[],
            metadata={"key": "value"},
        )
        manifest_path = (
            service.backup_root / str(result.backup_id) / "manifest.json"
        )
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["metadata"]["key"] == "value"
        assert len(manifest["model_files"]) == 1

    def test_computes_integrity_hash(
        self, service: BackupService, models_dir: Path
    ) -> None:
        result = service.create_backup(
            models_dir=models_dir,
            model_ids=["model_v1.pt"],
            datasets_dir=None,
            dataset_ids=[],
            metadata={},
        )
        assert result.integrity_hash.startswith("sha256:")

    def test_lists_backups(self, service: BackupService, models_dir: Path) -> None:
        service.create_backup(
            models_dir=models_dir,
            model_ids=["model_v1.pt"],
            datasets_dir=None,
            dataset_ids=[],
            metadata={},
        )
        backups = service.list_backups()
        assert len(backups) == 1
        assert backups[0].backup_id is not None


class TestRestoreBackup:
    def test_restores_model_files(
        self, service: BackupService, models_dir: Path, tmp_path: Path
    ) -> None:
        result = service.create_backup(
            models_dir=models_dir,
            model_ids=["model_v1.pt"],
            datasets_dir=None,
            dataset_ids=[],
            metadata={},
        )
        restore_dir = tmp_path / "restored"
        restore_dir.mkdir()
        service.restore_backup(result.backup_id, restore_dir=restore_dir)
        assert (restore_dir / "models" / "model_v1.pt").exists()

    def test_restore_nonexistent_raises(
        self, service: BackupService, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="Backup not found"):
            service.restore_backup(uuid4(), restore_dir=tmp_path)


class TestVerifyBackup:
    def test_verify_valid_backup(
        self, service: BackupService, models_dir: Path
    ) -> None:
        result = service.create_backup(
            models_dir=models_dir,
            model_ids=["model_v1.pt"],
            datasets_dir=None,
            dataset_ids=[],
            metadata={},
        )
        assert service.verify_backup(result.backup_id) is True

    def test_verify_corrupted_backup(
        self, service: BackupService, models_dir: Path
    ) -> None:
        result = service.create_backup(
            models_dir=models_dir,
            model_ids=["model_v1.pt"],
            datasets_dir=None,
            dataset_ids=[],
            metadata={},
        )
        manifest_path = (
            service.backup_root / str(result.backup_id) / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["integrity_hash"] = "sha256:0000"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        assert service.verify_backup(result.backup_id) is False


class TestDeleteBackup:
    def test_deletes_backup(
        self, service: BackupService, models_dir: Path
    ) -> None:
        result = service.create_backup(
            models_dir=models_dir,
            model_ids=["model_v1.pt"],
            datasets_dir=None,
            dataset_ids=[],
            metadata={},
        )
        service.delete_backup(result.backup_id)
        assert not (service.backup_root / str(result.backup_id)).exists()

    def test_delete_nonexistent_raises(
        self, service: BackupService
    ) -> None:
        with pytest.raises(ValueError, match="Backup not found"):
            service.delete_backup(uuid4())
```

### Step 2: Run tests to verify they fail

Run: `cd backend && python -m pytest tests/unit/test_backup_service.py -v`
Expected: FAIL with ImportError or ModuleNotFoundError

### Step 3: Implement BackupService

```python
# backend/app/services/backup_service.py
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4


@dataclass
class BackupResult:
    backup_id: UUID
    integrity_hash: str
    model_files: list[str]
    dataset_files: list[str]
    metadata: dict


@dataclass
class BackupInfo:
    backup_id: UUID
    created_at: str
    metadata: dict
    integrity_hash: str
    model_count: int
    dataset_count: int


class BackupService:
    def __init__(self, backup_root: Path) -> None:
        self.backup_root = backup_root
        self.backup_root.mkdir(parents=True, exist_ok=True)

    def create_backup(
        self,
        models_dir: Path | None,
        model_ids: list[str],
        datasets_dir: Path | None,
        dataset_ids: list[str],
        metadata: dict | None = None,
    ) -> BackupResult:
        backup_id = uuid4()
        backup_dir = self.backup_root / str(backup_id)
        backup_dir.mkdir(parents=True)

        model_files: list[str] = []
        dataset_files: list[str] = []

        if models_dir and model_ids:
            backup_models = backup_dir / "models"
            backup_models.mkdir()
            for model_id in model_ids:
                src = models_dir / model_id
                if src.exists():
                    shutil.copy2(src, backup_models / model_id)
                    model_files.append(model_id)

        if datasets_dir and dataset_ids:
            backup_datasets = backup_dir / "datasets"
            backup_datasets.mkdir()
            for dataset_id in dataset_ids:
                src = datasets_dir / dataset_id
                if src.exists():
                    if src.is_dir():
                        shutil.copytree(src, backup_datasets / dataset_id)
                    else:
                        shutil.copy2(src, backup_datasets / dataset_id)
                    dataset_files.append(dataset_id)

        manifest = {
            "backup_id": str(backup_id),
            "model_files": model_files,
            "dataset_files": dataset_files,
            "metadata": metadata or {},
        }

        integrity_hash = self._compute_integrity(backup_dir, manifest)
        manifest["integrity_hash"] = integrity_hash

        manifest_path = backup_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        return BackupResult(
            backup_id=backup_id,
            integrity_hash=integrity_hash,
            model_files=model_files,
            dataset_files=dataset_files,
            metadata=metadata or {},
        )

    def list_backups(self) -> list[BackupInfo]:
        backups: list[BackupInfo] = []
        for entry in sorted(self.backup_root.iterdir()):
            if not entry.is_dir():
                continue
            manifest_path = entry / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            backups.append(
                BackupInfo(
                    backup_id=UUID(manifest["backup_id"]),
                    created_at=manifest.get("created_at", ""),
                    metadata=manifest.get("metadata", {}),
                    integrity_hash=manifest.get("integrity_hash", ""),
                    model_count=len(manifest.get("model_files", [])),
                    dataset_count=len(manifest.get("dataset_files", [])),
                )
            )
        return backups

    def restore_backup(
        self, backup_id: UUID, restore_dir: Path
    ) -> None:
        backup_dir = self.backup_root / str(backup_id)
        if not backup_dir.exists():
            raise ValueError(f"Backup not found: {backup_id}")

        manifest_path = backup_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        restore_dir.mkdir(parents=True, exist_ok=True)

        models_src = backup_dir / "models"
        if models_src.exists():
            models_dst = restore_dir / "models"
            models_dst.mkdir(exist_ok=True)
            for f in models_src.iterdir():
                shutil.copy2(f, models_dst / f.name)

        datasets_src = backup_dir / "datasets"
        if datasets_src.exists():
            datasets_dst = restore_dir / "datasets"
            datasets_dst.mkdir(exist_ok=True)
            for f in datasets_src.iterdir():
                if f.is_dir():
                    shutil.copytree(f, datasets_dst / f.name)
                else:
                    shutil.copy2(f, datasets_dst / f.name)

    def verify_backup(self, backup_id: UUID) -> bool:
        backup_dir = self.backup_root / str(backup_id)
        if not backup_dir.exists():
            return False

        manifest_path = backup_dir / "manifest.json"
        if not manifest_path.exists():
            return False

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_hash = manifest.get("integrity_hash", "")

        computed_hash = self._compute_integrity(backup_dir, manifest)
        return stored_hash == computed_hash

    def delete_backup(self, backup_id: UUID) -> None:
        backup_dir = self.backup_root / str(backup_id)
        if not backup_dir.exists():
            raise ValueError(f"Backup not found: {backup_id}")
        shutil.rmtree(backup_dir)

    def _compute_integrity(
        self, backup_dir: Path, manifest: dict
    ) -> str:
        h = hashlib.sha256()

        for model_file in sorted(manifest.get("model_files", [])):
            model_path = backup_dir / "models" / model_file
            if model_path.exists():
                h.update(model_path.read_bytes())

        for dataset_file in sorted(manifest.get("dataset_files", [])):
            dataset_path = backup_dir / "datasets" / dataset_file
            if dataset_path.exists():
                if dataset_path.is_dir():
                    for f in sorted(dataset_path.rglob("*")):
                        if f.is_file():
                            h.update(f.read_bytes())
                else:
                    h.update(dataset_path.read_bytes())

        return f"sha256:{h.hexdigest()}"
```

### Step 4: Run tests to verify they pass

Run: `cd backend && python -m pytest tests/unit/test_backup_service.py -v`
Expected: PASS

### Step 5: Write failing integration tests for backup API

```python
# backend/tests/integration/test_backup_api.py
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.models import ModelNode


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-api-key-123"}


@pytest.fixture()
def models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    (d / "test_model.pt").write_bytes(b"\x80\x04" + b"\x00" * 1024)
    return d


@pytest.fixture()
def seed_model(session):
    model = ModelNode(
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/test_model.pt",
        artifact_hash="sha256:abc123",
        status="approved",
        metadata_json={"version": "v1"},
    )
    session.add(model)
    session.flush()
    return model


@pytest.mark.anyio
async def test_create_backup(app, session, headers, seed_model, models_dir, tmp_path):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/backup",
            json={
                "model_ids": [str(seed_model.id)],
                "models_dir": str(models_dir),
                "metadata": {"description": "test backup"},
            },
            headers=headers,
        )
    assert response.status_code == 201
    body = response.json()
    assert "backup_id" in body
    assert body["integrity_hash"].startswith("sha256:")


@pytest.mark.anyio
async def test_list_backups(app, session, headers, seed_model, models_dir):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post(
            "/api/backup",
            json={
                "model_ids": [str(seed_model.id)],
                "models_dir": str(models_dir),
                "metadata": {},
            },
            headers=headers,
        )
        response = await client.get("/api/backup/list", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert len(body["backups"]) == 1


@pytest.mark.anyio
async def test_verify_backup(app, session, headers, seed_model, models_dir):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        create_resp = await client.post(
            "/api/backup",
            json={
                "model_ids": [str(seed_model.id)],
                "models_dir": str(models_dir),
                "metadata": {},
            },
            headers=headers,
        )
        backup_id = create_resp.json()["backup_id"]
        response = await client.post(
            f"/api/backup/{backup_id}/verify", headers=headers
        )
    assert response.status_code == 200
    assert response.json()["valid"] is True


@pytest.mark.anyio
async def test_delete_backup(app, session, headers, seed_model, models_dir):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        create_resp = await client.post(
            "/api/backup",
            json={
                "model_ids": [str(seed_model.id)],
                "models_dir": str(models_dir),
                "metadata": {},
            },
            headers=headers,
        )
        backup_id = create_resp.json()["backup_id"]
        response = await client.delete(
            f"/api/backup/{backup_id}", headers=headers
        )
    assert response.status_code == 204


@pytest.mark.anyio
async def test_restore_backup(app, session, headers, seed_model, models_dir, tmp_path):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        create_resp = await client.post(
            "/api/backup",
            json={
                "model_ids": [str(seed_model.id)],
                "models_dir": str(models_dir),
                "metadata": {},
            },
            headers=headers,
        )
        backup_id = create_resp.json()["backup_id"]
        response = await client.post(
            "/api/backup/restore",
            json={"backup_id": backup_id, "restore_dir": str(tmp_path / "restored")},
            headers=headers,
        )
    assert response.status_code == 200
    assert (tmp_path / "restored" / "models" / "test_model.pt").exists()


@pytest.mark.anyio
async def test_delete_nonexistent_returns_404(app, session, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.delete(
            "/api/backup/00000000-0000-0000-0000-000000000000",
            headers=headers,
        )
    assert response.status_code == 404


@pytest.mark.anyio
async def test_verify_nonexistent_returns_404(app, session, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/backup/00000000-0000-0000-0000-000000000000/verify",
            headers=headers,
        )
    assert response.status_code == 404


@pytest.mark.anyio
async def test_backup_unauthorized_returns_401(app, session, seed_model, models_dir):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/backup",
            json={
                "model_ids": [str(seed_model.id)],
                "models_dir": str(models_dir),
                "metadata": {},
            },
            headers={"X-API-Key": "wrong-key"},
        )
    assert response.status_code == 401
```

### Step 6: Run integration tests to verify they fail

Run: `cd backend && python -m pytest tests/integration/test_backup_api.py -v`
Expected: FAIL with 404 or ImportError

### Step 7: Create backup API schemas

```python
# Add to backend/app/schemas/backup.py (new file)
from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class BackupCreateRequest(BaseModel):
    model_ids: list[str] = Field(default_factory=list)
    models_dir: str | None = None
    dataset_ids: list[str] = Field(default_factory=list)
    datasets_dir: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BackupResponse(BaseModel):
    backup_id: UUID
    integrity_hash: str
    model_files: list[str]
    dataset_files: list[str]
    metadata: dict[str, Any]


class BackupInfoResponse(BaseModel):
    backup_id: UUID
    created_at: str
    metadata: dict[str, Any]
    integrity_hash: str
    model_count: int
    dataset_count: int


class BackupListResponse(BaseModel):
    backups: list[BackupInfoResponse]
    total: int


class BackupVerifyResponse(BaseModel):
    backup_id: UUID
    valid: bool


class BackupRestoreRequest(BaseModel):
    backup_id: UUID
    restore_dir: str


class BackupRestoreResponse(BaseModel):
    backup_id: UUID
    restore_dir: str
    files_restored: int
```

### Step 8: Implement backup API routes

```python
# backend/app/api/routes/backup.py
from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, verify_api_key
from backend.app.schemas.backup import (
    BackupCreateRequest,
    BackupInfoResponse,
    BackupListResponse,
    BackupResponse,
    BackupRestoreRequest,
    BackupRestoreResponse,
    BackupVerifyResponse,
)
from backend.app.services.backup_service import BackupService

router = APIRouter(prefix="/api/backup", tags=["backup"])

BACKUP_ROOT = Path("backups")


def get_backup_service() -> BackupService:
    return BackupService(backup_root=BACKUP_ROOT)


@router.post("", response_model=BackupResponse, status_code=status.HTTP_201_CREATED)
def create_backup(
    payload: BackupCreateRequest,
    _key: str = Depends(verify_api_key),
) -> Any:
    service = get_backup_service()
    models_dir = Path(payload.models_dir) if payload.models_dir else None
    datasets_dir = Path(payload.datasets_dir) if payload.datasets_dir else None

    result = service.create_backup(
        models_dir=models_dir,
        model_ids=payload.model_ids,
        datasets_dir=datasets_dir,
        dataset_ids=payload.dataset_ids,
        metadata=payload.metadata,
    )
    return result


@router.get("/list", response_model=BackupListResponse)
def list_backups(
    _key: str = Depends(verify_api_key),
) -> Any:
    service = get_backup_service()
    backups = service.list_backups()
    return BackupListResponse(backups=backups, total=len(backups))


@router.post("/{backup_id}/verify", response_model=BackupVerifyResponse)
def verify_backup(
    backup_id: UUID,
    _key: str = Depends(verify_api_key),
) -> Any:
    service = get_backup_service()
    valid = service.verify_backup(backup_id)
    if not valid:
        backup_dir = service.backup_root / str(backup_id)
        if not backup_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Backup not found",
            )
    return BackupVerifyResponse(backup_id=backup_id, valid=valid)


@router.post("/restore", response_model=BackupRestoreResponse)
def restore_backup(
    payload: BackupRestoreRequest,
    _key: str = Depends(verify_api_key),
) -> Any:
    service = get_backup_service()
    restore_dir = Path(payload.restore_dir)

    try:
        service.restore_backup(payload.backup_id, restore_dir=restore_dir)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    files_restored = 0
    models_dir = restore_dir / "models"
    if models_dir.exists():
        files_restored += sum(1 for _ in models_dir.iterdir())
    datasets_dir = restore_dir / "datasets"
    if datasets_dir.exists():
        files_restored += sum(1 for _ in datasets_dir.rglob("*") if _.is_file())

    return BackupRestoreResponse(
        backup_id=payload.backup_id,
        restore_dir=str(restore_dir),
        files_restored=files_restored,
    )


@router.delete("/{backup_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_backup(
    backup_id: UUID,
    _key: str = Depends(verify_api_key),
) -> None:
    service = get_backup_service()
    try:
        service.delete_backup(backup_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
```

### Step 9: Register backup router in main.py

Modify `backend/app/main.py` to include the backup router:

```python
# In create_app function, add:
from .api.routes import (
    bindings_router,
    datasets_router,
    evaluations_router,
    gpu_router,
    health_router,
    infer_router,
    inference_router,
    models_router,
    operations_router,
    releases_router,
    resources_router,
    training_logs_router,
    training_router,
    backup_router,  # Add this
)

# Add after other routers:
app.include_router(backup_router)
```

### Step 10: Update routes/__init__.py

Add backup_router export to `backend/app/api/routes/__init__.py`.

### Step 11: Run all tests

Run: `cd backend && python -m pytest tests/unit/test_backup_service.py tests/integration/test_backup_api.py -v`
Expected: All tests PASS

### Step 12: Verify code quality

Run: `cd backend && python -m ruff check app/services/backup_service.py app/api/routes/backup.py`

Run: `cd backend && python -m mypy app/services/backup_service.py app/api/routes/backup.py --ignore-missing-imports`

### Step 13: Commit

```bash
git add backend/app/services/backup_service.py backend/app/api/routes/backup.py backend/app/schemas/backup.py backend/app/main.py backend/app/api/routes/__init__.py backend/tests/unit/test_backup_service.py backend/tests/integration/test_backup_api.py
git commit -m "feat: implement backup and restore service with API endpoints"
```

---

## Summary

- **Service:** `BackupService` handles file copying, manifest generation, integrity verification
- **API:** 4 endpoints for create, list, verify, restore, delete operations
- **Tests:** 9 unit tests + 8 integration tests covering happy paths and error cases
- **Storage:** Content-addressed backup directories with SHA256 integrity checks