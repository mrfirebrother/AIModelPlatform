from __future__ import annotations

import io
import os
import tarfile
import tempfile
import zipfile
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

import backend.app.api.routes.datasets
import backend.app.api.routes.models
from backend.app.config import Settings
from backend.app.api.dependencies import set_settings_override


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-api-key-123"}


@pytest.fixture()
def tmp_upload_dirs():
    with tempfile.TemporaryDirectory() as model_tmp, tempfile.TemporaryDirectory() as dataset_tmp:
        set_settings_override(
            Settings(
                model_dir=model_tmp,
                dataset_dir=dataset_tmp,
                platform_api_key="test-api-key-123",
            )
        )
        yield Path(model_tmp), Path(dataset_tmp)
        set_settings_override(None)


def _make_pt_payload(filename: str = "yolov8n.pt", size: int = 1024) -> bytes:
    return os.urandom(size)


def _make_zip_payload(filename: str = "dataset.zip") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("images/001.jpg", b"\xff\xd8\xff\xe0" + os.urandom(100))
        zf.writestr("labels/001.txt", "0 0.5 0.5 0.1 0.1")
    buf.seek(0)
    return buf.read()


def _make_tar_gz_payload(filename: str = "dataset.tar.gz") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="images/001.jpg")
        data = b"\xff\xd8\xff\xe0" + os.urandom(100)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        info2 = tarfile.TarInfo(name="labels/001.txt")
        data2 = b"0 0.5 0.5 0.1 0.1"
        info2.size = len(data2)
        tf.addfile(info2, io.BytesIO(data2))
    buf.seek(0)
    return buf.read()


# ── Model upload tests ──────────────────────────────────────────────

@pytest.mark.anyio
async def test_upload_model_pt(app, tmp_upload_dirs, headers):
    model_dir, _ = tmp_upload_dirs
    content = _make_pt_payload()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/models/upload",
            files={"file": ("yolov8n.pt", content, "application/octet-stream")},
            headers=headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "yolov8n.pt"
    assert body["file_path"].endswith(".pt")
    saved = Path(body["file_path"])
    assert saved.exists()
    assert saved.stat().st_size == len(content)


@pytest.mark.anyio
async def test_upload_model_rejects_non_pt(app, tmp_upload_dirs, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/models/upload",
            files={"file": ("model.txt", b"hello", "text/plain")},
            headers=headers,
        )
    assert response.status_code == 400
    assert "Only .pt files" in response.json()["detail"]


@pytest.mark.anyio
async def test_upload_model_rejects_too_large(app, tmp_upload_dirs, headers):
    _orig = backend.app.api.routes.models._MAX_MODEL_BYTES
    backend.app.api.routes.models._MAX_MODEL_BYTES = 1024  # 1 KB for test
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/models/upload",
                files={"file": ("huge.pt", os.urandom(2048), "application/octet-stream")},
                headers=headers,
            )
        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"].lower() or "limit" in response.json()["detail"].lower()
    finally:
        backend.app.api.routes.models._MAX_MODEL_BYTES = _orig


@pytest.mark.anyio
async def test_upload_model_unauthorized(app, tmp_upload_dirs):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/models/upload",
            files={"file": ("yolov8n.pt", b"\x00", "application/octet-stream")},
        )
    assert response.status_code == 401


# ── Dataset upload tests ────────────────────────────────────────────

@pytest.mark.anyio
async def test_upload_dataset_zip(app, tmp_upload_dirs, headers):
    _, dataset_dir = tmp_upload_dirs
    content = _make_zip_payload()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/datasets/upload",
            files={"file": ("dataset.zip", content, "application/zip")},
            headers=headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "dataset.zip"
    assert body["file_path"].endswith(".zip")
    saved = Path(body["file_path"])
    assert saved.exists()
    assert saved.stat().st_size == len(content)


@pytest.mark.anyio
async def test_upload_dataset_tar_gz(app, tmp_upload_dirs, headers):
    _, dataset_dir = tmp_upload_dirs
    content = _make_tar_gz_payload()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/datasets/upload",
            files={"file": ("dataset.tar.gz", content, "application/gzip")},
            headers=headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "dataset.tar.gz"
    assert body["file_path"].endswith(".tar.gz")
    saved = Path(body["file_path"])
    assert saved.exists()


@pytest.mark.anyio
async def test_upload_dataset_rejects_non_archive(app, tmp_upload_dirs, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/datasets/upload",
            files={"file": ("data.txt", b"hello", "text/plain")},
            headers=headers,
        )
    assert response.status_code == 400
    assert ".zip or .tar.gz" in response.json()["detail"]


@pytest.mark.anyio
async def test_upload_dataset_rejects_too_large(app, tmp_upload_dirs, headers):
    _orig = backend.app.api.routes.datasets._MAX_DATASET_BYTES
    backend.app.api.routes.datasets._MAX_DATASET_BYTES = 1024  # 1 KB for test
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/datasets/upload",
                files={"file": ("huge.zip", os.urandom(2048), "application/zip")},
                headers=headers,
            )
        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"].lower() or "limit" in response.json()["detail"].lower()
    finally:
        backend.app.api.routes.datasets._MAX_DATASET_BYTES = _orig


@pytest.mark.anyio
async def test_upload_dataset_unauthorized(app, tmp_upload_dirs):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/datasets/upload",
            files={"file": ("data.zip", b"\x00", "application/zip")},
        )
    assert response.status_code == 401
