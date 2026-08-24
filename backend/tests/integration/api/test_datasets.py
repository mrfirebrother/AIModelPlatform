from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.models import Dataset, DatasetSnapshot, LabelSchema


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-api-key-123"}


@pytest.mark.anyio
async def test_create_dataset(app, session, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/datasets",
            json={
                "name": "test-dataset",
                "description": "A test dataset",
            },
            headers=headers,
        )
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "test-dataset"


@pytest.mark.anyio
async def test_create_dataset_accepts_list_names_yaml(app, session, headers, tmp_path):
    from backend.app.api.dependencies import get_effective_settings
    from backend.app.config import Settings

    source = tmp_path / "dataset"
    (source / "images" / "train").mkdir(parents=True)
    (source / "images" / "val").mkdir(parents=True)
    (source / "labels" / "train").mkdir(parents=True)
    (source / "labels" / "val").mkdir(parents=True)
    (source / "data.yaml").write_text(
        "nc: 1\nnames: ['corrosion']\n"
        "train: images/train\nval: images/val\n",
        encoding="utf-8",
    )
    app.dependency_overrides[get_effective_settings] = lambda: Settings(
        dataset_dir=str(tmp_path / "storage")
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/datasets",
            json={"name": "list-names-dataset", "source_path": str(source)},
            headers=headers,
        )

    assert response.status_code == 201


@pytest.mark.anyio
async def test_list_datasets(app, session, headers):
    dataset = Dataset(name="existing-dataset", description="test")
    session.add(dataset)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/datasets", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1


@pytest.mark.anyio
async def test_list_datasets_includes_latest_snapshot_statistics(app, session, headers):
    schema = LabelSchema(name="corrosion-schema", status="active")
    dataset = Dataset(
        name="imported-dataset",
        description="test",
        source_path="/data/datasets/imported.zip",
    )
    session.add_all([schema, dataset])
    session.flush()
    session.add(
        DatasetSnapshot(
            dataset_id=dataset.id,
            label_schema_id=schema.id,
            manifest_path="/data/snapshots/imported/manifest.json",
            manifest_hash="sha256:test",
            train_manifest_json=[],
            val_manifest_json=[],
            test_manifest_json=[],
            train_positive_count=2,
            val_positive_count=1,
            test_positive_count=1,
        )
    )
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/datasets", headers=headers)

    assert response.status_code == 200
    item = next(item for item in response.json()["datasets"] if item["id"] == str(dataset.id))
    assert item["label_schema_name"] == "corrosion-schema"
    assert item["image_count"] == 4
    assert item["train_count"] == 2
    assert item["val_count"] == 1
    assert item["test_count"] == 1
    assert item["latest_snapshot_id"]


@pytest.mark.anyio
async def test_create_snapshot(app, session, headers):
    schema = LabelSchema(name="test-schema", status="active")
    dataset = Dataset(name="test-dataset", description="test")
    session.add(schema)
    session.add(dataset)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/datasets/snapshots",
            json={
                "dataset_id": str(dataset.id),
                "label_schema_id": str(schema.id),
                "source_path": "/data/datasets/test",
            },
            headers=headers,
        )
    assert response.status_code == 201
    body = response.json()
    assert body["dataset_id"] == str(dataset.id)
    assert body["label_schema_id"] == str(schema.id)
