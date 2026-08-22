from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.models import Dataset, LabelSchema


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
