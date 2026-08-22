from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.models import Dataset, DatasetSnapshot, LabelSchema, TrainingTask


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-api-key-123"}


@pytest.fixture()
def dataset_snapshot(session):
    schema = LabelSchema(name="test-schema", status="active")
    dataset = Dataset(name="test-dataset", description="test")
    session.add(schema)
    session.add(dataset)
    session.flush()
    snapshot = DatasetSnapshot(
        dataset_id=dataset.id,
        label_schema_id=schema.id,
        manifest_path="/data/manifest.json",
        manifest_hash="sha256:test",
        train_manifest_json=[],
        val_manifest_json=[],
        test_manifest_json=[],
    )
    session.add(snapshot)
    session.flush()
    return snapshot


@pytest.mark.anyio
async def test_create_training_task(app, session, headers, dataset_snapshot):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/training/tasks",
            json={
                "dataset_snapshot_id": str(dataset_snapshot.id),
                "task_type": "object_detection",
                "model_family": "yolo",
                "training_config_json": {"epochs": 100},
                "resource_config_json": {"gpu_device": "0"},
                "evaluation_policy_json": {"min_mAP50": 0.5},
            },
            headers=headers,
        )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "queued"
    assert body["task_type"] == "object_detection"


@pytest.mark.anyio
async def test_list_training_tasks(app, session, headers, dataset_snapshot):
    task = TrainingTask(
        dataset_snapshot_id=dataset_snapshot.id,
        task_type="object_detection",
        model_family="yolo",
        training_config_json={},
        resource_config_json={},
        evaluation_policy_json={},
        status="queued",
    )
    session.add(task)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/training/tasks", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1


@pytest.mark.anyio
async def test_get_training_task(app, session, headers, dataset_snapshot):
    task = TrainingTask(
        dataset_snapshot_id=dataset_snapshot.id,
        task_type="object_detection",
        model_family="yolo",
        training_config_json={},
        resource_config_json={},
        evaluation_policy_json={},
        status="queued",
    )
    session.add(task)
    session.flush()
    task_id = str(task.id)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/training/tasks/{task_id}", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == task_id
