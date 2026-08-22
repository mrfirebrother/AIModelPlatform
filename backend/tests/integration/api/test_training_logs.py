from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.models import (
    Checkpoint,
    Dataset,
    DatasetSnapshot,
    LabelSchema,
    TrainingAttempt,
    TrainingTask,
)


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
async def test_get_training_logs_returns_task_and_attempts(
    app, session, headers, dataset_snapshot
):
    task = TrainingTask(
        dataset_snapshot_id=dataset_snapshot.id,
        task_type="object_detection",
        model_family="yolo",
        training_config_json={"epochs": 100},
        resource_config_json={},
        evaluation_policy_json={},
        status="running",
    )
    session.add(task)
    session.flush()

    attempt = TrainingAttempt(
        task_id=task.id,
        attempt_no=1,
        status="running",
        current_epoch=10,
        log_path="/logs/task1/attempt1.log",
    )
    session.add(attempt)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/training/{task.id}/logs", headers=headers
        )
    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == str(task.id)
    assert body["status"] == "running"
    assert body["current_epoch"] == 10
    assert body["total_epochs"] == 100
    assert len(body["attempts"]) == 1
    assert body["attempts"][0]["attempt_no"] == 1
    assert body["attempts"][0]["status"] == "running"


@pytest.mark.anyio
async def test_get_training_logs_404_for_nonexistent_task(app, session, headers):
    from uuid import uuid4

    fake_id = uuid4()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/training/{fake_id}/logs", headers=headers
        )
    assert response.status_code == 404


@pytest.mark.anyio
async def test_get_training_metrics_returns_loss_curve(
    app, session, headers, dataset_snapshot
):
    task = TrainingTask(
        dataset_snapshot_id=dataset_snapshot.id,
        task_type="object_detection",
        model_family="yolo",
        training_config_json={"epochs": 100},
        resource_config_json={},
        evaluation_policy_json={},
        status="running",
    )
    session.add(task)
    session.flush()

    attempt = TrainingAttempt(
        task_id=task.id,
        attempt_no=1,
        status="running",
        current_epoch=5,
    )
    session.add(attempt)
    session.flush()

    for ep in range(1, 6):
        ckpt = Checkpoint(
            attempt_id=attempt.id,
            parent_artifact_hash="sha256:parent",
            dataset_snapshot_id=dataset_snapshot.id,
            training_config_hash="sha256:config",
            epoch=ep,
            artifact_path=f"/artifacts/epoch_{ep}.pt",
            artifact_hash=f"sha256:ep{ep}",
            metrics_json={"loss": 1.0 - ep * 0.1, "lr": 0.001},
        )
        session.add(ckpt)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/training/{task.id}/metrics", headers=headers
        )
    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == str(task.id)
    assert len(body["epochs"]) == 5
    assert body["epochs"][0]["epoch"] == 1
    assert body["epochs"][0]["loss"] == pytest.approx(0.9)
    assert body["epochs"][4]["loss"] == pytest.approx(0.5)
    assert "best_loss" in body
    assert body["best_loss"] == pytest.approx(0.5)


@pytest.mark.anyio
async def test_get_training_metrics_404_for_nonexistent_task(app, session, headers):
    from uuid import uuid4

    fake_id = uuid4()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/training/{fake_id}/metrics", headers=headers
        )
    assert response.status_code == 404


@pytest.mark.anyio
async def test_get_training_checkpoint_returns_checkpoint_status(
    app, session, headers, dataset_snapshot
):
    task = TrainingTask(
        dataset_snapshot_id=dataset_snapshot.id,
        task_type="object_detection",
        model_family="yolo",
        training_config_json={"epochs": 100},
        resource_config_json={},
        evaluation_policy_json={},
        status="running",
    )
    session.add(task)
    session.flush()

    attempt = TrainingAttempt(
        task_id=task.id,
        attempt_no=1,
        status="running",
        current_epoch=5,
    )
    session.add(attempt)
    session.flush()

    ckpt = Checkpoint(
        attempt_id=attempt.id,
        parent_artifact_hash="sha256:parent",
        dataset_snapshot_id=dataset_snapshot.id,
        training_config_hash="sha256:config",
        epoch=5,
        artifact_path="/artifacts/epoch_5.pt",
        artifact_hash="sha256:ep5",
        metrics_json={"loss": 0.5, "mAP50": 0.8},
    )
    session.add(ckpt)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/api/training/{task.id}/checkpoint", headers=headers
        )
    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == str(task.id)
    assert body["attempt_no"] == 1
    assert body["latest_checkpoint"] is not None
    assert body["latest_checkpoint"]["epoch"] == 5
    assert body["latest_checkpoint"]["artifact_path"] == "/artifacts/epoch_5.pt"
    assert body["latest_checkpoint"]["metrics"]["loss"] == 0.5
    assert body["latest_checkpoint"]["metrics"]["mAP50"] == 0.8


@pytest.mark.anyio
async def test_get_training_checkpoint_returns_null_when_no_checkpoints(
    app, session, headers, dataset_snapshot
):
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
        response = await client.get(
            f"/api/training/{task.id}/checkpoint", headers=headers
        )
    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == str(task.id)
    assert body["latest_checkpoint"] is None
    assert body["attempt_no"] is None
