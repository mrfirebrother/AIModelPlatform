from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.models import Dataset, DatasetSnapshot, Evaluation, LabelSchema, ModelNode


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-api-key-123"}


@pytest.fixture()
def model_and_snapshot(session):
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
    model = ModelNode(
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/test.pt",
        artifact_hash="sha256:abc123",
        status="candidate",
    )
    session.add(snapshot)
    session.add(model)
    session.flush()
    return model, snapshot


@pytest.mark.anyio
async def test_submit_evaluation(app, session, headers, model_and_snapshot):
    model, snapshot = model_and_snapshot
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/evaluations",
            json={
                "model_node_id": str(model.id),
                "dataset_snapshot_id": str(snapshot.id),
                "policy": {
                    "min_mAP50": 0.5,
                    "min_precision": 0.6,
                    "min_recall": 0.5,
                    "max_regression_ratio": 0.1,
                    "require_per_class_coverage": True,
                },
            },
            headers=headers,
        )
    assert response.status_code == 201
    body = response.json()
    assert body["auto_status"] == "pending"


@pytest.mark.anyio
async def test_get_evaluation(app, session, headers, model_and_snapshot):
    model, snapshot = model_and_snapshot
    ev = Evaluation(
        model_node_id=model.id,
        dataset_snapshot_id=snapshot.id,
        auto_status="pending",
        human_status="pending",
        evaluation_policy_json={},
        auto_metrics_json={},
    )
    session.add(ev)
    session.flush()
    ev_id = str(ev.id)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/evaluations/{ev_id}", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == ev_id


@pytest.mark.anyio
async def test_list_evaluations(app, session, headers, model_and_snapshot):
    model, snapshot = model_and_snapshot
    ev = Evaluation(
        model_node_id=model.id,
        dataset_snapshot_id=snapshot.id,
        auto_status="pending",
        human_status="pending",
        evaluation_policy_json={},
        auto_metrics_json={},
    )
    session.add(ev)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/evaluations", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
