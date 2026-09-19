from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from backend.app.models import LabelSchema, ModelNode


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-api-key-123"}


@pytest.mark.anyio
async def test_create_model_node(app, session, headers):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/models",
            json={
                "task_type": "object_detection",
                "model_family": "yolo",
                "artifact_path": "/data/models/test.pt",
                "artifact_hash": "sha256:abc123",
            },
            headers=headers,
        )
    assert response.status_code == 201
    body = response.json()
    assert body["task_type"] == "object_detection"
    assert body["model_family"] == "yolo"
    assert body["status"] == "candidate"


@pytest.mark.anyio
async def test_create_multiple_root_models(app, session, headers):
    """Multi-tree design: a second root model is allowed.

    Each application (rust segmentation, crack detection, bridge anomaly)
    imports its own pre-trained base and keeps its own lineage tree.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await client.post(
            "/api/models",
            json={
                "task_type": "object_detection",
                "model_family": "yolo",
                "artifact_path": "/data/models/detect.pt",
                "artifact_hash": "sha256:detect",
            },
            headers=headers,
        )
        second = await client.post(
            "/api/models",
            json={
                "task_type": "instance_segmentation",
                "model_family": "yolo",
                "artifact_path": "/data/models/segment.pt",
                "artifact_hash": "sha256:segment",
            },
            headers=headers,
        )
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["task_type"] == "instance_segmentation"

    roots = list(
        session.execute(
            select(ModelNode).where(ModelNode.parent_id.is_(None))
        ).scalars().all()
    )
    assert len(roots) == 2


@pytest.mark.anyio
async def test_delete_model_clears_task_parent_reference(app, session, headers):
    """Deleting a model that trained children must clear the tasks' parent
    pointer instead of failing with the raw FK 400 (reported bug)."""
    from backend.app.models import Dataset, DatasetSnapshot, LabelSchema, TrainingTask

    parent = ModelNode(
        code="TX-PARENT",
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/parent.pt",
        artifact_hash="sha256:parent",
        status="candidate",
    )
    session.add(parent)
    session.flush()
    schema = LabelSchema(name="tx-schema")
    session.add(schema)
    session.flush()
    ds = Dataset(name="tx-ds")
    session.add(ds)
    session.flush()
    snap = DatasetSnapshot(
        dataset_id=ds.id,
        label_schema_id=schema.id,
        manifest_path="/data/store/tx.json",
        manifest_hash="sha256:tx",
    )
    session.add(snap)
    session.flush()
    task = TrainingTask(
        parent_model_node_id=parent.id,
        dataset_snapshot_id=snap.id,
        task_type="object_detection",
        model_family="yolo",
        status="completed",
    )
    session.add(task)
    session.commit()
    task_id = task.id
    parent_id = parent.id
    # Detach the committed objects: production API requests run in a session
    # that never loaded these rows, while this test session would otherwise
    # hold expired copies the immutability guard rejects refreshing after the
    # route's out-of-band UPDATE.
    session.expunge(task)
    session.expunge(parent)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.delete(f"/api/models/{parent_id}", headers=headers)
    assert response.status_code == 200
    session.expire_all()
    assert session.get(TrainingTask, task_id).parent_model_node_id is None


@pytest.mark.anyio
async def test_delete_model_with_active_release_is_refused(app, session, headers):
    """A model referenced by an ACTIVE release (live traffic) cannot be deleted."""
    from backend.app.models import BindingRelease, ModelBinding

    model = ModelNode(
        code="TX-REL",
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/rel.pt",
        artifact_hash="sha256:rel",
        status="approved",
    )
    session.add(model)
    session.flush()
    binding = ModelBinding(external_ref="tx-ref")
    session.add(binding)
    session.flush()
    release = BindingRelease(
        binding_id=binding.id,
        model_node_id=model.id,
        revision_no=1,
        inference_config_json={},
        inference_config_hash="sha256:cfg",
        release_type="normal",
        status="active",
    )
    session.add(release)
    session.commit()
    model_id = model.id
    session.expunge(model)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.delete(f"/api/models/{model_id}", headers=headers)
    assert response.status_code == 400
    assert "active release" in response.json()["detail"].lower()


@pytest.mark.anyio
async def test_delete_model_clears_historical_release_reference(app, session, headers):
    """Historical (non-active) release rows keep their history but lose the model
    pointer, so the model can be deleted."""
    from backend.app.models import BindingRelease, ModelBinding

    model = ModelNode(
        code="TX-HIST",
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/hist.pt",
        artifact_hash="sha256:hist",
        status="approved",
    )
    session.add(model)
    session.flush()
    binding = ModelBinding(external_ref="tx-hist-ref")
    session.add(binding)
    session.flush()
    release = BindingRelease(
        binding_id=binding.id,
        model_node_id=model.id,
        revision_no=1,
        inference_config_json={},
        inference_config_hash="sha256:cfg",
        release_type="normal",
        status="superseded",
    )
    session.add(release)
    session.commit()
    model_id = model.id
    release_id = release.id
    session.expunge(model)
    session.expunge(release)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.delete(f"/api/models/{model_id}", headers=headers)
    assert response.status_code == 200
    session.expire_all()
    assert session.get(BindingRelease, release_id).model_node_id is None


@pytest.mark.anyio
async def test_list_model_nodes(app, session, headers):
    model = ModelNode(
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/test.pt",
        artifact_hash="sha256:abc123",
        status="candidate",
    )
    session.add(model)
    session.flush()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/models", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1


@pytest.mark.anyio
async def test_get_model_node(app, session, headers):
    model = ModelNode(
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/test.pt",
        artifact_hash="sha256:abc123",
        status="candidate",
    )
    session.add(model)
    session.flush()
    model_id = str(model.id)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/api/models/{model_id}", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == model_id


@pytest.mark.anyio
async def test_update_model_status(app, session, headers):
    model = ModelNode(
        task_type="object_detection",
        model_family="yolo",
        artifact_path="/data/models/test.pt",
        artifact_hash="sha256:abc123",
        status="candidate",
    )
    session.add(model)
    session.flush()
    model_id = str(model.id)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.patch(
            f"/api/models/{model_id}/status",
            json={"status": "approved"},
            headers=headers,
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"


@pytest.mark.anyio
async def test_unauthorized_without_api_key(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/models")
    assert response.status_code == 401
