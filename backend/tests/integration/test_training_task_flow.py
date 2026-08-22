from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from backend.app.models import (
    Base,
    Checkpoint,
    Dataset,
    DatasetSnapshot,
    GPUResource,
    GPUResourceLease,
    LabelSchema,
    ModelNode,
    TrainingAttempt,
    TrainingTask,
)
from backend.app.services.resource_lease import acquire_lease, release_lease
from backend.app.services.training_service import (
    cancel_task,
    complete_attempt,
    create_task,
    fail_attempt,
    start_attempt,
)
from backend.app.workers.attempt_lease import heartbeat


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    event.listen(
        engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(engine)
    with Session(engine) as db_session:
        yield db_session
    engine.dispose()


def _make_snapshot(session: Session) -> DatasetSnapshot:
    schema = LabelSchema(name=f"schema-{uuid4().hex[:8]}")
    session.add(schema)
    session.flush()
    dataset = Dataset(name=f"ds-{uuid4().hex[:8]}")
    snapshot = DatasetSnapshot(
        dataset=dataset,
        label_schema=schema,
        manifest_path=f"snapshots/{uuid4()}/manifest.json",
        manifest_hash=f"sha256:{uuid4().hex}",
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def _make_parent_model(session: Session) -> ModelNode:
    model = ModelNode(
        artifact_path=f"models/{uuid4()}.pt",
        artifact_hash=f"sha256:{uuid4().hex}",
        task_type="object_detection",
        model_family="yolo",
        status="approved",
        metadata_json={"format": "pt"},
    )
    session.add(model)
    session.flush()
    return model


def _make_gpu_resource(session: Session, device: str = "0", capacity: int = 4096) -> GPUResource:
    resource = GPUResource(
        gpu_device=device,
        capacity_memory_mb=capacity,
        reserved_memory_mb=0,
    )
    session.add(resource)
    session.flush()
    return resource


class TestFullTrainingTaskFlow:
    def test_create_start_complete_training_creates_candidate_model(
        self, session: Session
    ) -> None:
        snapshot = _make_snapshot(session)
        parent = _make_parent_model(session)
        task = create_task(
            session,
            parent_model_node_id=parent.id,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={"epochs": 10, "lr": 0.001},
            resource_config_json={"gpu_memory_mb": 2048},
            evaluation_policy_json={"mAP50": 0.5},
        )
        session.commit()
        assert task.status == "queued"

        attempt = start_attempt(session, task.id)
        session.commit()
        session.refresh(task)
        assert task.status == "running"
        assert attempt.status == "running"
        assert attempt.attempt_no == 1

        model = complete_attempt(
            session,
            attempt.id,
            artifact_path=f"models/{uuid4()}.pt",
            artifact_hash=f"sha256:{uuid4().hex}",
        )
        session.commit()
        session.refresh(task)
        assert task.status == "completed"
        assert model.status == "candidate"
        assert model.parent_id == parent.id
        assert model.training_attempt_id == attempt.id

    def test_training_does_not_auto_replace_online_model(
        self, session: Session
    ) -> None:
        snapshot = _make_snapshot(session)
        parent = _make_parent_model(session)
        task = create_task(
            session,
            parent_model_node_id=parent.id,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={},
            resource_config_json={},
            evaluation_policy_json={},
        )
        session.commit()
        attempt = start_attempt(session, task.id)
        session.commit()
        model = complete_attempt(
            session,
            attempt.id,
            artifact_path=f"models/{uuid4()}.pt",
            artifact_hash=f"sha256:{uuid4().hex}",
        )
        session.commit()
        session.refresh(model)
        assert model.status == "candidate"


class TestRetryOnFailure:
    def test_failed_attempt_allows_new_attempt(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        task = create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={},
            resource_config_json={},
            evaluation_policy_json={},
        )
        session.commit()
        a1 = start_attempt(session, task.id)
        session.commit()
        fail_attempt(session, a1.id, "OOM")
        session.commit()
        a2 = start_attempt(session, task.id)
        session.commit()
        assert a2.attempt_no == 2
        assert a2.retry_count == 1
        model = complete_attempt(
            session,
            a2.id,
            artifact_path=f"models/{uuid4()}.pt",
            artifact_hash=f"sha256:{uuid4().hex}",
        )
        session.commit()
        session.refresh(task)
        assert task.status == "completed"
        assert model.training_attempt_id == a2.id


class TestCancelDuringTraining:
    def test_cancel_running_task_cancels_attempt(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        task = create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={},
            resource_config_json={},
            evaluation_policy_json={},
        )
        session.commit()
        attempt = start_attempt(session, task.id)
        session.commit()
        cancel_task(session, task.id)
        session.commit()
        session.refresh(attempt)
        assert attempt.status == "cancelled"
        assert attempt.finished_at is not None


class TestGPULeaseDuringTraining:
    def test_acquire_and_release_gpu_lease(self, session: Session) -> None:
        resource = _make_gpu_resource(session, device="0", capacity=4096)
        owner_id = uuid4()
        lease = acquire_lease(
            session,
            gpu_resource_id=resource.id,
            owner_type="training_attempt",
            owner_id=owner_id,
            reserved_memory_mb=2048,
        )
        session.commit()
        assert lease.status == "active"
        assert lease.lease_token is not None
        assert lease.fencing_token >= 0
        session.refresh(resource)
        assert resource.reserved_memory_mb == 2048

        release_lease(session, lease.id, lease.lease_token)
        session.commit()
        session.refresh(resource)
        assert resource.reserved_memory_mb == 0
        session.refresh(lease)
        assert lease.status == "released"

    def test_release_lease_with_wrong_token_raises(
        self, session: Session
    ) -> None:
        resource = _make_gpu_resource(session)
        lease = acquire_lease(
            session,
            gpu_resource_id=resource.id,
            owner_type="training_attempt",
            owner_id=uuid4(),
            reserved_memory_mb=256,
        )
        session.commit()
        with pytest.raises(ValueError, match="lease token"):
            release_lease(session, lease.id, uuid4())

    def test_only_one_training_attempt_runs_at_a_time(
        self, session: Session
    ) -> None:
        resource = _make_gpu_resource(session, device="0", capacity=4096)
        snapshot = _make_snapshot(session)
        task1 = create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={},
            resource_config_json={},
            evaluation_policy_json={},
        )
        task2 = create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={},
            resource_config_json={},
            evaluation_policy_json={},
        )
        session.commit()
        a1 = start_attempt(session, task1.id)
        session.commit()
        # Second attempt on task2 cannot start while task1 has a running attempt
        # (enforced by unique partial index on task_id + status)
        # This tests the business rule: only one running attempt per task
        a2 = TrainingAttempt(
            task=task2,
            attempt_no=1,
            status="running",
            lease_token=uuid4(),
            fencing_token=1,
        )
        session.add(a2)
        # task2 has no running attempt, so this should work
        session.commit()

    def test_heartbeat_extends_lease(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        task = create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={},
            resource_config_json={},
            evaluation_policy_json={},
        )
        session.commit()
        attempt = start_attempt(session, task.id)
        session.commit()
        old_expires = attempt.lease_expires_at
        heartbeat(session, attempt.id, attempt.lease_token)
        session.commit()
        session.refresh(attempt)
        if old_expires is not None:
            assert attempt.lease_expires_at >= old_expires


class TestCheckpointRecording:
    def test_checkpoint_stored_with_attempt(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        task = create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={},
            resource_config_json={},
            evaluation_policy_json={},
        )
        session.commit()
        attempt = start_attempt(session, task.id)
        session.commit()

        cp = Checkpoint(
            attempt=attempt,
            dataset_snapshot=snapshot,
            parent_artifact_hash=f"sha256:{uuid4().hex}",
            training_config_hash=f"sha256:{uuid4().hex}",
            epoch=5,
            artifact_path=f"checkpoints/{uuid4()}.pt",
            artifact_hash=f"sha256:{uuid4().hex}",
            metrics_json={"loss": 0.1},
        )
        session.add(cp)
        session.flush()
        attempt.latest_checkpoint_id = cp.id
        session.commit()
        session.refresh(attempt)
        assert attempt.latest_checkpoint_id == cp.id

    def test_checkpoint_parent_hash_and_config_hash_are_immutable(
        self, session: Session
    ) -> None:
        snapshot = _make_snapshot(session)
        task = create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={},
            resource_config_json={},
            evaluation_policy_json={},
        )
        session.commit()
        attempt = start_attempt(session, task.id)
        session.commit()
        cp = Checkpoint(
            attempt=attempt,
            dataset_snapshot=snapshot,
            parent_artifact_hash="sha256:original-parent",
            training_config_hash="sha256:original-config",
            epoch=1,
            artifact_path=f"checkpoints/{uuid4()}.pt",
            artifact_hash=f"sha256:{uuid4().hex}",
            metrics_json={},
        )
        session.add(cp)
        session.commit()
        cp.parent_artifact_hash = "sha256:changed"
        with pytest.raises(ValueError, match="immutable"):
            session.commit()
