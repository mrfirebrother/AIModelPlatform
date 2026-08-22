from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
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
from backend.app.services.training_service import (
    cancel_task,
    complete_attempt,
    create_task,
    fail_attempt,
    get_task,
    start_attempt,
)


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


class TestCreateTask:
    def test_creates_queued_task(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        parent = _make_parent_model(session)
        task = create_task(
            session,
            parent_model_node_id=parent.id,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={"epochs": 10},
            resource_config_json={"gpu_memory_mb": 512},
            evaluation_policy_json={"mAP50": 0.5},
        )
        session.commit()
        assert task.status == "queued"
        assert task.parent_model_node_id == parent.id
        assert task.dataset_snapshot_id == snapshot.id
        assert task.training_config_json == {"epochs": 10}
        assert task.resource_config_json == {"gpu_memory_mb": 512}
        assert task.evaluation_policy_json == {"mAP50": 0.5}

    def test_task_without_parent_model(self, session: Session) -> None:
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
        assert task.parent_model_node_id is None
        assert task.status == "queued"

    def test_task_with_invalid_snapshot_raises(self, session: Session) -> None:
        with pytest.raises(ValueError, match="not found"):
            create_task(
                session,
                parent_model_node_id=None,
                dataset_snapshot_id=uuid4(),
                task_type="object_detection",
                model_family="yolo",
                training_config_json={},
                resource_config_json={},
                evaluation_policy_json={},
            )

    def test_task_with_invalid_parent_raises(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        with pytest.raises(ValueError, match="not found"):
            create_task(
                session,
                parent_model_node_id=uuid4(),
                dataset_snapshot_id=snapshot.id,
                task_type="object_detection",
                model_family="yolo",
                training_config_json={},
                resource_config_json={},
                evaluation_policy_json={},
            )


class TestStartAttempt:
    def test_start_attempt_creates_running_attempt(self, session: Session) -> None:
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
        assert attempt.task_id == task.id
        assert attempt.attempt_no == 1
        assert attempt.status == "running"
        assert attempt.lease_token is not None
        assert attempt.fencing_token is not None

    def test_start_attempt_transitions_task_to_running(
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
        session.refresh(task)
        assert task.status == "running"

    def test_second_attempt_increments_attempt_no(self, session: Session) -> None:
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
        # Fail the first attempt so we can start another
        fail_attempt(session, a1.id, "error")
        session.commit()
        a2 = start_attempt(session, task.id)
        session.commit()
        assert a2.attempt_no == 2
        assert a2.retry_count == 1

    def test_start_attempt_when_already_running_raises(
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
        start_attempt(session, task.id)
        session.commit()
        with pytest.raises(ValueError, match="already has an active attempt"):
            start_attempt(session, task.id)


class TestCompleteAttempt:
    def test_complete_attempt_creates_model_node(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        parent = _make_parent_model(session)
        task = create_task(
            session,
            parent_model_node_id=parent.id,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={"epochs": 5},
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
        assert model is not None
        assert model.parent_id == parent.id
        assert model.dataset_snapshot_id == snapshot.id
        assert model.training_attempt_id == attempt.id
        assert model.status == "candidate"
        session.refresh(task)
        assert task.status == "completed"

    def test_complete_attempt_without_parent(self, session: Session) -> None:
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
        model = complete_attempt(
            session,
            attempt.id,
            artifact_path=f"models/{uuid4()}.pt",
            artifact_hash=f"sha256:{uuid4().hex}",
        )
        session.commit()
        assert model.parent_id is None

    def test_complete_attempt_updates_attempt_status(
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
        complete_attempt(
            session,
            attempt.id,
            artifact_path=f"models/{uuid4()}.pt",
            artifact_hash=f"sha256:{uuid4().hex}",
        )
        session.commit()
        session.refresh(attempt)
        assert attempt.status == "completed"
        assert attempt.finished_at is not None


class TestFailAttempt:
    def test_fail_attempt_sets_failed_status(self, session: Session) -> None:
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
        fail_attempt(session, attempt.id, "CUDA out of memory")
        session.commit()
        session.refresh(attempt)
        assert attempt.status == "failed"
        assert attempt.last_error == "CUDA out of memory"
        assert attempt.finished_at is not None

    def test_fail_attempt_sets_task_to_failed_when_no_recovery(
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
        fail_attempt(session, attempt.id, "error")
        session.commit()
        session.refresh(task)
        assert task.status == "failed"


class TestCancelTask:
    def test_cancel_queued_task(self, session: Session) -> None:
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
        cancel_task(session, task.id)
        session.commit()
        session.refresh(task)
        assert task.status == "cancelled"

    def test_cancel_running_task(self, session: Session) -> None:
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
        session.refresh(task)
        session.refresh(attempt)
        assert task.status == "cancelled"
        assert attempt.status == "cancelled"


class TestGetTask:
    def test_get_task_returns_task(self, session: Session) -> None:
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
        fetched = get_task(session, task.id)
        assert fetched is not None
        assert fetched.id == task.id

    def test_get_task_returns_none_for_missing(self, session: Session) -> None:
        assert get_task(session, uuid4()) is None


class TestAttemptLeaseAndHeartbeat:
    def test_heartbeat_updates_timestamp(self, session: Session) -> None:
        from backend.app.workers.attempt_lease import heartbeat

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
        old_heartbeat = attempt.heartbeat_at
        heartbeat(session, attempt.id, attempt.lease_token)
        session.commit()
        session.refresh(attempt)
        assert attempt.heartbeat_at is not None
        if old_heartbeat is not None:
            assert attempt.heartbeat_at >= old_heartbeat

    def test_heartbeat_with_wrong_token_rejected(self, session: Session) -> None:
        from backend.app.workers.attempt_lease import heartbeat

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
        with pytest.raises(ValueError, match="lease token"):
            heartbeat(session, attempt.id, uuid4())
