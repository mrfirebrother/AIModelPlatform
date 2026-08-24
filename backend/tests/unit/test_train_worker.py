from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from backend.app.models import (
    Base,
    Checkpoint,
    Dataset,
    DatasetSnapshot,
    LabelSchema,
    ModelNode,
    TrainingAttempt,
    TrainingTask,
)
from backend.app.services.training_service import (
    create_task,
    start_attempt,
    complete_attempt,
    fail_attempt,
)
from backend.app.workers.db_session import worker_session, reset_worker_session_factory, set_worker_engine_override
from backend.app.workers.train_worker import run_training


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


class TestWorkerContract:
    def test_queued_task_creates_attempt_and_transitions_to_running(
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
            training_config_json={"epochs": 1},
            resource_config_json={},
            evaluation_policy_json={},
        )
        session.commit()
        assert task.status == "queued"

        attempt = start_attempt(session, task.id)
        session.commit()
        session.refresh(task)
        assert task.status == "running"
        assert attempt.status == "running"
        assert attempt.attempt_no == 1

    def test_successful_training_transitions_to_completed_and_creates_candidate(
        self, session: Session, tmp_path: Path
    ) -> None:
        snapshot = _make_snapshot(session)
        parent = _make_parent_model(session)
        task = create_task(
            session,
            parent_model_node_id=parent.id,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={"epochs": 1},
            resource_config_json={},
            evaluation_policy_json={},
        )
        session.commit()

        attempt = start_attempt(session, task.id)
        session.commit()

        best_path = tmp_path / "best.pt"
        best_path.write_bytes(b"fake model weights")
        artifact_hash = f"sha256:{hashlib.sha256(b'fake model weights').hexdigest()}"

        model = complete_attempt(
            session,
            attempt.id,
            artifact_path=str(best_path),
            artifact_hash=artifact_hash,
        )
        session.commit()

        session.refresh(attempt)
        session.refresh(task)
        assert attempt.status == "completed"
        assert task.status == "completed"
        assert model.status == "candidate"
        assert model.parent_id == parent.id
        assert model.training_attempt_id == attempt.id

    def test_failed_training_transitions_to_failed_and_releases_resource(
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

        fail_attempt(session, attempt.id, error="GPU OOM")
        session.commit()

        session.refresh(attempt)
        session.refresh(task)
        assert attempt.status == "failed"
        assert attempt.last_error == "GPU OOM"
        assert task.status == "failed"

    def test_missing_task_id_fails_without_creating_attempt(
        self, session: Session
    ) -> None:
        from backend.app.models import TrainingTask

        fake_id = uuid4()
        with pytest.raises(ValueError, match="not found"):
            start_attempt(session, fake_id)

        attempts = session.query(TrainingAttempt).all()
        assert len(attempts) == 0

    def test_terminal_task_is_idempotent_no_second_attempt(
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

        session.refresh(task)
        assert task.status == "completed"

        with pytest.raises(ValueError, match="Cannot start attempt"):
            start_attempt(session, task.id)


class TestWorkerSession:
    def test_worker_session_commits_on_success(self, session: Session) -> None:
        from sqlalchemy import create_engine, event

        engine = create_engine("sqlite+pysqlite:///:memory:")
        event.listen(
            engine,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
        Base.metadata.create_all(engine)
        set_worker_engine_override(engine)

        try:
            with Session(engine) as test_session:
                snapshot = _make_snapshot(test_session)
                test_session.commit()
                snapshot_id = snapshot.id

            with worker_session() as ws:
                task = TrainingTask(
                    dataset_snapshot_id=snapshot_id,
                    task_type="object_detection",
                    model_family="yolo",
                    training_config_json={},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="queued",
                )
                ws.add(task)
                ws.flush()
                task_id = task.id

            with Session(engine) as verify_session:
                retrieved = verify_session.get(TrainingTask, task_id)
                assert retrieved is not None
                assert retrieved.status == "queued"
        finally:
            reset_worker_session_factory()
            engine.dispose()

    def test_worker_session_rolls_back_on_exception(self, session: Session) -> None:
        from sqlalchemy import create_engine, event

        engine = create_engine("sqlite+pysqlite:///:memory:")
        event.listen(
            engine,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
        Base.metadata.create_all(engine)
        set_worker_engine_override(engine)

        try:
            with Session(engine) as test_session:
                snapshot = _make_snapshot(test_session)
                test_session.commit()
                snapshot_id = snapshot.id

            with pytest.raises(RuntimeError):
                with worker_session() as ws:
                    task = TrainingTask(
                        dataset_snapshot_id=snapshot_id,
                        task_type="object_detection",
                        model_family="yolo",
                        training_config_json={},
                        resource_config_json={},
                        evaluation_policy_json={},
                        status="queued",
                    )
                    ws.add(task)
                    ws.flush()
                    raise RuntimeError("Simulated failure")

            with Session(engine) as verify_session:
                count = verify_session.query(TrainingTask).count()
                assert count == 0
        finally:
            reset_worker_session_factory()
            engine.dispose()


class TestTrainingOrchestration:
    def test_run_training_invalid_task_id(self) -> None:
        result = run_training("invalid-id")
        assert "invalid task id" in result

    def test_run_training_missing_task(self) -> None:
        from sqlalchemy import create_engine, event

        engine = create_engine("sqlite+pysqlite:///:memory:")
        event.listen(
            engine,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
        Base.metadata.create_all(engine)
        set_worker_engine_override(engine)

        try:
            fake_id = str(uuid4())
            result = run_training(fake_id)
            assert "not found" in result
        finally:
            reset_worker_session_factory()
            engine.dispose()

    def test_run_training_terminal_task_noop(self) -> None:
        from sqlalchemy import create_engine, event

        engine = create_engine("sqlite+pysqlite:///:memory:")
        event.listen(
            engine,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
        Base.metadata.create_all(engine)
        set_worker_engine_override(engine)

        try:
            with Session(engine) as test_session:
                snapshot = _make_snapshot(test_session)
                test_session.commit()
                snapshot_id = snapshot.id

            with worker_session() as ws:
                task = TrainingTask(
                    dataset_snapshot_id=snapshot_id,
                    task_type="object_detection",
                    model_family="yolo",
                    training_config_json={},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="completed",
                )
                ws.add(task)
                ws.flush()
                task_id = str(task.id)

            result = run_training(task_id)
            assert "already completed" in result
        finally:
            reset_worker_session_factory()
            engine.dispose()
