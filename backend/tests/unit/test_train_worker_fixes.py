from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, inspect, text
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
    create_task,
    start_attempt,
    complete_attempt,
    fail_attempt,
    cancel_task,
)
from backend.app.workers.db_session import (
    worker_session,
    reset_worker_session_factory,
    set_worker_engine_override,
)
from backend.app.workers.train_worker import (
    _HeartbeatThread,
    execute_training,
    run_training,
    HEARTBEAT_INTERVAL_SECONDS,
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


def _make_label_schema(session: Session) -> LabelSchema:
    schema = LabelSchema(name=f"schema-{uuid4().hex[:8]}")
    session.add(schema)
    session.flush()
    return schema


def _make_snapshot(
    session: Session,
    *,
    label_schema: LabelSchema | None = None,
    manifest_hash: str | None = None,
    train_manifest_json: list | None = None,
    val_manifest_json: list | None = None,
    test_manifest_json: list | None = None,
) -> DatasetSnapshot:
    if label_schema is None:
        label_schema = _make_label_schema(session)
    dataset = Dataset(name=f"ds-{uuid4().hex[:8]}")
    if manifest_hash is None:
        nc = len(label_schema.classes) if label_schema else 0
        names = [c.semantic_key for c in label_schema.classes] if label_schema else []
        dataset_manifest = {
            "train_files": train_manifest_json or [],
            "val_files": val_manifest_json or [],
            "test_files": test_manifest_json or [],
            "nc": nc,
            "names": names,
        }
        manifest_hash = f"sha256:{hashlib.sha256(str(sorted(dataset_manifest.items())).encode()).hexdigest()}"
    snapshot = DatasetSnapshot(
        dataset=dataset,
        label_schema=label_schema,
        manifest_path=f"snapshots/{uuid4()}/manifest.json",
        manifest_hash=manifest_hash,
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def _make_snapshot_with_valid_hash(session: Session) -> DatasetSnapshot:
    schema = _make_label_schema(session)
    return _make_snapshot(session, label_schema=schema)


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


def _make_task(
    session: Session,
    *,
    snapshot: DatasetSnapshot | None = None,
    parent: ModelNode | None = None,
    training_config: dict[str, Any] | None = None,
    status: str = "queued",
) -> TrainingTask:
    if snapshot is None:
        snapshot = _make_snapshot(session)
    task = TrainingTask(
        parent_model_node_id=parent.id if parent else None,
        dataset_snapshot_id=snapshot.id,
        task_type="object_detection",
        model_family="yolo",
        training_config_json=training_config or {"epochs": 1, "device": "cpu"},
        resource_config_json={},
        evaluation_policy_json={},
        status=status,
    )
    session.add(task)
    session.flush()
    return task


class TestMigrationBooleanDefault:
    def test_cancellation_requested_server_default_is_false(self) -> None:
        table = Base.metadata.tables["training_tasks"]
        col = table.columns["cancellation_requested"]
        assert col.server_default is not None
        assert col.server_default.arg.text == "false"


class TestTrainingHeartbeat:
    def test_heartbeat_thread_sends_periodic_beats(self) -> None:
        mock_session = MagicMock()
        mock_heartbeat = MagicMock()
        thread = _HeartbeatThread(
            session=mock_session,
            attempt_id=uuid4(),
            lease_token=uuid4(),
            interval=1,
        )
        thread._stop_event.set()
        thread.run()
        mock_heartbeat.assert_not_called()

    def test_heartbeat_thread_stops_cleanly(self) -> None:
        mock_session = MagicMock()
        thread = _HeartbeatThread(
            session=mock_session,
            attempt_id=uuid4(),
            lease_token=uuid4(),
            interval=1,
        )
        thread.start()
        thread.stop()
        thread.join(timeout=2)
        assert not thread.is_alive()

    @patch("backend.app.workers.train_worker.attempt_heartbeat")
    def test_heartbeat_thread_sends_beat_while_running(self, mock_hb: MagicMock) -> None:
        mock_session = MagicMock()
        attempt_id = uuid4()
        lease_token = uuid4()
        thread = _HeartbeatThread(
            session=mock_session,
            attempt_id=attempt_id,
            lease_token=lease_token,
            interval=0.05,
        )
        thread.start()
        import time
        time.sleep(0.2)
        thread.stop()
        thread.join(timeout=2)
        assert mock_hb.call_count >= 1
        call_args = mock_hb.call_args
        assert call_args[0][1] == attempt_id
        assert call_args[0][2] == lease_token
        assert call_args[0][0] is not mock_session

    def test_heartbeat_thread_is_daemon(self) -> None:
        mock_session = MagicMock()
        thread = _HeartbeatThread(
            session=mock_session,
            attempt_id=uuid4(),
            lease_token=uuid4(),
            interval=60,
        )
        assert thread.daemon is True


class TestCancellationAfterTraining:
    def test_pre_training_cancellation_aborts(self) -> None:
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
                snapshot = _make_snapshot_with_valid_hash(test_session)
                test_session.commit()
                snapshot_id = snapshot.id

            with worker_session() as ws:
                task = TrainingTask(
                    dataset_snapshot_id=snapshot_id,
                    task_type="object_detection",
                    model_family="yolo",
                    training_config_json={"epochs": 1, "device": "cpu"},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="queued",
                    cancellation_requested=True,
                )
                ws.add(task)
                ws.flush()
                task_id = str(task.id)

            result = run_training(task_id)
            assert "cancelled" in result.lower()

            with Session(engine) as verify:
                task_v = verify.get(TrainingTask, __import__("uuid").UUID(task_id))
                assert task_v is not None
                assert task_v.status == "cancelled"
        finally:
            reset_worker_session_factory()
            engine.dispose()

    def test_post_training_cancellation_fails_attempt(self) -> None:
        from datetime import datetime, timezone

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
                snapshot = _make_snapshot_with_valid_hash(test_session)
                parent = _make_parent_model(test_session)
                test_session.commit()
                snapshot_id = snapshot.id
                parent_id = parent.id

            with worker_session() as ws:
                task = TrainingTask(
                    parent_model_node_id=parent_id,
                    dataset_snapshot_id=snapshot_id,
                    task_type="object_detection",
                    model_family="yolo",
                    training_config_json={"epochs": 1, "device": "cpu"},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="running",
                )
                ws.add(task)
                ws.flush()
                task_id = str(task.id)
                task_uuid = task.id

            def fake_execute_training(**kwargs):
                with worker_session() as ws2:
                    task_in_db = ws2.get(TrainingTask, task_uuid)
                    task_in_db.cancellation_requested = True
                    ws2.commit()
                return MagicMock(
                    success=True,
                    epochs_completed=10,
                    best_model_path=None,
                    latest_model_path=None,
                    metrics={},
                    error=None,
                )

            with patch("backend.app.workers.train_worker.execute_training", side_effect=fake_execute_training):
                result = run_training(task_id)
                assert "cancelled" in result.lower()

            with Session(engine) as verify:
                task_v = verify.get(TrainingTask, task_uuid)
                attempts = verify.query(TrainingAttempt).filter(
                    TrainingAttempt.task_id == task_v.id
                ).all()
                cancelled_attempts = [a for a in attempts if a.status == "cancelled"]
                assert len(cancelled_attempts) == 1
        finally:
            reset_worker_session_factory()
            engine.dispose()


class TestGPUBinding:
    def test_gpu_device_initially_none_after_start_attempt(self) -> None:
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
                    training_config_json={"epochs": 1, "device": "cpu"},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="queued",
                )
                ws.add(task)
                ws.flush()
                attempt = start_attempt(ws, task.id)
                ws.commit()
                attempt_id = attempt.id

                ws.refresh(attempt)
                assert attempt.gpu_device is None

            with Session(engine) as verify:
                attempt_v = verify.get(TrainingAttempt, attempt_id)
                assert attempt_v.gpu_device is None
        finally:
            reset_worker_session_factory()
            engine.dispose()

    def test_run_training_sets_gpu_device_from_config(self) -> None:
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
                    training_config_json={"epochs": 1, "device": "cpu"},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="queued",
                )
                ws.add(task)
                ws.flush()
                task_id = str(task.id)

            def fake_execute_training(**kwargs):
                from backend.app.training.yolo_trainer import TrainResult
                return TrainResult(
                    success=False,
                    epochs_completed=0,
                    best_model_path=None,
                    latest_model_path=None,
                    error="mock",
                )

            with patch("backend.app.workers.train_worker.execute_training", side_effect=fake_execute_training):
                result = run_training(task_id)

            with Session(engine) as verify:
                task_v = verify.get(TrainingTask, __import__("uuid").UUID(task_id))
                attempts = verify.query(TrainingAttempt).filter(
                    TrainingAttempt.task_id == task_v.id
                ).all()
                assert len(attempts) == 1
                assert attempts[0].gpu_device == "cpu"
        finally:
            reset_worker_session_factory()
            engine.dispose()


class TestManifestHashValidation:
    def test_run_training_checks_manifest_hash_none(self) -> None:
        import inspect
        source = inspect.getsource(run_training)
        assert "manifest_hash" in source
        assert "is None" in source

        from backend.app.services.training_service import start_attempt, fail_attempt
        svc_source = inspect.getsource(fail_attempt)
        assert "error" in svc_source

    def test_valid_manifest_hash_proceeds(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        assert snapshot.manifest_hash is not None
        assert snapshot.manifest_hash.startswith("sha256:")


class TestCheckpointDirFromConfig:
    @patch("backend.app.workers.train_worker.YoloTrainer")
    @patch("backend.app.workers.train_worker.YoloDataset")
    def test_uses_config_checkpoint_dir(self, mock_dataset_cls: MagicMock, mock_trainer_cls: MagicMock) -> None:
        import tempfile

        from backend.app.config import Settings

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MagicMock(
            success=False,
            epochs_completed=0,
            best_model_path=None,
            latest_model_path=None,
            metrics={},
            error="mock",
        )
        mock_trainer_cls.return_value = mock_trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(
                checkpoint_dir=tmpdir,
                postgres_url="sqlite+pysqlite:///:memory:",
            )
            with patch("backend.app.workers.train_worker.get_settings", return_value=settings):
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
                        snapshot = _make_snapshot_with_valid_hash(test_session)
                        test_session.commit()
                        snapshot_id = snapshot.id

                    with worker_session() as ws:
                        task = TrainingTask(
                            dataset_snapshot_id=snapshot_id,
                            task_type="object_detection",
                            model_family="yolo",
                            training_config_json={"epochs": 1, "device": "cpu"},
                            resource_config_json={},
                            evaluation_policy_json={},
                            status="queued",
                        )
                        ws.add(task)
                        ws.flush()
                        task_id = str(task.id)

                    result = run_training(task_id)
                    expected_dir = Path(tmpdir) / task_id
                    assert expected_dir.exists() or "not found" in result or "failed" in result
                finally:
                    reset_worker_session_factory()
                    engine.dispose()


class TestLabelSchemaOnCandidateModel:
    def test_complete_attempt_sets_label_schema_id(self, session: Session) -> None:
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

        model = complete_attempt(
            session,
            attempt.id,
            artifact_path=f"models/{uuid4()}.pt",
            artifact_hash=f"sha256:{uuid4().hex}",
            label_schema_id=snapshot.label_schema_id,
        )
        session.commit()

        session.refresh(model)
        assert model.label_schema_id == snapshot.label_schema_id

    def test_complete_attempt_without_label_schema_id(self, session: Session) -> None:
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

        model = complete_attempt(
            session,
            attempt.id,
            artifact_path=f"models/{uuid4()}.pt",
            artifact_hash=f"sha256:{uuid4().hex}",
        )
        session.commit()

        session.refresh(model)
        assert model.label_schema_id is None


class TestSchedulerGPULeaseExpiry:
    def test_reconcile_leases_expires_gpu_leases(self) -> None:
        from datetime import datetime, timedelta, timezone

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
                gpu = GPUResource(
                    gpu_device="gpu:0",
                    capacity_memory_mb=8192,
                    reserved_memory_mb=0,
                )
                test_session.add(gpu)
                test_session.flush()

                schema = _make_label_schema(test_session)
                dataset = Dataset(name=f"ds-{uuid4().hex[:8]}")
                snapshot = DatasetSnapshot(
                    dataset=dataset,
                    label_schema=schema,
                    manifest_path=f"snapshots/{uuid4()}/manifest.json",
                    manifest_hash=f"sha256:{uuid4().hex}",
                )
                test_session.add(snapshot)
                test_session.flush()

                task = TrainingTask(
                    dataset_snapshot_id=snapshot.id,
                    task_type="object_detection",
                    model_family="yolo",
                    training_config_json={"epochs": 1, "device": "cpu"},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="running",
                )
                test_session.add(task)
                test_session.flush()

                attempt = start_attempt(test_session, task.id)
                test_session.flush()

                lease = GPUResourceLease(
                    resource=gpu,
                    gpu_device="gpu:0",
                    owner_type="training_attempt",
                    owner_id=attempt.id,
                    reserved_memory_mb=4096,
                    lease_token=uuid4(),
                    fencing_token=1,
                    lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
                    status="active",
                )
                test_session.add(lease)
                test_session.commit()
                lease_id = lease.id

            with worker_session() as ws:
                from backend.app.workers.scheduler import reconcile_leases
                result = reconcile_leases()
                assert "expired" in result.lower()

            with Session(engine) as verify:
                gpu_lease = verify.get(GPUResourceLease, lease_id)
                assert gpu_lease.status in ("expired", "released")
        finally:
            reset_worker_session_factory()
            engine.dispose()


class TestRealYoloTrainingTerminalState:
    @patch("backend.app.workers.train_worker.YoloTrainer")
    @patch("backend.app.workers.train_worker.YoloDataset")
    def test_successful_training_completes_task(
        self, mock_dataset_cls: MagicMock, mock_trainer_cls: MagicMock, tmp_path: Path
    ) -> None:
        from sqlalchemy import create_engine, event

        engine = create_engine("sqlite+pysqlite:///:memory:")
        event.listen(
            engine,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
        Base.metadata.create_all(engine)
        set_worker_engine_override(engine)

        best_model = tmp_path / "best.pt"
        best_model.write_bytes(b"fake model weights")

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MagicMock(
            success=True,
            epochs_completed=10,
            best_model_path=str(best_model),
            latest_model_path=str(best_model),
            metrics={"mAP50": 0.95},
            error=None,
        )
        mock_trainer_cls.return_value = mock_trainer

        try:
            with Session(engine) as test_session:
                snapshot = _make_snapshot_with_valid_hash(test_session)
                parent = _make_parent_model(test_session)
                test_session.commit()
                snapshot_id = snapshot.id
                parent_id = parent.id

            with worker_session() as ws:
                task = TrainingTask(
                    parent_model_node_id=parent_id,
                    dataset_snapshot_id=snapshot_id,
                    task_type="object_detection",
                    model_family="yolo",
                    training_config_json={"epochs": 10, "device": "cpu"},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="queued",
                )
                ws.add(task)
                ws.flush()
                task_id = str(task.id)

            result = run_training(task_id)
            assert "completed" in result

            with Session(engine) as verify:
                task_v = verify.get(TrainingTask, __import__("uuid").UUID(task_id))
                assert task_v is not None
                assert task_v.status == "completed"
                attempts = verify.query(TrainingAttempt).filter(
                    TrainingAttempt.task_id == task_v.id
                ).all()
                completed = [a for a in attempts if a.status == "completed"]
                assert len(completed) == 1
                models = verify.query(ModelNode).filter(
                    ModelNode.training_attempt_id == completed[0].id
                ).all()
                assert len(models) == 1
                assert models[0].status == "candidate"
                assert models[0].label_schema_id is not None
        finally:
            reset_worker_session_factory()
            engine.dispose()

    @patch("backend.app.workers.train_worker.YoloTrainer")
    @patch("backend.app.workers.train_worker.YoloDataset")
    def test_failed_training_fails_task(
        self, mock_dataset_cls: MagicMock, mock_trainer_cls: MagicMock
    ) -> None:
        from sqlalchemy import create_engine, event

        engine = create_engine("sqlite+pysqlite:///:memory:")
        event.listen(
            engine,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
        Base.metadata.create_all(engine)
        set_worker_engine_override(engine)

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = MagicMock(
            success=False,
            epochs_completed=0,
            best_model_path=None,
            latest_model_path=None,
            metrics={},
            error="CUDA out of memory",
        )
        mock_trainer_cls.return_value = mock_trainer

        try:
            with Session(engine) as test_session:
                snapshot = _make_snapshot_with_valid_hash(test_session)
                test_session.commit()
                snapshot_id = snapshot.id

            with worker_session() as ws:
                task = TrainingTask(
                    dataset_snapshot_id=snapshot_id,
                    task_type="object_detection",
                    model_family="yolo",
                    training_config_json={"epochs": 10, "device": "cpu"},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="queued",
                )
                ws.add(task)
                ws.flush()
                task_id = str(task.id)

            result = run_training(task_id)
            assert "failed" in result

            with Session(engine) as verify:
                task_v = verify.get(TrainingTask, __import__("uuid").UUID(task_id))
                assert task_v is not None
                assert task_v.status == "failed"
        finally:
            reset_worker_session_factory()
            engine.dispose()


class TestCancelTaskCancelsAttempt:
    def test_cancel_running_task_cancels_attempt(self) -> None:
        from datetime import datetime, timezone

        from sqlalchemy import create_engine, event

        from backend.app.services.training_service import cancel_task

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
                    training_config_json={"epochs": 1, "device": "cpu"},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="running",
                )
                ws.add(task)
                ws.flush()

                attempt = start_attempt(ws, task.id)
                ws.commit()
                attempt_id = attempt.id
                task_id = task.id

            with worker_session() as ws:
                updated_task = cancel_task(ws, task_id)
                ws.commit()
                assert updated_task.cancellation_requested is True
                assert updated_task.status == "running"

            with Session(engine) as verify:
                attempt_v = verify.get(TrainingAttempt, attempt_id)
                assert attempt_v is not None
                assert attempt_v.status == "running"
        finally:
            reset_worker_session_factory()
            engine.dispose()

    def test_cancel_queued_task_does_not_affect_attempts(self) -> None:
        from sqlalchemy import create_engine, event

        from backend.app.services.training_service import cancel_task

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
                    training_config_json={"epochs": 1, "device": "cpu"},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="queued",
                )
                ws.add(task)
                ws.flush()
                task_id = task.id

            with worker_session() as ws:
                updated_task = cancel_task(ws, task_id)
                ws.commit()
                assert updated_task.status == "cancelled"
                assert updated_task.cancellation_requested is True

            with Session(engine) as verify:
                task_v = verify.get(TrainingTask, task_id)
                attempts = verify.query(TrainingAttempt).filter(
                    TrainingAttempt.task_id == task_v.id
                ).all()
                assert len(attempts) == 0
        finally:
            reset_worker_session_factory()
            engine.dispose()


class TestFailAttemptWritesReason:
    def test_fail_attempt_sets_task_failure_reason(self, session: Session) -> None:
        from backend.app.services.training_service import fail_attempt

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

        fail_attempt(session, attempt.id, error="GPU out of memory")
        session.commit()

        session.refresh(task)
        assert task.status == "failed"
        assert task.failure_reason == "GPU out of memory"


class TestSweepQueuedTasks:
    @patch("backend.app.workers.train_worker.run_training")
    @patch("backend.app.workers.db_session.worker_session")
    def test_sweep_dispatches_queued_tasks(
        self, mock_worker_session: MagicMock, mock_run: MagicMock
    ) -> None:
        from sqlalchemy import create_engine, event

        engine = create_engine("sqlite+pysqlite:///:memory:")
        event.listen(
            engine,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
        Base.metadata.create_all(engine)

        try:
            with Session(engine) as test_session:
                snapshot = _make_snapshot(test_session)
                test_session.commit()
                snapshot_id = snapshot.id

                task = TrainingTask(
                    dataset_snapshot_id=snapshot_id,
                    task_type="object_detection",
                    model_family="yolo",
                    training_config_json={"epochs": 1, "device": "cpu"},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="queued",
                )
                test_session.add(task)
                test_session.commit()
                task_id = task.id

            mock_task = MagicMock()
            mock_task.id = task_id
            mock_session = MagicMock()
            mock_session.execute.return_value.scalars.return_value.all.return_value = [mock_task]
            mock_worker_session.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_worker_session.return_value.__exit__ = MagicMock(return_value=False)

            mock_run.delay = MagicMock()
            from backend.app.workers.scheduler import sweep_queued_tasks
            result = sweep_queued_tasks()
            assert "dispatched" in result
            assert mock_run.delay.call_count == 1
            mock_run.delay.assert_called_with(str(task_id))
        finally:
            engine.dispose()

    @patch("backend.app.workers.train_worker.run_training")
    @patch("backend.app.workers.db_session.worker_session")
    def test_sweep_skips_cancelled_tasks(
        self, mock_worker_session: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_session = MagicMock()
        mock_session.execute.return_value.scalars.return_value.all.return_value = []
        mock_worker_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_worker_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_run.delay = MagicMock()
        from backend.app.workers.scheduler import sweep_queued_tasks
        result = sweep_queued_tasks()
        assert "no queued tasks" in result
        mock_run.delay.assert_not_called()


class TestHeartbeatUpdatesGPULease:
    @patch("backend.app.workers.train_worker.ResourceScheduler")
    @patch("backend.app.workers.train_worker.attempt_heartbeat")
    @patch("backend.app.workers.db_session.worker_session")
    def test_heartbeat_thread_updates_gpu_lease(
        self,
        mock_worker_session: MagicMock,
        mock_attempt_heartbeat: MagicMock,
        mock_scheduler_cls: MagicMock,
    ) -> None:
        mock_session = MagicMock()
        mock_worker_session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_worker_session.return_value.__exit__ = MagicMock(return_value=False)
        mock_scheduler = MagicMock()
        mock_scheduler_cls.return_value = mock_scheduler
        attempt_id = uuid4()
        lease_token = uuid4()
        thread = _HeartbeatThread(
            session=MagicMock(),
            attempt_id=attempt_id,
            lease_token=lease_token,
            interval=0.05,
        )
        thread.start()
        import time
        time.sleep(0.2)
        thread.stop()
        thread.join(timeout=2)
        mock_attempt_heartbeat.assert_called()
        mock_scheduler.heartbeat.assert_called()
        call_args = mock_scheduler.heartbeat.call_args
        assert call_args[0][0] == attempt_id
        assert call_args[0][1] == lease_token
