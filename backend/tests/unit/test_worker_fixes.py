"""Tests for training worker critical defect fixes (8 P0/P1 issues).

TDD: tests written BEFORE code changes.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, StaticPool
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
    start_attempt,
)
from backend.app.workers.db_session import (
    reset_worker_session_factory,
    set_worker_engine_override,
    worker_session,
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


# ──────────────────────────────────────────────────────────────────────
# Fix 1: Migration chain
# ──────────────────────────────────────────────────────────────────────
class TestMigrationChain:
    def test_0002_down_revision_matches_0001_revision(self) -> None:
        import importlib
        mod = importlib.import_module(
            "backend.alembic.versions.0002_training_cancellation_request"
        )
        assert mod.down_revision == "0001_initial_schema"


# ──────────────────────────────────────────────────────────────────────
# Fix 2: execute_training should NOT mutate attempt.status
# ──────────────────────────────────────────────────────────────────────
class TestExecuteTrainingStatusConsolidation:
    def test_execute_training_does_not_set_attempt_status(
        self, session: Session, tmp_path: Path
    ) -> None:
        from backend.app.workers.train_worker import execute_training

        snapshot = _make_snapshot(session)
        parent = _make_parent_model(session)
        task = create_task(
            session,
            parent_model_node_id=parent.id,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={"epochs": 1, "device": "cpu"},
            resource_config_json={},
            evaluation_policy_json={},
        )
        session.commit()
        attempt = start_attempt(session, task.id)
        session.commit()

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        dataset_manifest = {
            "train_files": [],
            "val_files": [],
            "test_files": [],
            "nc": 1,
            "names": ["obj"],
        }

        result = execute_training(
            session=session,
            task=task,
            attempt=attempt,
            training_config={"epochs": 1, "device": "cpu"},
            dataset_manifest=dataset_manifest,
            output_dir=output_dir,
            parent_model_path=None,
            required_memory_mb=1024,
        )

        session.refresh(attempt)
        assert attempt.status == "running", (
            "execute_training should not set attempt.status; "
            "run_training will call complete_attempt/fail_attempt"
        )

    def test_run_training_orchestrates_complete_attempt(
        self, tmp_path: Path
    ) -> None:
        """run_training should be the sole authority for attempt status transition."""
        from unittest.mock import patch
        from backend.app.workers.train_worker import run_training
        from backend.app.training.yolo_trainer import TrainResult

        db_path = tmp_path / "test.db"
        eng = create_engine(f"sqlite:///{db_path}")
        event.listen(
            eng,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
        Base.metadata.create_all(eng)

        with Session(eng) as setup:
            snapshot = _make_snapshot(setup)
            parent = _make_parent_model(setup)
            task = create_task(
                setup,
                parent_model_node_id=parent.id,
                dataset_snapshot_id=snapshot.id,
                task_type="object_detection",
                model_family="yolo",
                training_config_json={"epochs": 1, "device": "cpu"},
                resource_config_json={},
                evaluation_policy_json={},
            )
            setup.commit()
            task_id = str(task.id)

        best_path = tmp_path / "best.pt"
        best_path.write_bytes(b"fake model weights")

        set_worker_engine_override(eng)
        try:
            with patch(
                "backend.app.workers.train_worker.execute_training"
            ) as mock_exec:
                mock_exec.return_value = TrainResult(
                    success=True,
                    epochs_completed=1,
                    best_model_path=str(best_path),
                    latest_model_path=None,
                    metrics={"mAP50": 0.5},
                )
                result = run_training(task_id)

            with Session(eng) as verify:
                from sqlalchemy import text
                row = verify.execute(
                    text("SELECT status FROM training_attempts WHERE task_id = replace(:tid, '-', '')"),
                    {"tid": task_id},
                ).fetchone()
                assert row is not None
                assert row[0] == "completed"
        finally:
            reset_worker_session_factory()
            eng.dispose()


# ──────────────────────────────────────────────────────────────────────
# Fix 3: Celery dispatch race condition - commit before dispatch
# ──────────────────────────────────────────────────────────────────────
class TestDispatchRaceCondition:
    def test_route_commits_before_dispatch(self) -> None:
        import inspect
        from backend.app.api.routes.training import create_training_task
        source = inspect.getsource(create_training_task)
        lines = source.split("\n")
        commit_line_idx = None
        dispatch_line_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "commit" in stripped:
                commit_line_idx = i
            if "delay" in stripped:
                dispatch_line_idx = i
                break
        assert commit_line_idx is not None, "Route must call db.commit()"
        assert dispatch_line_idx is not None, "Route must dispatch Celery task"
        assert commit_line_idx < dispatch_line_idx, (
            "db.commit() must appear before run_training.delay()"
        )


# ──────────────────────────────────────────────────────────────────────
# Fix 4: Safe cancellation
# ──────────────────────────────────────────────────────────────────────
class TestSafeCancellation:
    def test_cancel_running_task_sets_cancellation_requested_only(
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

        cancel_task(session, task.id)
        session.commit()

        session.refresh(task)
        session.refresh(attempt)
        assert task.cancellation_requested is True
        assert attempt.status == "running"

    def test_cancel_queued_task_marks_cancelled_directly(
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

        cancel_task(session, task.id)
        session.commit()

        session.refresh(task)
        assert task.status == "cancelled"
        assert task.cancellation_requested is True


# ──────────────────────────────────────────────────────────────────────
# Fix 5: Lease initialization + scheduler sweep
# ──────────────────────────────────────────────────────────────────────
class TestLeaseInitialization:
    def test_start_attempt_sets_lease_expires_at_30min(
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
        session.refresh(attempt)

        assert attempt.lease_expires_at is not None
        now = datetime.now(timezone.utc)
        expires = attempt.lease_expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        delta = expires - now
        assert timedelta(minutes=29) <= delta <= timedelta(minutes=31), (
            f"lease_expires_at should be ~30min from now, got delta={delta}"
        )

    def test_start_attempt_sets_heartbeat_at(self, session: Session) -> None:
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
        session.refresh(attempt)
        assert attempt.heartbeat_at is not None


class TestSchedulerReconcileLeases:
    def test_reconcile_leases_fails_expired_attempts(self, tmp_path: Path) -> None:
        from backend.app.workers.scheduler import reconcile_leases

        db_path = tmp_path / "test_reconcile.db"
        eng = create_engine(f"sqlite:///{db_path}")
        event.listen(
            eng,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
        Base.metadata.create_all(eng)

        with Session(eng) as setup:
            snapshot = _make_snapshot(setup)
            task = create_task(
                setup,
                parent_model_node_id=None,
                dataset_snapshot_id=snapshot.id,
                task_type="object_detection",
                model_family="yolo",
                training_config_json={},
                resource_config_json={},
                evaluation_policy_json={},
            )
            setup.commit()
            attempt = start_attempt(setup, task.id)
            setup.commit()
            attempt_id = attempt.id

            attempt.lease_expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)
            setup.commit()

        set_worker_engine_override(eng)
        try:
            result = reconcile_leases()
            assert "expired" in result.lower() or "reconcil" in result.lower() or "failed" in result.lower()
        finally:
            reset_worker_session_factory()

        with Session(eng) as verify:
            from sqlalchemy import text
            row = verify.execute(
                text("SELECT status, last_error FROM training_attempts WHERE id = replace(:aid, '-', '')"),
                {"aid": str(attempt_id)},
            ).fetchone()
            assert row is not None
            assert row[0] == "failed"
            assert "lease expired" in row[1].lower()
        eng.dispose()


# ──────────────────────────────────────────────────────────────────────
# Fix 6: GPU config - cpu_mode from task config
# ──────────────────────────────────────────────────────────────────────
class TestGPUConfigCpuMode:
    def test_execute_training_derives_cpu_mode_from_config(
        self, session: Session, tmp_path: Path
    ) -> None:
        from unittest.mock import patch
        from backend.app.workers.train_worker import execute_training

        snapshot = _make_snapshot(session)
        parent = _make_parent_model(session)
        task = create_task(
            session,
            parent_model_node_id=parent.id,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={"epochs": 1, "device": "cpu"},
            resource_config_json={},
            evaluation_policy_json={},
        )
        session.commit()
        attempt = start_attempt(session, task.id)
        session.commit()

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        dataset_manifest = {
            "train_files": [],
            "val_files": [],
            "test_files": [],
            "nc": 1,
            "names": ["obj"],
        }

        with patch(
            "backend.app.workers.train_worker.ResourceScheduler"
        ) as MockScheduler:
            mock_scheduler = MockScheduler.return_value
            mock_scheduler.release_gpu.return_value = None
            try:
                execute_training(
                    session=session,
                    task=task,
                    attempt=attempt,
                    training_config={"device": "cpu"},
                    dataset_manifest=dataset_manifest,
                    output_dir=output_dir,
                    parent_model_path=None,
                    required_memory_mb=4096,
                )
            except Exception:
                pass
            call_kwargs = MockScheduler.call_args
            assert call_kwargs.kwargs.get("cpu_mode") is True, (
                "ResourceScheduler should use cpu_mode=True when device=cpu"
            )

    def test_execute_training_gpu_mode_when_device_not_cpu(
        self, session: Session, tmp_path: Path
    ) -> None:
        from unittest.mock import patch
        from backend.app.workers.train_worker import execute_training

        snapshot = _make_snapshot(session)
        parent = _make_parent_model(session)
        task = create_task(
            session,
            parent_model_node_id=parent.id,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={"epochs": 1, "device": "0"},
            resource_config_json={},
            evaluation_policy_json={},
        )
        session.commit()
        attempt = start_attempt(session, task.id)
        session.commit()

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        dataset_manifest = {
            "train_files": [],
            "val_files": [],
            "test_files": [],
            "nc": 1,
            "names": ["obj"],
        }

        with patch(
            "backend.app.workers.train_worker.ResourceScheduler"
        ) as MockScheduler:
            mock_scheduler = MockScheduler.return_value
            mock_scheduler.release_gpu.return_value = None
            try:
                execute_training(
                    session=session,
                    task=task,
                    attempt=attempt,
                    training_config={"device": "0"},
                    dataset_manifest=dataset_manifest,
                    output_dir=output_dir,
                    parent_model_path=None,
                    required_memory_mb=4096,
                )
            except Exception:
                pass
            call_kwargs = MockScheduler.call_args
            assert call_kwargs.kwargs.get("cpu_mode") is False, (
                "ResourceScheduler should use cpu_mode=False when device is GPU"
            )


# ──────────────────────────────────────────────────────────────────────
# Fix 7: Checkpoint info in complete_attempt
# ──────────────────────────────────────────────────────────────────────
class TestCheckpointInfoInCompleteAttempt:
    def test_complete_attempt_creates_checkpoint_record(
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
            training_config_json={"epochs": 3},
            resource_config_json={},
            evaluation_policy_json={},
        )
        session.commit()
        attempt = start_attempt(session, task.id)
        session.commit()

        best_path = f"models/{uuid4()}.pt"
        model = complete_attempt(
            session,
            attempt.id,
            artifact_path=best_path,
            artifact_hash=f"sha256:{uuid4().hex}",
            checkpoint_info={
                "epoch": 3,
                "is_best": True,
                "metrics": {"mAP50": 0.75, "mAP50-95": 0.55},
            },
        )
        session.commit()

        checkpoints = (
            session.query(Checkpoint)
            .filter_by(attempt_id=attempt.id)
            .all()
        )
        assert len(checkpoints) == 1
        assert checkpoints[0].epoch == 3
        assert checkpoints[0].metrics_json == {"mAP50": 0.75, "mAP50-95": 0.55}
        assert checkpoints[0].artifact_path == best_path

        session.refresh(attempt)
        assert attempt.best_checkpoint_id == checkpoints[0].id


# ──────────────────────────────────────────────────────────────────────
# Fix 8: Parent model missing -> fail
# ──────────────────────────────────────────────────────────────────────
class TestParentModelMissingFails:
    def test_yolo_trainer_fails_when_parent_model_not_found(
        self, tmp_path: Path
    ) -> None:
        from backend.app.training.yolo_trainer import YoloTrainer

        trainer = YoloTrainer({"epochs": 1, "device": "cpu"})
        nonexistent = str(tmp_path / "nonexistent_model.pt")
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "data.yaml").write_text(
            "train: train\nval: val\nnc: 1\nnames: [x]\n"
        )

        result = trainer.train(
            dataset_dir, checkpoint_dir, parent_model_path=nonexistent
        )
        assert result.success is False
        err = result.error.lower()
        assert "not found" in err or "does not exist" in err or "missing" in err


# ──────────────────────────────────────────────────────────────────────
# Fix: Worker heartbeat
# ──────────────────────────────────────────────────────────────────────
class TestWorkerHeartbeat:
    def test_execute_training_has_heartbeat_logic(self) -> None:
        import inspect
        from backend.app.workers.train_worker import execute_training
        source = inspect.getsource(execute_training)
        assert "heartbeat" in source, (
            "execute_training should call heartbeat() during training"
        )


# ──────────────────────────────────────────────────────────────────────
# Fix: Worker checks cancellation_requested
# ──────────────────────────────────────────────────────────────────────
class TestWorkerChecksCancellation:
    def test_run_training_checks_cancellation_requested(self) -> None:
        import inspect
        from backend.app.workers.train_worker import run_training
        source = inspect.getsource(run_training)
        assert "cancellation_requested" in source, (
            "run_training should check cancellation_requested"
        )
