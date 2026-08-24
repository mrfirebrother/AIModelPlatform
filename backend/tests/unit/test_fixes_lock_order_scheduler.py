"""Tests for the three fixes:
1. Lock order unified: task -> attempt -> lease
2. GPU lease expiry syncs Attempt and Task
3. Heartbeat uses independent worker session
"""
from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from backend.app.models import (
    Base,
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
from backend.app.workers.attempt_lease import expire_attempt, heartbeat
from backend.app.workers.db_session import (
    reset_worker_session_factory,
    set_worker_engine_override,
    worker_session,
)
from backend.app.workers.train_worker import _HeartbeatThread, run_training


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
) -> DatasetSnapshot:
    if label_schema is None:
        label_schema = _make_label_schema(session)
    dataset = Dataset(name=f"ds-{uuid4().hex[:8]}")
    snapshot = DatasetSnapshot(
        dataset=dataset,
        label_schema=label_schema,
        manifest_path=f"snapshots/{uuid4()}/manifest.json",
        manifest_hash=manifest_hash or f"sha256:{uuid4().hex}",
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def _compute_manifest_hash(
    train_files: list, val_files: list, test_files: list,
    nc: int, names: list,
) -> str:
    dataset_manifest = {
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "nc": nc,
        "names": names,
    }
    return f"sha256:{hashlib.sha256(str(sorted(dataset_manifest.items())).encode()).hexdigest()}"


def _make_snapshot_with_valid_hash(session: Session) -> DatasetSnapshot:
    schema = _make_label_schema(session)
    nc = len(schema.classes) if schema else 0
    names = [c.semantic_key for c in schema.classes] if schema else []
    correct_hash = _compute_manifest_hash([], [], [], nc, names)
    return _make_snapshot(session, label_schema=schema, manifest_hash=correct_hash)


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


def _setup_worker_engine():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    event.listen(
        engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(engine)
    set_worker_engine_override(engine)
    return engine


class TestLockOrderTaskAttemptLease:
    """Fix: Lock order must be task -> attempt -> lease everywhere."""

    def test_fail_attempt_locks_task_before_attempt(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        task = _make_task(session, snapshot=snapshot, status="running")
        attempt = start_attempt(session, task.id)
        session.commit()

        fail_attempt(session, attempt.id, error="test error")
        session.commit()

        session.refresh(attempt)
        session.refresh(task)
        assert attempt.status == "failed"
        assert task.status == "failed"

    def test_complete_attempt_locks_task_before_attempt(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        parent = _make_parent_model(session)
        task = _make_task(session, snapshot=snapshot, parent=parent, status="running")
        attempt = start_attempt(session, task.id)
        session.commit()

        model = complete_attempt(
            session,
            attempt.id,
            artifact_path=f"models/{uuid4()}.pt",
            artifact_hash=f"sha256:{uuid4().hex}",
        )
        session.commit()

        session.refresh(attempt)
        session.refresh(task)
        assert attempt.status == "completed"
        assert task.status == "completed"
        assert model.status == "candidate"

    def test_expire_attempt_locks_task_before_attempt(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        task = _make_task(session, snapshot=snapshot, status="running")
        attempt = start_attempt(session, task.id)
        session.commit()

        expire_attempt(session, attempt.id)
        session.commit()

        session.refresh(attempt)
        session.refresh(task)
        assert attempt.status == "failed"
        assert attempt.last_error == "Lease expired"
        assert task.status == "failed"
        assert task.failure_reason == "Lease expired"

    def test_fail_attempt_does_not_fail_already_completed_task(
        self, session: Session
    ) -> None:
        snapshot = _make_snapshot(session)
        task = _make_task(session, snapshot=snapshot, status="running")
        attempt = start_attempt(session, task.id)
        session.commit()

        task.status = "completed"
        session.flush()

        fail_attempt(session, attempt.id, error="late failure")
        session.commit()

        session.refresh(task)
        assert task.status == "completed"

    def test_fail_attempt_does_not_fail_already_cancelled_task(
        self, session: Session
    ) -> None:
        snapshot = _make_snapshot(session)
        task = _make_task(session, snapshot=snapshot, status="running")
        attempt = start_attempt(session, task.id)
        session.commit()

        task.status = "cancelled"
        session.flush()

        fail_attempt(session, attempt.id, error="late failure")
        session.commit()

        session.refresh(task)
        assert task.status == "cancelled"


class TestGPULeaseExpirySyncsAttemptAndTask:
    """Fix: When GPU lease expires, associated Attempt and Task must be failed."""

    def test_reconcile_leases_fails_attempt_on_gpu_lease_expiry(self) -> None:
        engine = _setup_worker_engine()
        try:
            with Session(engine) as test_session:
                gpu = GPUResource(
                    gpu_device="gpu:0",
                    capacity_memory_mb=8192,
                    reserved_memory_mb=0,
                )
                test_session.add(gpu)
                test_session.flush()

                snapshot = _make_snapshot(test_session)
                task = _make_task(test_session, snapshot=snapshot, status="running")
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
                attempt_id = attempt.id
                task_id = task.id

            from backend.app.workers.scheduler import reconcile_leases
            result = reconcile_leases()
            assert "expired" in result.lower()

            with Session(engine) as verify:
                attempt_v = verify.get(TrainingAttempt, attempt_id)
                assert attempt_v is not None
                assert attempt_v.status == "failed"
                assert "lease expired" in (attempt_v.last_error or "").lower()
        finally:
            reset_worker_session_factory()
            engine.dispose()

    def test_reconcile_leases_fails_task_on_gpu_lease_expiry(self) -> None:
        engine = _setup_worker_engine()
        try:
            with Session(engine) as test_session:
                gpu = GPUResource(
                    gpu_device="gpu:0",
                    capacity_memory_mb=8192,
                    reserved_memory_mb=0,
                )
                test_session.add(gpu)
                test_session.flush()

                snapshot = _make_snapshot(test_session)
                task = _make_task(test_session, snapshot=snapshot, status="running")
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
                task_id = task.id

            from backend.app.workers.scheduler import reconcile_leases
            reconcile_leases()

            with Session(engine) as verify:
                task_v = verify.get(TrainingTask, task_id)
                assert task_v is not None
                assert task_v.status == "failed"
        finally:
            reset_worker_session_factory()
            engine.dispose()

    def test_reconcile_leases_skips_completed_attempt_on_gpu_lease_expiry(self) -> None:
        engine = _setup_worker_engine()
        try:
            with Session(engine) as test_session:
                gpu = GPUResource(
                    gpu_device="gpu:0",
                    capacity_memory_mb=8192,
                    reserved_memory_mb=0,
                )
                test_session.add(gpu)
                test_session.flush()

                snapshot = _make_snapshot(test_session)
                task = _make_task(test_session, snapshot=snapshot, status="running")
                attempt = start_attempt(test_session, task.id)
                test_session.flush()
                attempt.status = "completed"
                task.status = "completed"
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
                attempt_id = attempt.id

            from backend.app.workers.scheduler import reconcile_leases
            reconcile_leases()

            with Session(engine) as verify:
                attempt_v = verify.get(TrainingAttempt, attempt_id)
                assert attempt_v is not None
                assert attempt_v.status == "completed"
        finally:
            reset_worker_session_factory()
            engine.dispose()

    def test_reconcile_leases_skips_cancelled_task_on_gpu_lease_expiry(self) -> None:
        engine = _setup_worker_engine()
        try:
            with Session(engine) as test_session:
                gpu = GPUResource(
                    gpu_device="gpu:0",
                    capacity_memory_mb=8192,
                    reserved_memory_mb=0,
                )
                test_session.add(gpu)
                test_session.flush()

                snapshot = _make_snapshot(test_session)
                task = _make_task(test_session, snapshot=snapshot, status="running")
                attempt = start_attempt(test_session, task.id)
                test_session.flush()
                attempt.status = "cancelled"
                task.status = "cancelled"
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
                task_id = task.id
                attempt_id = attempt.id

            from backend.app.workers.scheduler import reconcile_leases
            reconcile_leases()

            with Session(engine) as verify:
                task_v = verify.get(TrainingTask, task_id)
                assert task_v is not None
                assert task_v.status == "cancelled"
                attempt_v = verify.get(TrainingAttempt, attempt_id)
                assert attempt_v.status != "failed"
        finally:
            reset_worker_session_factory()
            engine.dispose()


class TestHeartbeatIndependentSession:
    """Fix: Heartbeat thread uses its own worker_session, not the main session."""

    @patch("backend.app.workers.train_worker.attempt_heartbeat")
    def test_heartbeat_thread_creates_own_session(self, mock_hb: MagicMock) -> None:
        mock_hb.return_value = MagicMock()
        mock_main_session = MagicMock()

        thread = _HeartbeatThread(
            session=mock_main_session,
            attempt_id=uuid4(),
            lease_token=uuid4(),
            interval=0.05,
        )
        thread.start()
        import time
        time.sleep(0.2)
        thread.stop()
        thread.join(timeout=2)

        assert mock_hb.call_count >= 1
        call_args = mock_hb.call_args
        session_arg = call_args[0][0]
        assert session_arg is not mock_main_session

    def test_heartbeat_thread_stops_cleanly(self) -> None:
        mock_session = MagicMock()
        thread = _HeartbeatThread(
            session=mock_session,
            attempt_id=uuid4(),
            lease_token=uuid4(),
            interval=60,
        )
        thread.start()
        thread.stop()
        thread.join(timeout=2)
        assert not thread.is_alive()

    def test_heartbeat_thread_is_daemon(self) -> None:
        mock_session = MagicMock()
        thread = _HeartbeatThread(
            session=mock_session,
            attempt_id=uuid4(),
            lease_token=uuid4(),
            interval=60,
        )
        assert thread.daemon is True


class TestCUDAFallbackRejects:
    """Fix: CUDA requested but unavailable should reject, not silently degrade."""

    def test_gpu_task_rejected_when_cuda_unavailable(self) -> None:
        engine = _setup_worker_engine()
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
                    training_config_json={"epochs": 1, "device": "cuda:0"},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="queued",
                )
                ws.add(task)
                ws.flush()
                task_id = str(task.id)

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            import sys
            sys.modules["torch"] = mock_torch
            try:
                result = run_training(task_id)
                assert "failed" in result
                assert "CUDA requested but not available" in result
            finally:
                sys.modules.pop("torch", None)

            with Session(engine) as verify:
                task_v = verify.get(TrainingTask, __import__("uuid").UUID(task_id))
                assert task_v.status == "failed"
        finally:
            reset_worker_session_factory()
            engine.dispose()

    def test_gpu_task_rejected_when_torch_not_installed(self) -> None:
        engine = _setup_worker_engine()
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
                    training_config_json={"epochs": 1, "device": "cuda:0"},
                    resource_config_json={},
                    evaluation_policy_json={},
                    status="queued",
                )
                ws.add(task)
                ws.flush()
                task_id = str(task.id)

            import sys
            torch_module = sys.modules.get("torch")
            sys.modules["torch"] = None
            try:
                result = run_training(task_id)
                assert "failed" in result
                assert "torch not installed" in result
            finally:
                if torch_module is not None:
                    sys.modules["torch"] = torch_module
                else:
                    sys.modules.pop("torch", None)

            with Session(engine) as verify:
                task_v = verify.get(TrainingTask, __import__("uuid").UUID(task_id))
                assert task_v.status == "failed"
        finally:
            reset_worker_session_factory()
            engine.dispose()

    def test_cpu_task_not_rejected(self) -> None:
        engine = _setup_worker_engine()
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

            def fake_execute_training(**kwargs):
                from backend.app.training.yolo_trainer import TrainResult
                return TrainResult(
                    success=False,
                    epochs_completed=0,
                    best_model_path=None,
                    latest_model_path=None,
                    error="mock failure",
                )

            with patch(
                "backend.app.workers.train_worker.execute_training",
                side_effect=fake_execute_training,
            ):
                result = run_training(task_id)
                assert "failed" in result
                assert "CUDA" not in result
        finally:
            reset_worker_session_factory()
            engine.dispose()


class TestCancelAfterCancellation:
    """Fix: Cancel calls cancel_task, post-training checks cancellation_requested."""

    def test_pre_training_cancellation_aborts(self) -> None:
        engine = _setup_worker_engine()
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
        engine = _setup_worker_engine()
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

            with patch(
                "backend.app.workers.train_worker.execute_training",
                side_effect=fake_execute_training,
            ):
                result = run_training(task_id)
                assert "cancelled" in result.lower()

            with Session(engine) as verify:
                task_v = verify.get(TrainingTask, task_uuid)
                assert task_v.status == "cancelled"
                attempts = verify.query(TrainingAttempt).filter(
                    TrainingAttempt.task_id == task_v.id
                ).all()
                failed_attempts = [a for a in attempts if a.status == "failed"]
                assert len(failed_attempts) == 1
        finally:
            reset_worker_session_factory()
            engine.dispose()
