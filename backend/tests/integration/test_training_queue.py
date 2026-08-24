from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
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
    LabelSchema,
    TrainingAttempt,
    TrainingTask,
)
from backend.app.runtime.gpu_monitor import GpuMonitor
from backend.app.training.resource_scheduler import ResourceScheduler
from backend.app.training.yolo_trainer import TrainResult
from backend.app.workers.train_worker import execute_training


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


@pytest.fixture()
def cpu_monitor() -> GpuMonitor:
    return GpuMonitor(cpu_mode=True)


@pytest.fixture()
def gpu_resource(session: Session) -> GPUResource:
    resource = GPUResource(
        gpu_device="0",
        capacity_memory_mb=8192,
        reserved_memory_mb=0,
    )
    session.add(resource)
    session.flush()
    return resource


def _create_label_schema(session: Session) -> LabelSchema:
    schema = LabelSchema(
        name="test-schema",
        status="active",
        definition_sealed=False,
    )
    session.add(schema)
    session.flush()
    return schema


def _create_dataset(session: Session) -> Dataset:
    dataset = Dataset(
        name="test-dataset",
        metadata_json={},
    )
    session.add(dataset)
    session.flush()
    return dataset


def _create_dataset_snapshot(
    session: Session, dataset: Dataset, label_schema: LabelSchema
) -> DatasetSnapshot:
    snapshot = DatasetSnapshot(
        dataset_id=dataset.id,
        label_schema_id=label_schema.id,
        train_manifest_json=[],
        val_manifest_json=[],
        test_manifest_json=[],
        manifest_path="/tmp/manifest.json",
        manifest_hash="abc123",
        train_positive_count=0,
        val_positive_count=0,
        test_positive_count=0,
        train_negative_count=0,
        val_negative_count=0,
        test_negative_count=0,
        quality_json={},
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def _create_training_task(
    session: Session, snapshot: DatasetSnapshot
) -> TrainingTask:
    task = TrainingTask(
        dataset_snapshot_id=snapshot.id,
        task_type="detection",
        model_family="yolo",
        training_config_json={"epochs": 1, "model_name": "yolov8n"},
        resource_config_json={"memory_mb": 2048},
        evaluation_policy_json={},
        status="queued",
    )
    session.add(task)
    session.flush()
    return task


def _create_training_attempt(
    session: Session, task: TrainingTask
) -> TrainingAttempt:
    attempt = TrainingAttempt(
        task_id=task.id,
        attempt_no=1,
        status="running",
    )
    session.add(attempt)
    session.flush()
    return attempt


class TestTrainingQueueIntegration:
    def test_training_with_gpu_reservation(
        self, session: Session, gpu_resource: GPUResource
    ) -> None:
        label_schema = _create_label_schema(session)
        dataset = _create_dataset(session)
        snapshot = _create_dataset_snapshot(session, dataset, label_schema)
        task = _create_training_task(session, snapshot)
        attempt = _create_training_attempt(session, task)

        mock_result = TrainResult(
            success=True,
            epochs_completed=1,
            best_model_path="/tmp/best.pt",
            latest_model_path="/tmp/last.pt",
            metrics={"mAP50": 0.95},
        )

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            dataset_manifest = {
                "train_files": [],
                "val_files": [],
                "test_files": [],
                "nc": 1,
                "names": ["object"],
            }

            with patch(
                "backend.app.workers.train_worker.YoloTrainer"
            ) as MockTrainer:
                instance = MockTrainer.return_value
                instance.train.return_value = mock_result

                result = execute_training(
                    session=session,
                    task=task,
                    attempt=attempt,
                    training_config={
                        "epochs": 1,
                        "model_name": "yolov8n",
                        "device": "cpu",
                    },
                    dataset_manifest=dataset_manifest,
                    output_dir=output_dir,
                    required_memory_mb=2048,
                )

        assert isinstance(result, TrainResult)
        assert result.success is True
        assert attempt.status == "running"

    def test_gpu_lease_released_after_training(
        self, session: Session, gpu_resource: GPUResource
    ) -> None:
        label_schema = _create_label_schema(session)
        dataset = _create_dataset(session)
        snapshot = _create_dataset_snapshot(session, dataset, label_schema)
        task = _create_training_task(session, snapshot)
        attempt = _create_training_attempt(session, task)

        mock_result = TrainResult(
            success=True,
            epochs_completed=1,
            best_model_path=None,
            latest_model_path=None,
        )

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            dataset_manifest = {
                "train_files": [],
                "val_files": [],
                "test_files": [],
                "nc": 1,
                "names": ["object"],
            }

            with patch(
                "backend.app.workers.train_worker.YoloTrainer"
            ) as MockTrainer:
                instance = MockTrainer.return_value
                instance.train.return_value = mock_result

                execute_training(
                    session=session,
                    task=task,
                    attempt=attempt,
                    training_config={
                        "epochs": 1,
                        "model_name": "yolov8n",
                        "device": "cpu",
                    },
                    dataset_manifest=dataset_manifest,
                    output_dir=output_dir,
                    required_memory_mb=2048,
                )

        scheduler = ResourceScheduler(session=session, cpu_mode=True)
        available = scheduler._calculate_available_memory(gpu_resource)
        assert available == gpu_resource.capacity_memory_mb

    def test_multiple_sequential_trainings(
        self, session: Session, gpu_resource: GPUResource
    ) -> None:
        label_schema = _create_label_schema(session)
        dataset = _create_dataset(session)
        snapshot = _create_dataset_snapshot(session, dataset, label_schema)

        mock_result = TrainResult(
            success=True,
            epochs_completed=1,
            best_model_path=None,
            latest_model_path=None,
        )

        for i in range(3):
            task = _create_training_task(session, snapshot)
            attempt = _create_training_attempt(session, task)

            with TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                dataset_manifest = {
                    "train_files": [],
                    "val_files": [],
                    "test_files": [],
                    "nc": 1,
                    "names": ["object"],
                }

                with patch(
                    "backend.app.workers.train_worker.YoloTrainer"
                ) as MockTrainer:
                    instance = MockTrainer.return_value
                    instance.train.return_value = mock_result

                    result = execute_training(
                        session=session,
                        task=task,
                        attempt=attempt,
                        training_config={
                            "epochs": 1,
                            "model_name": "yolov8n",
                            "device": "cpu",
                        },
                        dataset_manifest=dataset_manifest,
                        output_dir=output_dir,
                        required_memory_mb=2048,
                    )

                    assert isinstance(result, TrainResult)
                    assert result.success is True

    def test_concurrent_training_same_attempt_rejected(
        self, session: Session, gpu_resource: GPUResource
    ) -> None:
        label_schema = _create_label_schema(session)
        dataset = _create_dataset(session)
        snapshot = _create_dataset_snapshot(session, dataset, label_schema)
        task = _create_training_task(session, snapshot)
        attempt = _create_training_attempt(session, task)

        scheduler = ResourceScheduler(session=session, cpu_mode=True)
        scheduler.reserve_gpu(attempt.id, required_memory_mb=2048)

        with pytest.raises(Exception, match="already has active lease"):
            scheduler.reserve_gpu(attempt.id, required_memory_mb=2048)
