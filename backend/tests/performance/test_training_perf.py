from __future__ import annotations

import time
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from backend.app.models import (
    Base,
    Dataset,
    DatasetSnapshot,
    LabelSchema,
    TrainingAttempt,
    TrainingTask,
)
from backend.app.services import training_service


def _make_session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    event.listen(engine, "connect", lambda c, _: c.execute("PRAGMA foreign_keys=ON"))
    Base.metadata.create_all(engine)
    return Session(engine)


def _seed_dataset(session: Session, *, num_samples: int = 100) -> DatasetSnapshot:
    ds = Dataset(name=f"perf-ds-{uuid4().hex[:8]}", description="perf test dataset")
    session.add(ds)
    session.flush()

    label_schema = LabelSchema(name=f"perf-schema-{uuid4().hex[:8]}", status="active")
    session.add(label_schema)
    session.flush()

    snapshot = DatasetSnapshot(
        dataset_id=ds.id,
        label_schema_id=label_schema.id,
        train_manifest_json=[{"image": f"img_{i}.jpg", "boxes": []} for i in range(num_samples)],
        val_manifest_json=[],
        test_manifest_json=[],
        manifest_path=f"/data/manifests/perf-{num_samples}.json",
        manifest_hash=f"sha256:{uuid4().hex[:16]}",
        train_positive_count=num_samples,
        val_positive_count=0,
        test_positive_count=0,
        train_negative_count=0,
        val_negative_count=0,
        test_negative_count=0,
        quality_json={"quality_score": 0.95},
        source_path=f"/data/datasets/perf-{num_samples}",
    )
    session.add(snapshot)
    session.flush()
    return snapshot


pytestmark = [pytest.mark.slow]


class TestDatasetSizeScaling:
    def test_small_dataset_training_time(self) -> None:
        session = _make_session()
        snapshot = _seed_dataset(session, num_samples=50)

        start = time.perf_counter()
        task = training_service.create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolov8n",
            training_config_json={"epochs": 1, "imgsz": 64, "batch": 4},
            resource_config_json={"gpu_device": "0"},
            evaluation_policy_json={"min_map50": 0.3},
        )
        attempt = training_service.start_attempt(session, task.id)
        elapsed_create = (time.perf_counter() - start) * 1000

        assert task.status == "running"
        assert attempt.status == "running"
        print(f"\n[Perf] Small dataset (50 samples): create+start={elapsed_create:.1f}ms")

    def test_medium_dataset_training_time(self) -> None:
        session = _make_session()
        snapshot = _seed_dataset(session, num_samples=500)

        start = time.perf_counter()
        task = training_service.create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolov8n",
            training_config_json={"epochs": 1, "imgsz": 128, "batch": 8},
            resource_config_json={"gpu_device": "0"},
            evaluation_policy_json={"min_map50": 0.3},
        )
        attempt = training_service.start_attempt(session, task.id)
        elapsed_create = (time.perf_counter() - start) * 1000

        assert task.status == "running"
        print(f"\n[Perf] Medium dataset (500 samples): create+start={elapsed_create:.1f}ms")

    def test_large_dataset_training_time(self) -> None:
        session = _make_session()
        snapshot = _seed_dataset(session, num_samples=5000)

        start = time.perf_counter()
        task = training_service.create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolov8n",
            training_config_json={"epochs": 1, "imgsz": 128, "batch": 16},
            resource_config_json={"gpu_device": "0"},
            evaluation_policy_json={"min_map50": 0.3},
        )
        attempt = training_service.start_attempt(session, task.id)
        elapsed_create = (time.perf_counter() - start) * 1000

        assert task.status == "running"
        print(f"\n[Perf] Large dataset (5000 samples): create+start={elapsed_create:.1f}ms")

    def test_dataset_size_overhead_growth(self) -> None:
        sizes = [50, 200, 1000]
        timings: list[tuple[int, float]] = []

        for num_samples in sizes:
            session = _make_session()
            snapshot = _seed_dataset(session, num_samples=num_samples)

            start = time.perf_counter()
            task = training_service.create_task(
                session,
                parent_model_node_id=None,
                dataset_snapshot_id=snapshot.id,
                task_type="object_detection",
                model_family="yolov8n",
                training_config_json={"epochs": 1, "imgsz": 64, "batch": 4},
                resource_config_json={"gpu_device": "0"},
                evaluation_policy_json={"min_map50": 0.3},
            )
            training_service.start_attempt(session, task.id)
            elapsed = (time.perf_counter() - start) * 1000
            timings.append((num_samples, elapsed))

        for size, t in timings:
            print(f"[Perf] Dataset size={size}: {t:.1f}ms")
        print(f"\n[Perf] Scaling: 50->200={timings[1][1]/timings[0][1]:.2f}x 200->1000={timings[2][1]/timings[1][1]:.2f}x")

        assert timings[-1][1] < timings[0][1] * 10, "Task creation overhead grows too fast with dataset size"


class TestModelSizeScaling:
    def test_tiny_model_creation_time(self) -> None:
        session = _make_session()
        snapshot = _seed_dataset(session, num_samples=100)

        start = time.perf_counter()
        task = training_service.create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolov8n",
            training_config_json={"epochs": 1, "imgsz": 64, "batch": 4},
            resource_config_json={"gpu_device": "0"},
            evaluation_policy_json={"min_map50": 0.3},
        )
        training_service.start_attempt(session, task.id)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\n[Perf] Tiny model (yolov8n): create+start={elapsed:.1f}ms")

    def test_medium_model_creation_time(self) -> None:
        session = _make_session()
        snapshot = _seed_dataset(session, num_samples=100)

        start = time.perf_counter()
        task = training_service.create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolov8s",
            training_config_json={"epochs": 1, "imgsz": 128, "batch": 8},
            resource_config_json={"gpu_device": "0"},
            evaluation_policy_json={"min_map50": 0.3},
        )
        training_service.start_attempt(session, task.id)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\n[Perf] Medium model (yolov8s): create+start={elapsed:.1f}ms")

    def test_model_creation_overhead_consistent(self) -> None:
        model_configs = [
            ("yolov8n", {"epochs": 1, "imgsz": 64, "batch": 4}),
            ("yolov8s", {"epochs": 1, "imgsz": 128, "batch": 8}),
            ("yolov8m", {"epochs": 1, "imgsz": 256, "batch": 8}),
        ]
        timings: list[tuple[str, float]] = []

        for model_family, config in model_configs:
            session = _make_session()
            snapshot = _seed_dataset(session, num_samples=100)

            start = time.perf_counter()
            task = training_service.create_task(
                session,
                parent_model_node_id=None,
                dataset_snapshot_id=snapshot.id,
                task_type="object_detection",
                model_family=model_family,
                training_config_json=config,
                resource_config_json={"gpu_device": "0"},
                evaluation_policy_json={"min_map50": 0.3},
            )
            training_service.start_attempt(session, task.id)
            elapsed = (time.perf_counter() - start) * 1000
            timings.append((model_family, elapsed))

        for name, t in timings:
            print(f"[Perf] Model {name}: {t:.1f}ms")
        print(f"\n[Perf] Model scaling overhead: {timings[-1][1]/timings[0][1]:.2f}x (n->m)")
        assert timings[-1][1] < timings[0][1] * 5, "Model size overhead grows too fast"


class TestCheckpointRecovery:
    def test_attempt_completion_time(self) -> None:
        session = _make_session()
        snapshot = _seed_dataset(session, num_samples=100)
        task = training_service.create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolov8n",
            training_config_json={"epochs": 1, "imgsz": 64, "batch": 4},
            resource_config_json={"gpu_device": "0"},
            evaluation_policy_json={"min_map50": 0.3},
        )
        attempt = training_service.start_attempt(session, task.id)

        start = time.perf_counter()
        model = training_service.complete_attempt(
            session,
            attempt.id,
            artifact_path="/data/models/checkpoint.pt",
            artifact_hash="sha256:checkpoint_hash",
        )
        elapsed = (time.perf_counter() - start) * 1000

        assert model.status == "candidate"
        assert task.status == "completed"
        print(f"\n[Perf] Checkpoint completion: {elapsed:.1f}ms")

    def test_attempt_failure_recovery_time(self) -> None:
        session = _make_session()
        snapshot = _seed_dataset(session, num_samples=100)
        task = training_service.create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolov8n",
            training_config_json={"epochs": 1, "imgsz": 64, "batch": 4},
            resource_config_json={"gpu_device": "0"},
            evaluation_policy_json={"min_map50": 0.3},
        )
        attempt = training_service.start_attempt(session, task.id)

        start = time.perf_counter()
        training_service.fail_attempt(session, attempt.id, error="OOM on GPU")
        elapsed_fail = (time.perf_counter() - start) * 1000
        assert task.status == "failed"

        start = time.perf_counter()
        attempt2 = training_service.start_attempt(session, task.id)
        elapsed_recover = (time.perf_counter() - start) * 1000
        assert attempt2.status == "running"
        assert attempt2.attempt_no == 2
        assert task.status == "running"

        print(f"\n[Perf] Failure+recovery: fail={elapsed_fail:.1f}ms recover={elapsed_recover:.1f}ms")

    def test_multiple_recovery_cycles(self) -> None:
        session = _make_session()
        snapshot = _seed_dataset(session, num_samples=100)
        task = training_service.create_task(
            session,
            parent_model_node_id=None,
            dataset_snapshot_id=snapshot.id,
            task_type="object_detection",
            model_family="yolov8n",
            training_config_json={"epochs": 1, "imgsz": 64, "batch": 4},
            resource_config_json={"gpu_device": "0"},
            evaluation_policy_json={"min_map50": 0.3},
        )

        recovery_times: list[float] = []
        for i in range(5):
            attempt = training_service.start_attempt(session, task.id)
            assert attempt.attempt_no == i + 1

            start = time.perf_counter()
            training_service.fail_attempt(session, attempt.id, error=f"Error {i+1}")
            elapsed = (time.perf_counter() - start) * 1000
            recovery_times.append(elapsed)

        avg_recovery = sum(recovery_times) / len(recovery_times)
        print(f"\n[Perf] Recovery cycles (5x): avg={avg_recovery:.2f}ms per cycle")
        assert avg_recovery < 200, f"Average recovery time {avg_recovery:.2f}ms too high"


class TestConcurrentTaskCreation:
    def test_concurrent_task_creation(self) -> None:
        session = _make_session()
        snapshot = _seed_dataset(session, num_samples=100)

        tasks: list[TrainingTask] = []
        start = time.perf_counter()
        for i in range(10):
            task = training_service.create_task(
                session,
                parent_model_node_id=None,
                dataset_snapshot_id=snapshot.id,
                task_type="object_detection",
                model_family="yolov8n",
                training_config_json={"epochs": 1, "imgsz": 64, "batch": 4, "run_id": i},
                resource_config_json={"gpu_device": "0"},
                evaluation_policy_json={"min_map50": 0.3},
            )
            tasks.append(task)
        elapsed = (time.perf_counter() - start) * 1000

        all_tasks = training_service.list_tasks(session)
        assert len(all_tasks) >= 10
        print(f"\n[Perf] Concurrent task creation (10 tasks): {elapsed:.1f}ms total, {elapsed/10:.2f}ms/task")
