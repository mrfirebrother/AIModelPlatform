"""Build deterministic test fixtures for end-to-end and integration tests.

This module provides factory functions that produce in-memory database records
matching the AI Model Platform schema.  All IDs are deterministic UUIDs derived
from fixed seeds so that tests are reproducible and not dependent on runtime
state.

Usage::

    from backend.tests.fixtures.build_fixtures import FixtureBuilder
    builder = FixtureBuilder()
    root_model = builder.root_model()
    schema = builder.first_gen_label_schema()
    ...
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from backend.app.models import (
    BindingRelease,
    Dataset,
    DatasetSnapshot,
    Evaluation,
    LabelSchema,
    LabelSchemaClass,
    ModelBinding,
    ModelNode,
    RuntimeInstance,
    TrainingAttempt,
    TrainingTask,
)

# ---------------------------------------------------------------------------
# Deterministic UUID generation
# ---------------------------------------------------------------------------

_SEED_PREFIX = "ai-platform-test-fixture"


def _deterministic_uuid(name: str) -> uuid.UUID:
    h = hashlib.sha256(f"{_SEED_PREFIX}:{name}".encode()).digest()
    return uuid.UUID(bytes=h[:16], version=4)


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Dataclass wrappers for readable fixture references
# ---------------------------------------------------------------------------

@dataclass
class RootModelFixture:
    model_node: ModelNode
    label_schema: LabelSchema
    model_path: str
    model_hash: str


@dataclass
class FirstGenDatasetFixture:
    dataset: Dataset
    snapshot: DatasetSnapshot
    label_schema: LabelSchema
    source_path: str


@dataclass
class AppendCategoryDatasetFixture:
    dataset: Dataset
    snapshot: DatasetSnapshot
    label_schema: LabelSchema
    source_path: str


@dataclass
class InvalidDatasetFixture:
    dataset: Dataset
    snapshot: DatasetSnapshot | None
    label_schema: LabelSchema
    source_path: str
    error_reason: str


@dataclass
class FailedModelLoadFixture:
    model_node: ModelNode
    label_schema: LabelSchema
    model_path: str


@dataclass
class InsufficientResourceFixture:
    config: dict[str, Any]
    gpu_device: str
    reserved_mb: int
    available_mb: int


@dataclass
class ExpiredLeaseFixture:
    task: TrainingTask
    attempt: TrainingAttempt
    lease_token: uuid.UUID


@dataclass
class ConcurrentSwitchFixture:
    binding: ModelBinding
    model_a: ModelNode
    model_b: ModelNode
    release_a: BindingRelease
    release_b: BindingRelease


@dataclass
class ServiceRestartFixture:
    binding: ModelBinding
    model: ModelNode
    release: BindingRelease
    instance: RuntimeInstance


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class FixtureBuilder:
    """Creates deterministic test fixtures in a given SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._counter = 0

    def _next_int(self) -> int:
        self._counter += 1
        return self._counter

    # ------------------------------------------------------------------
    # Root model
    # ------------------------------------------------------------------

    def root_model(
        self,
        *,
        name: str = "YOLOv8n-root",
        parent_id: uuid.UUID | None = None,
        label_schema: LabelSchema | None = None,
    ) -> RootModelFixture:
        schema = label_schema or self._create_label_schema(
            name=f"schema-root-{self._next_int()}",
            classes=[
                (0, "person", "Person"),
                (1, "car", "Car"),
                (2, "truck", "Truck"),
            ],
        )
        model_path = f"/data/models/{name}.pt"
        model_hash = f"sha256:{_sha256_hex(name)}"
        model = ModelNode(
            id=_deterministic_uuid(f"root-model-{name}"),
            parent_id=parent_id,
            label_schema_id=schema.id,
            task_type="object_detection",
            model_family="yolo",
            artifact_path=model_path,
            artifact_hash=model_hash,
            framework="pytorch",
            artifact_format="pt",
            status="approved",
            metadata_json={
                "description": f"Root model fixture: {name}",
                "inputSize": [640, 640],
                "labelOrder": ["person", "car", "truck"],
            },
        )
        self._session.add(model)
        self._session.flush()
        return RootModelFixture(
            model_node=model,
            label_schema=schema,
            model_path=model_path,
            model_hash=model_hash,
        )

    # ------------------------------------------------------------------
    # First-generation label schema (new labels not from root)
    # ------------------------------------------------------------------

    def first_gen_label_schema(self) -> LabelSchema:
        return self._create_label_schema(
            name=f"schema-firstgen-{self._next_int()}",
            classes=[
                (0, "fire", "Fire"),
                (1, "smoke", "Smoke"),
            ],
        )

    # ------------------------------------------------------------------
    # First-generation dataset
    # ------------------------------------------------------------------

    def first_gen_dataset(
        self,
        label_schema: LabelSchema | None = None,
    ) -> FirstGenDatasetFixture:
        schema = label_schema or self.first_gen_label_schema()
        dataset = Dataset(
            id=_deterministic_uuid(f"dataset-firstgen-{self._next_int()}"),
            name=f"dataset-fire-{self._next_int()}",
            description="Fire detection training data",
            metadata_json={"source": "fixture"},
        )
        self._session.add(dataset)
        self._session.flush()

        train_files = [
            {"path": "images/train/img_001.jpg", "label": "labels/train/img_001.txt", "width": 640, "height": 480},
            {"path": "images/train/img_002.jpg", "label": "labels/train/img_002.txt", "width": 640, "height": 480},
            {"path": "images/train/img_003.jpg", "label": "labels/train/img_003.txt", "width": 640, "height": 480},
        ]
        val_files = [
            {"path": "images/val/img_101.jpg", "label": "labels/val/img_101.txt", "width": 640, "height": 480},
            {"path": "images/val/img_102.jpg", "label": "labels/val/img_102.txt", "width": 640, "height": 480},
        ]
        test_files = [
            {"path": "images/test/img_201.jpg", "label": "labels/test/img_201.txt", "width": 640, "height": 480},
        ]

        source_path = f"/data/datasets/import/fire-{self._counter}"
        snapshot = DatasetSnapshot(
            id=_deterministic_uuid(f"snapshot-firstgen-{self._counter}"),
            dataset_id=dataset.id,
            label_schema_id=schema.id,
            train_manifest_json=train_files,
            val_manifest_json=val_files,
            test_manifest_json=test_files,
            manifest_path=f"/data/snapshots/manifest-{self._counter}.json",
            manifest_hash=f"sha256:{_sha256_hex(f'manifest-{self._counter}')}",
            train_positive_count=len(train_files),
            val_positive_count=len(val_files),
            test_positive_count=len(test_files),
            train_negative_count=0,
            val_negative_count=0,
            test_negative_count=0,
            quality_json={"warnings": [], "errors": []},
            source_path=source_path,
        )
        self._session.add(snapshot)
        self._session.flush()

        return FirstGenDatasetFixture(
            dataset=dataset,
            snapshot=snapshot,
            label_schema=schema,
            source_path=source_path,
        )

    # ------------------------------------------------------------------
    # Append-category dataset (inherits parent + adds new class)
    # ------------------------------------------------------------------

    def append_category_dataset(
        self,
        parent_schema: LabelSchema,
    ) -> AppendCategoryDatasetFixture:
        child_schema = self._create_label_schema(
            name=f"schema-append-{self._next_int()}",
            parent_schema_id=parent_schema.id,
            classes=[
                *[
                    (cls.class_id, cls.semantic_key, cls.display_name)
                    for cls in parent_schema.classes
                ],
                (len(parent_schema.classes), "rebar_exposure", "Rebar Exposure"),
            ],
        )

        dataset = Dataset(
            id=_deterministic_uuid(f"dataset-append-{self._next_int()}"),
            name=f"dataset-bridge-concrete-{self._next_int()}",
            description="Bridge concrete damage with inherited labels",
            metadata_json={"source": "fixture-append"},
        )
        self._session.add(dataset)
        self._session.flush()

        train_files = [
            {"path": "images/train/bridge_001.jpg", "label": "labels/train/bridge_001.txt", "width": 800, "height": 600},
            {"path": "images/train/bridge_002.jpg", "label": "labels/train/bridge_002.txt", "width": 800, "height": 600},
        ]
        val_files = [
            {"path": "images/val/bridge_101.jpg", "label": "labels/val/bridge_101.txt", "width": 800, "height": 600},
        ]
        test_files: list[dict[str, Any]] = []

        source_path = f"/data/datasets/import/bridge-concrete-{self._counter}"
        snapshot = DatasetSnapshot(
            id=_deterministic_uuid(f"snapshot-append-{self._counter}"),
            dataset_id=dataset.id,
            label_schema_id=child_schema.id,
            train_manifest_json=train_files,
            val_manifest_json=val_files,
            test_manifest_json=test_files,
            manifest_path=f"/data/snapshots/manifest-append-{self._counter}.json",
            manifest_hash=f"sha256:{_sha256_hex(f'manifest-append-{self._counter}')}",
            train_positive_count=len(train_files),
            val_positive_count=len(val_files),
            test_positive_count=0,
            train_negative_count=0,
            val_negative_count=0,
            test_negative_count=0,
            quality_json={"warnings": [], "errors": []},
            source_path=source_path,
        )
        self._session.add(snapshot)
        self._session.flush()

        return AppendCategoryDatasetFixture(
            dataset=dataset,
            snapshot=snapshot,
            label_schema=child_schema,
            source_path=source_path,
        )

    # ------------------------------------------------------------------
    # Invalid dataset (missing inherited classes in val split)
    # ------------------------------------------------------------------

    def invalid_dataset(
        self,
        parent_schema: LabelSchema,
    ) -> InvalidDatasetFixture:
        schema = self._create_label_schema(
            name=f"schema-invalid-{self._next_int()}",
            parent_schema_id=parent_schema.id,
            classes=[
                *[
                    (cls.class_id, cls.semantic_key, cls.display_name)
                    for cls in parent_schema.classes
                ],
                (len(parent_schema.classes), "spalling", "Spalling"),
            ],
        )

        dataset = Dataset(
            id=_deterministic_uuid(f"dataset-invalid-{self._next_int()}"),
            name=f"dataset-invalid-{self._next_int()}",
            description="Dataset missing inherited class coverage in val",
            metadata_json={"source": "fixture-invalid"},
        )
        self._session.add(dataset)
        self._session.flush()

        train_files = [
            {"path": "images/train/img_001.jpg", "label": "labels/train/img_001.txt", "width": 640, "height": 480},
        ]
        val_files: list[dict[str, Any]] = []
        test_files: list[dict[str, Any]] = []

        source_path = f"/data/datasets/import/invalid-{self._counter}"
        snapshot = DatasetSnapshot(
            id=_deterministic_uuid(f"snapshot-invalid-{self._counter}"),
            dataset_id=dataset.id,
            label_schema_id=schema.id,
            train_manifest_json=train_files,
            val_manifest_json=val_files,
            test_manifest_json=test_files,
            manifest_path=f"/data/snapshots/manifest-invalid-{self._counter}.json",
            manifest_hash=f"sha256:{_sha256_hex(f'manifest-invalid-{self._counter}')}",
            train_positive_count=1,
            val_positive_count=0,
            test_positive_count=0,
            train_negative_count=0,
            val_negative_count=0,
            test_negative_count=0,
            quality_json={
                "warnings": [],
                "errors": ["val split has 0 images for inherited class 'fire'"],
            },
            source_path=source_path,
        )
        self._session.add(snapshot)
        self._session.flush()

        return InvalidDatasetFixture(
            dataset=dataset,
            snapshot=snapshot,
            label_schema=schema,
            source_path=source_path,
            error_reason="val split missing inherited class coverage",
        )

    # ------------------------------------------------------------------
    # Failed model load candidate
    # ------------------------------------------------------------------

    def failed_model_load(
        self,
        parent_model_node_id: uuid.UUID | None = None,
        label_schema: LabelSchema | None = None,
    ) -> FailedModelLoadFixture:
        schema = label_schema or self._create_label_schema(
            name=f"schema-failload-{self._next_int()}",
            classes=[(0, "defect", "Defect")],
        )
        model_path = f"/data/models/corrupted-{self._next_int()}.pt"
        model = ModelNode(
            id=_deterministic_uuid(f"failed-model-{self._next_int()}"),
            parent_id=parent_model_node_id,
            label_schema_id=schema.id,
            task_type="object_detection",
            model_family="yolo",
            artifact_path=model_path,
            artifact_hash=f"sha256:{_sha256_hex(model_path)}",
            framework="pytorch",
            artifact_format="pt",
            status="candidate",
            metadata_json={"simulated": "load_failure"},
        )
        self._session.add(model)
        self._session.flush()
        return FailedModelLoadFixture(
            model_node=model,
            label_schema=schema,
            model_path=model_path,
        )

    # ------------------------------------------------------------------
    # Insufficient GPU resources
    # ------------------------------------------------------------------

    def insufficient_resources(self) -> InsufficientResourceFixture:
        return InsufficientResourceFixture(
            config={
                "gpu_device": "0",
                "reserved_memory_mb": 8192,
            },
            gpu_device="0",
            reserved_mb=8192,
            available_mb=4096,
        )

    # ------------------------------------------------------------------
    # Expired lease attempt
    # ------------------------------------------------------------------

    def expired_lease(
        self,
        parent_model_node_id: uuid.UUID | None = None,
        dataset_snapshot_id: uuid.UUID | None = None,
    ) -> ExpiredLeaseFixture:
        task = TrainingTask(
            id=_deterministic_uuid(f"task-expired-{self._next_int()}"),
            parent_model_node_id=parent_model_node_id,
            dataset_snapshot_id=dataset_snapshot_id or _deterministic_uuid("snapshot-placeholder"),
            task_type="object_detection",
            model_family="yolo",
            training_config_json={"epochs": 50, "batchSize": 16},
            resource_config_json={"gpu_device": "0"},
            evaluation_policy_json={"min_mAP50": 0.5},
            status="running",
        )
        self._session.add(task)
        self._session.flush()

        lease_token = _deterministic_uuid(f"lease-expired-{self._next_int()}")
        expired_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)

        attempt = TrainingAttempt(
            id=_deterministic_uuid(f"attempt-expired-{self._next_int()}"),
            task_id=task.id,
            attempt_no=1,
            status="running",
            gpu_device="0",
            current_epoch=10,
            lease_token=lease_token,
            fencing_token=1,
            heartbeat_at=expired_time,
            lease_expires_at=expired_time,
            started_at=now,
        )
        self._session.add(attempt)
        self._session.flush()

        return ExpiredLeaseFixture(
            task=task,
            attempt=attempt,
            lease_token=lease_token,
        )

    # ------------------------------------------------------------------
    # Concurrent switch (two candidates fighting for the same binding)
    # ------------------------------------------------------------------

    def concurrent_switch(self) -> ConcurrentSwitchFixture:
        binding = ModelBinding(
            id=_deterministic_uuid("binding-concurrent"),
            external_ref="concurrent-test-binding",
            status="bound",
        )
        self._session.add(binding)
        self._session.flush()

        schema = self._create_label_schema(
            name=f"schema-concurrent-{self._next_int()}",
            classes=[(0, "fire", "Fire")],
        )

        model_a = ModelNode(
            id=_deterministic_uuid("model-concurrent-a"),
            label_schema_id=schema.id,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="/data/models/concurrent-a.pt",
            artifact_hash=f"sha256:{_sha256_hex('concurrent-a')}",
            status="approved",
            metadata_json={"variant": "A"},
        )
        model_b = ModelNode(
            id=_deterministic_uuid("model-concurrent-b"),
            label_schema_id=schema.id,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="/data/models/concurrent-b.pt",
            artifact_hash=f"sha256:{_sha256_hex('concurrent-b')}",
            status="approved",
            metadata_json={"variant": "B"},
        )
        self._session.add(model_a)
        self._session.add(model_b)
        self._session.flush()

        release_a = BindingRelease(
            id=_deterministic_uuid("release-concurrent-a"),
            binding_id=binding.id,
            revision_no=1,
            model_node_id=model_a.id,
            inference_config_json={},
            inference_config_hash=f"sha256:{_sha256_hex('config-a')}",
            release_type="normal",
            status="active",
        )
        self._session.add(release_a)
        self._session.flush()

        binding.current_release_id = release_a.id
        self._session.flush()

        release_b = BindingRelease(
            id=_deterministic_uuid("release-concurrent-b"),
            binding_id=binding.id,
            revision_no=2,
            model_node_id=model_b.id,
            inference_config_json={},
            inference_config_hash=f"sha256:{_sha256_hex('config-b')}",
            release_type="normal",
            status="pending",
        )
        self._session.add(release_b)
        self._session.flush()

        return ConcurrentSwitchFixture(
            binding=binding,
            model_a=model_a,
            model_b=model_b,
            release_a=release_a,
            release_b=release_b,
        )

    # ------------------------------------------------------------------
    # Service restart (serving instance that survived a process restart)
    # ------------------------------------------------------------------

    def service_restart(self) -> ServiceRestartFixture:
        schema = self._create_label_schema(
            name=f"schema-restart-{self._next_int()}",
            classes=[(0, "smoke", "Smoke")],
        )

        model = ModelNode(
            id=_deterministic_uuid("model-restart"),
            label_schema_id=schema.id,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="/data/models/restart-model.pt",
            artifact_hash=f"sha256:{_sha256_hex('restart-model')}",
            status="approved",
        )
        self._session.add(model)
        self._session.flush()

        binding = ModelBinding(
            id=_deterministic_uuid("binding-restart"),
            external_ref="restart-test-binding",
            status="bound",
        )
        self._session.add(binding)
        self._session.flush()

        release = BindingRelease(
            id=_deterministic_uuid("release-restart"),
            binding_id=binding.id,
            revision_no=1,
            model_node_id=model.id,
            inference_config_json={"confidence": 0.6},
            inference_config_hash=f"sha256:{_sha256_hex('config-restart')}",
            release_type="normal",
            status="active",
        )
        self._session.add(release)
        self._session.flush()

        instance = RuntimeInstance(
            id=_deterministic_uuid("runtime-restart"),
            binding_id=binding.id,
            release_id=release.id,
            model_node_id=model.id,
            config_hash=f"sha256:{_sha256_hex('config-restart')}",
            generation=1,
            fencing_token=1,
            status="serving",
            gpu_device="0",
            reserved_memory_mb=4096,
            actual_memory_mb=3800,
            worker_id="worker-old-001",
            heartbeat_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        self._session.add(instance)
        self._session.flush()

        binding.current_release_id = release.id
        binding.current_runtime_instance_id = instance.id
        self._session.flush()

        return ServiceRestartFixture(
            binding=binding,
            model=model,
            release=release,
            instance=instance,
        )

    # ------------------------------------------------------------------
    # Evaluation fixture
    # ------------------------------------------------------------------

    def evaluation(
        self,
        model_node_id: uuid.UUID,
        dataset_snapshot_id: uuid.UUID,
        *,
        auto_status: str = "pending",
        human_status: str = "pending",
    ) -> Evaluation:
        ev = Evaluation(
            id=_deterministic_uuid(f"eval-{self._next_int()}"),
            model_node_id=model_node_id,
            dataset_snapshot_id=dataset_snapshot_id,
            auto_status=auto_status,
            human_status=human_status,
            evaluation_policy_json={
                "min_mAP50": 0.5,
                "min_precision": 0.6,
                "min_recall": 0.5,
                "max_regression_ratio": 0.1,
            },
            auto_metrics_json={},
        )
        self._session.add(ev)
        self._session.flush()
        return ev

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_label_schema(
        self,
        name: str,
        classes: list[tuple[int, str, str]],
        parent_schema_id: uuid.UUID | None = None,
    ) -> LabelSchema:
        schema = LabelSchema(
            id=_deterministic_uuid(f"schema-{name}"),
            name=name,
            parent_schema_id=parent_schema_id,
            status="active",
            definition_sealed=False,
        )
        self._session.add(schema)
        self._session.flush()

        for class_id, semantic_key, display_name in classes:
            cls = LabelSchemaClass(
                schema_id=schema.id,
                class_id=class_id,
                semantic_key=semantic_key,
                display_name=display_name,
                description=f"Fixture class {semantic_key}",
            )
            self._session.add(cls)
        self._session.flush()

        return schema
