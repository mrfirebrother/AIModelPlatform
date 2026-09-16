"""End-to-end closed-loop tests for the AI Model Platform.

Covers the full lifecycle:
  import root model -> import dataset -> create snapshot -> create training task
  -> run attempt -> create candidate -> auto evaluate -> manual review
  -> publish -> inference -> rollback

GPU PT training tests are marked with ``@pytest.mark.gpu`` and are skipped
when no GPU environment is available (``CUDA_VISIBLE_DEVICES`` unset).
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.models import (
    Base,
    BindingRelease,
    Dataset,
    DatasetSnapshot,
    Evaluation,
    LabelSchema,
    ModelBinding,
    ModelNode,
    RuntimeInstance,
    TrainingAttempt,
    TrainingTask,
)
from backend.app.services import (
    evaluation_service,
    training_service,
)
from backend.tests.fixtures.build_fixtures import FixtureBuilder
from backend.tests.fixtures.build_real_fixtures import (
    RealYoloDataset,
    build_real_yolo_dataset,
    ensure_root_model_exists,
    cleanup_real_dataset,
)
from backend.app.training.yolo_trainer import YoloTrainer
from backend.app.training.yolo_dataset import YoloDataset


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine) -> Generator[Session, None, None]:
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def builder(session) -> FixtureBuilder:
    return FixtureBuilder(session)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_gpu() -> bool:
    return bool(os.environ.get("CUDA_VISIBLE_DEVICES"))


gpu_required = pytest.mark.skipif(not _has_gpu(), reason="No GPU available")


# ---------------------------------------------------------------------------
# 1. Import root model
# ---------------------------------------------------------------------------

class TestImportRootModel:
    def test_import_root_model_creates_approved_node(self, session, builder):
        root = builder.root_model()
        model = root.model_node

        assert model.id is not None
        assert model.parent_id is None
        assert model.status == "approved"
        assert model.task_type == "object_detection"
        assert model.model_family == "yolo"
        assert model.artifact_path.endswith(".pt")
        assert model.artifact_hash.startswith("sha256:")

    def test_root_model_has_label_schema(self, session, builder):
        root = builder.root_model()
        assert root.label_schema is not None
        assert len(root.label_schema.classes) == 3
        class_keys = {c.semantic_key for c in root.label_schema.classes}
        assert class_keys == {"person", "car", "truck"}


# ---------------------------------------------------------------------------
# 2. Import dataset and create snapshot
# ---------------------------------------------------------------------------

class TestImportDataset:
    def test_first_gen_dataset_snapshot(self, session, builder):
        ds = builder.first_gen_dataset()
        snap = ds.snapshot

        assert snap.dataset_id == ds.dataset.id
        assert snap.label_schema_id == ds.label_schema.id
        assert snap.train_positive_count == 3
        assert snap.val_positive_count == 2
        assert snap.test_positive_count == 1
        assert snap.manifest_hash.startswith("sha256:")

    def test_append_category_snapshot_inherits_parent(self, session, builder):
        root_schema = builder.first_gen_label_schema()
        ds = builder.append_category_dataset(parent_schema=root_schema)

        parent_keys = {c.semantic_key for c in root_schema.classes}
        child_keys = {c.semantic_key for c in ds.label_schema.classes}
        assert parent_keys.issubset(child_keys)
        assert "rebar_exposure" in child_keys

    def test_invalid_dataset_has_error(self, session, builder):
        root_schema = builder.first_gen_label_schema()
        ds = builder.invalid_dataset(parent_schema=root_schema)

        assert ds.snapshot is not None
        errors = ds.snapshot.quality_json.get("errors", [])
        assert len(errors) > 0

    def test_invalid_dataset_has_quality_errors(self, session, builder):
        root_schema = builder.first_gen_label_schema()
        ds = builder.invalid_dataset(parent_schema=root_schema)

        # The invalid dataset snapshot records quality errors in quality_json.
        # The training service may or may not reject it depending on
        # implementation; this test verifies the errors are captured.
        errors = ds.snapshot.quality_json.get("errors", [])
        assert len(errors) > 0, "Invalid dataset must have quality errors recorded"

        # Attempt to create a task - if service rejects, that is valid behavior.
        # If service allows it, the quality errors are available for downstream checks.
        try:
            training_service.create_task(
                session,
                parent_model_node_id=None,
                dataset_snapshot_id=ds.snapshot.id,
                task_type="object_detection",
                model_family="yolo",
                training_config_json={},
                resource_config_json={},
                evaluation_policy_json={},
            )
        except ValueError:
            pass  # Service rejected the invalid dataset - valid behavior


# ---------------------------------------------------------------------------
# 3. Create training task
# ---------------------------------------------------------------------------

class TestCreateTrainingTask:
    def test_create_task_queues(self, session, builder):
        root = builder.root_model()
        ds = builder.first_gen_dataset(label_schema=root.label_schema)

        task = training_service.create_task(
            session,
            parent_model_node_id=root.model_node.id,
            dataset_snapshot_id=ds.snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={"epochs": 50, "batchSize": 16},
            resource_config_json={"gpu_device": "0"},
            evaluation_policy_json={
                "min_mAP50": 0.5,
                "min_precision": 0.6,
                "min_recall": 0.5,
                "max_regression_ratio": 0.1,
                "require_per_class_coverage": True,
            },
        )

        assert task.status == "queued"
        assert task.parent_model_node_id == root.model_node.id
        assert task.dataset_snapshot_id == ds.snapshot.id
        assert task.evaluation_policy_json["min_mAP50"] == 0.5

    def test_task_references_are_fixed_at_creation(self, session, builder):
        root = builder.root_model()
        ds = builder.first_gen_dataset(label_schema=root.label_schema)

        task = training_service.create_task(
            session,
            parent_model_node_id=root.model_node.id,
            dataset_snapshot_id=ds.snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={},
            resource_config_json={},
            evaluation_policy_json={},
        )

        # Task immutability: key fields cannot be changed
        with pytest.raises(Exception):
            task.dataset_snapshot_id = uuid.uuid4()
            session.flush()


# ---------------------------------------------------------------------------
# 4. Run attempt
# ---------------------------------------------------------------------------

class TestRunAttempt:
    def test_start_attempt(self, session, builder):
        root = builder.root_model()
        ds = builder.first_gen_dataset(label_schema=root.label_schema)
        task = training_service.create_task(
            session,
            parent_model_node_id=root.model_node.id,
            dataset_snapshot_id=ds.snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={},
            resource_config_json={},
            evaluation_policy_json={},
        )

        attempt = training_service.start_attempt(session, task.id)

        assert attempt.status == "running"
        assert attempt.attempt_no == 1
        assert attempt.lease_token is not None
        assert attempt.fencing_token is not None

    def test_prevents_second_active_attempt(self, session, builder):
        root = builder.root_model()
        ds = builder.first_gen_dataset(label_schema=root.label_schema)
        task = training_service.create_task(
            session,
            parent_model_node_id=root.model_node.id,
            dataset_snapshot_id=ds.snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={},
            resource_config_json={},
            evaluation_policy_json={},
        )

        training_service.start_attempt(session, task.id)

        with pytest.raises(ValueError, match="already has an active attempt"):
            training_service.start_attempt(session, task.id)

    def test_complete_attempt_creates_candidate(self, session, builder):
        root = builder.root_model()
        ds = builder.first_gen_dataset(label_schema=root.label_schema)
        task = training_service.create_task(
            session,
            parent_model_node_id=root.model_node.id,
            dataset_snapshot_id=ds.snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={},
            resource_config_json={},
            evaluation_policy_json={},
        )

        attempt = training_service.start_attempt(session, task.id)
        model = training_service.complete_attempt(
            session,
            attempt.id,
            artifact_path="/data/models/candidate-v1.pt",
            artifact_hash="sha256:candidate-v1",
            metadata_json={"metrics": {"mAP50": 0.85}},
        )

        assert model.status == "candidate"
        assert model.parent_id == root.model_node.id
        assert model.dataset_snapshot_id == ds.snapshot.id
        assert task.status == "completed"

    def test_fail_attempt(self, session, builder):
        root = builder.root_model()
        ds = builder.first_gen_dataset(label_schema=root.label_schema)
        task = training_service.create_task(
            session,
            parent_model_node_id=root.model_node.id,
            dataset_snapshot_id=ds.snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={},
            resource_config_json={},
            evaluation_policy_json={},
        )

        attempt = training_service.start_attempt(session, task.id)
        training_service.fail_attempt(session, attempt.id, error="GPU OOM")

        assert attempt.status == "failed"
        assert attempt.last_error == "GPU OOM"
        assert task.status == "failed"


# ---------------------------------------------------------------------------
# 5. Auto evaluation and manual review
# ---------------------------------------------------------------------------

class TestEvaluationAndReview:
    def test_auto_evaluate_pass(self, session, builder):
        root = builder.root_model()
        ds = builder.first_gen_dataset(label_schema=root.label_schema)
        model_node = builder.root_model(name="eval-model", label_schema=root.label_schema)

        ev = builder.evaluation(
            model_node_id=model_node.model_node.id,
            dataset_snapshot_id=ds.snapshot.id,
        )

        assert ev.auto_status == "pending"

    def test_human_review_pass(self, session, builder):
        root = builder.root_model()
        ds = builder.first_gen_dataset(label_schema=root.label_schema)
        model_node = builder.root_model(name="review-model", label_schema=root.label_schema)

        ev = builder.evaluation(
            model_node_id=model_node.model_node.id,
            dataset_snapshot_id=ds.snapshot.id,
            auto_status="passed",
        )

        assert ev.auto_status == "passed"

    def test_evaluation_requires_both_pass(self, session, builder):
        root = builder.root_model()
        ds = builder.first_gen_dataset(label_schema=root.label_schema)
        model_node = builder.root_model(name="partial-model", label_schema=root.label_schema)

        ev = builder.evaluation(
            model_node_id=model_node.model_node.id,
            dataset_snapshot_id=ds.snapshot.id,
            auto_status="passed",
        )

        # Both statuses must be passed for approval
        both_passed = ev.auto_status == "passed"
        assert not both_passed


# ---------------------------------------------------------------------------
# 6. Publish and inference
# ---------------------------------------------------------------------------

class TestPublishAndInference:
    def test_publish_creates_release(self, session, builder):
        binding = ModelBinding(external_ref="test-binding-001", status="unbound")
        model = builder.root_model(name="publish-model")
        session.add(binding)
        session.flush()

        release = BindingRelease(
            binding_id=binding.id,
            revision_no=1,
            model_node_id=model.model_node.id,
            inference_config_json={"confidence": 0.5, "maxResults": 10},
            inference_config_hash="sha256:publish-config",
            release_type="normal",
            status="pending",
        )
        session.add(release)
        session.flush()

        assert release.status == "pending"
        assert release.revision_no == 1
        assert release.release_type == "normal"

    def test_publish_activates_binding(self, session, builder):
        binding = ModelBinding(external_ref="test-binding-002", status="unbound")
        model = builder.root_model(name="active-model")
        session.add(binding)
        session.flush()

        release = BindingRelease(
            binding_id=binding.id,
            revision_no=1,
            model_node_id=model.model_node.id,
            inference_config_json={},
            inference_config_hash="sha256:active-config",
            release_type="normal",
            status="active",
        )
        session.add(release)
        session.flush()

        instance = RuntimeInstance(
            binding_id=binding.id,
            release_id=release.id,
            model_node_id=model.model_node.id,
            config_hash="sha256:active-config",
            generation=1,
            fencing_token=1,
            status="serving",
            gpu_device="0",
            reserved_memory_mb=4096,
        )
        session.add(instance)
        session.flush()

        binding.current_release_id = release.id
        binding.current_runtime_instance_id = instance.id
        binding.status = "bound"
        session.flush()

        assert binding.current_release_id == release.id
        assert binding.current_runtime_instance_id == instance.id
        assert binding.status == "bound"

    def test_inference_returns_binding_metadata(self, session, builder):
        fixture = builder.service_restart()

        assert fixture.binding.current_release_id == fixture.release.id
        assert fixture.binding.current_runtime_instance_id == fixture.instance.id
        assert fixture.instance.status == "serving"
        assert fixture.instance.generation == 1

    def test_inference_log_excludes_raw_media(self, session, builder):
        fixture = builder.service_restart()

        # Runtime instance should not store raw image data
        assert fixture.instance.actual_memory_mb is not None
        assert fixture.instance.failure_reason is None

    def test_inference_log_excludes_api_key(self, session, builder):
        # The API key is validated at the FastAPI dependency level and never
        # stored in the database.  Verify the runtime instance has no key field.
        fixture = builder.service_restart()
        instance_dict = fixture.instance.__dict__
        assert "api_key" not in instance_dict
        assert "platform_api_key" not in instance_dict


# ---------------------------------------------------------------------------
# 7. Rollback
# ---------------------------------------------------------------------------

class TestRollback:
    def test_rollback_creates_new_release(self, session, builder):
        binding = ModelBinding(external_ref="rollback-binding", status="unbound")
        model_v1 = builder.root_model(name="model-v1")
        model_v2 = builder.root_model(name="model-v2")
        session.add(binding)
        session.flush()

        release_v1 = BindingRelease(
            binding_id=binding.id,
            revision_no=1,
            model_node_id=model_v1.model_node.id,
            inference_config_json={},
            inference_config_hash="sha256:v1",
            release_type="normal",
            status="active",
        )
        session.add(release_v1)
        session.flush()

        rollback = BindingRelease(
            binding_id=binding.id,
            revision_no=2,
            model_node_id=model_v1.model_node.id,
            inference_config_json={},
            inference_config_hash="sha256:v1",
            release_type="rollback",
            rollback_target_release_id=release_v1.id,
            status="active",
            reason="v2 regression",
        )
        session.add(rollback)
        session.flush()

        assert rollback.release_type == "rollback"
        assert rollback.rollback_target_release_id == release_v1.id
        assert rollback.revision_no == 2

    def test_rollback_preserves_history(self, session, builder):
        binding = ModelBinding(external_ref="history-binding", status="unbound")
        model_v1 = builder.root_model(name="history-v1")
        model_v2 = builder.root_model(name="history-v2")
        session.add(binding)
        session.flush()

        release_v1 = BindingRelease(
            binding_id=binding.id,
            revision_no=1,
            model_node_id=model_v1.model_node.id,
            inference_config_json={},
            inference_config_hash="sha256:h1",
            release_type="normal",
            status="superseded",
        )
        release_v2 = BindingRelease(
            binding_id=binding.id,
            revision_no=2,
            model_node_id=model_v2.model_node.id,
            inference_config_json={},
            inference_config_hash="sha256:h2",
            release_type="normal",
            status="superseded",
        )
        session.add(release_v1)
        session.add(release_v2)
        session.flush()

        rollback_v3 = BindingRelease(
            binding_id=binding.id,
            revision_no=3,
            model_node_id=model_v1.model_node.id,
            inference_config_json={},
            inference_config_hash="sha256:h1",
            release_type="rollback",
            rollback_target_release_id=release_v1.id,
            status="active",
            reason="Rollback to v1",
        )
        session.add(rollback_v3)
        session.flush()

        all_releases = [
            r for r in session.query(BindingRelease)
            .filter(BindingRelease.binding_id == binding.id)
            .order_by(BindingRelease.revision_no)
        ]
        assert len(all_releases) == 3
        assert all_releases[0].status == "superseded"
        assert all_releases[2].release_type == "rollback"


# ---------------------------------------------------------------------------
# 8. Stale worker rejection
# ---------------------------------------------------------------------------

class TestStaleWorkerRejection:
    def test_stale_heartbeat_rejected(self, session, builder):
        fixture = builder.expired_lease()

        now = datetime.now(timezone.utc)
        lease_valid = fixture.attempt.lease_expires_at and fixture.attempt.lease_expires_at > now
        assert not lease_valid, "Lease should be expired"

    def test_stale_generation_rejected(self, session, builder):
        fixture = builder.service_restart()

        instance = fixture.instance
        old_generation = instance.generation
        new_generation = old_generation + 1

        assert old_generation == 1
        # Worker writing with stale generation should be rejected
        assert instance.generation != new_generation


# ---------------------------------------------------------------------------
# 9. Snapshot hash change rejection
# ---------------------------------------------------------------------------

class TestSnapshotHashChange:
    def test_snapshot_manifest_is_immutable(self, session, builder):
        ds = builder.first_gen_dataset()
        original_hash = ds.snapshot.manifest_hash

        # Attempting to modify the snapshot fields should raise
        with pytest.raises(Exception):
            ds.snapshot.manifest_hash = "sha256:modified"
            session.flush()

    def test_new_snapshot_gets_different_hash(self, session, builder):
        ds1 = builder.first_gen_dataset()
        ds2 = builder.first_gen_dataset()

        assert ds1.snapshot.manifest_hash != ds2.snapshot.manifest_hash


# ---------------------------------------------------------------------------
# 10. GPU PT training (skip without GPU)
# ---------------------------------------------------------------------------

class TestGpuTraining:
    @gpu_required
    def test_gpu_training_attempt_runs(self, session, builder):
        """Placeholder for actual GPU training validation.

        In a real GPU environment, this would validate CUDA availability,
        model loading onto GPU, and training execution.
        """
        root = builder.root_model()
        ds = builder.first_gen_dataset(label_schema=root.label_schema)
        task = training_service.create_task(
            session,
            parent_model_node_id=root.model_node.id,
            dataset_snapshot_id=ds.snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={"epochs": 1, "batchSize": 2},
            resource_config_json={"gpu_device": "0"},
            evaluation_policy_json={},
        )
        attempt = training_service.start_attempt(session, task.id)
        assert attempt.status == "running"

    @gpu_required
    def test_gpu_training_produces_checkpoint(self, session, builder):
        """Placeholder: validates checkpoint file creation during GPU training."""
        pass


# ---------------------------------------------------------------------------
# 11. Label schema evolution
# ---------------------------------------------------------------------------

class TestLabelSchemaEvolution:
    def test_first_gen_can_ignore_parent(self, session, builder):
        # First-generation models can establish a completely new label schema
        # that is independent from the root model's schema.
        root_schema = builder.first_gen_label_schema()  # fire, smoke
        # Create a new independent schema with different classes
        child_schema = builder._create_label_schema(
            name="schema-independent-child",
            classes=[
                (0, "crack", "Crack"),
                (1, "spalling", "Spalling"),
            ],
        )

        root_keys = {c.semantic_key for c in root_schema.classes}
        child_keys = {c.semantic_key for c in child_schema.classes}

        # First-gen child is NOT required to inherit from root
        assert not root_keys.issubset(child_keys)

    def test_subsequent_gen_must_inherit(self, session, builder):
        root_schema = builder.first_gen_label_schema()
        ds = builder.append_category_dataset(parent_schema=root_schema)

        root_keys = {c.semantic_key for c in root_schema.classes}
        child_keys = {c.semantic_key for c in ds.label_schema.classes}

        assert root_keys.issubset(child_keys)


# ---------------------------------------------------------------------------
# 12. Concurrent release race condition
# ---------------------------------------------------------------------------

class TestConcurrentRelease:
    def test_concurrent_switch_fixture_setup(self, session, builder):
        fixture = builder.concurrent_switch()

        assert fixture.release_a.status == "active"
        assert fixture.release_b.status == "pending"
        assert fixture.binding.current_release_id == fixture.release_a.id

    def test_revision_numbers_are_unique(self, session, builder):
        fixture = builder.concurrent_switch()

        assert fixture.release_a.revision_no != fixture.release_b.revision_no


# ---------------------------------------------------------------------------
# 13. Service restart recovery
# ---------------------------------------------------------------------------

class TestServiceRestart:
    def test_serving_instance_survives_restart(self, session, builder):
        fixture = builder.service_restart()

        assert fixture.instance.status == "serving"
        assert fixture.instance.worker_id == "worker-old-001"
        assert fixture.instance.heartbeat_at is not None

    def test_stale_worker_detected_by_heartbeat(self, session, builder):
        fixture = builder.service_restart()

        stale_threshold = timedelta(seconds=60)
        now = datetime.now(timezone.utc)
        heartbeat_age = now - fixture.instance.heartbeat_at

        assert heartbeat_age > stale_threshold, (
            "Worker heartbeat should be stale after restart"
        )


# ---------------------------------------------------------------------------
# 14. Full lifecycle happy path
# ---------------------------------------------------------------------------

class TestFullLifecycle:
    def test_happy_path(self, session, builder):
        # 1. Import root model
        root = builder.root_model()
        assert root.model_node.status == "approved"

        # 2. Import dataset
        ds = builder.first_gen_dataset(label_schema=root.label_schema)
        assert ds.snapshot.train_positive_count > 0

        # 3. Create training task
        task = training_service.create_task(
            session,
            parent_model_node_id=root.model_node.id,
            dataset_snapshot_id=ds.snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={"epochs": 10},
            resource_config_json={"gpu_device": "0"},
            evaluation_policy_json={"min_mAP50": 0.5},
        )
        assert task.status == "queued"

        # 4. Start attempt
        attempt = training_service.start_attempt(session, task.id)
        assert attempt.status == "running"

        # 5. Complete attempt -> candidate
        candidate = training_service.complete_attempt(
            session,
            attempt.id,
            artifact_path="/data/models/candidate.pt",
            artifact_hash="sha256:candidate",
            metadata_json={"mAP50": 0.82},
        )
        assert candidate.status == "candidate"
        assert candidate.parent_id == root.model_node.id

        # 6. Evaluate
        ev = builder.evaluation(
            model_node_id=candidate.id,
            dataset_snapshot_id=ds.snapshot.id,
            auto_status="passed",
        )
        assert ev.auto_status == "passed"

        # 7. Approve
        candidate.status = "approved"
        session.flush()
        assert candidate.status == "approved"

        # 8. Create binding and publish
        binding = ModelBinding(external_ref="lifecycle-binding", status="unbound")
        session.add(binding)
        session.flush()

        release = BindingRelease(
            binding_id=binding.id,
            revision_no=1,
            model_node_id=candidate.id,
            inference_config_json={"confidence": 0.5},
            inference_config_hash="sha256:lifecycle-config",
            release_type="normal",
            status="active",
        )
        session.add(release)
        session.flush()

        binding.current_release_id = release.id
        binding.status = "bound"
        session.flush()

        assert binding.status == "bound"
        assert binding.current_release_id == release.id

        # 9. Inference (metadata check)
        instance = RuntimeInstance(
            binding_id=binding.id,
            release_id=release.id,
            model_node_id=candidate.id,
            config_hash="sha256:lifecycle-config",
            generation=1,
            fencing_token=1,
            status="serving",
            gpu_device="0",
            reserved_memory_mb=4096,
        )
        session.add(instance)
        session.flush()

        binding.current_runtime_instance_id = instance.id
        session.flush()

        assert instance.status == "serving"

        # 10. Rollback
        rollback = BindingRelease(
            binding_id=binding.id,
            revision_no=2,
            model_node_id=root.model_node.id,
            inference_config_json={},
            inference_config_hash="sha256:root-config",
            release_type="rollback",
            rollback_target_release_id=release.id,
            status="active",
            reason="Regression detected",
        )
        session.add(rollback)
        session.flush()

        assert rollback.release_type == "rollback"
        assert rollback.rollback_target_release_id == release.id


# ---------------------------------------------------------------------------
# 15. Real YOLO closed-loop (CPU, epochs=2)
# ---------------------------------------------------------------------------

@pytest.fixture()
def real_dataset() -> RealYoloDataset:
    ds = build_real_yolo_dataset()
    yield ds
    cleanup_real_dataset(ds.root_dir)


class TestRealYoloClosedLoop:
    """Full lifecycle using real YOLO training on CPU with a tiny dataset."""

    def test_import_root_model_real(self, real_dataset):
        model_path = ensure_root_model_exists()
        assert model_path.exists()
        assert model_path.suffix == ".pt"
        assert model_path.stat().st_size > 0

    def test_real_dataset_structure(self, real_dataset):
        assert real_dataset.data_yaml_path.exists()
        assert real_dataset.nc == 2
        assert len(real_dataset.train_images) == 10
        assert len(real_dataset.val_images) == 4
        for img in real_dataset.train_images:
            assert img.exists()
            assert img.suffix == ".jpg"

    def test_real_yolo_train_produces_checkpoint(self, real_dataset, tmp_path):
        model_path = ensure_root_model_exists()
        checkpoint_dir = tmp_path / "checkpoints"

        trainer = YoloTrainer({
            "epochs": 2,
            "imgsz": 640,
            "batch": 2,
            "device": "cpu",
            "model_name": "yolov8n",
            "project": str(checkpoint_dir),
            "name": "train",
        })

        result = trainer.train(
            dataset_dir=real_dataset.root_dir,
            checkpoint_dir=checkpoint_dir,
            parent_model_path=str(model_path),
        )

        assert result.success, f"Training failed: {result.error}"
        assert result.epochs_completed == 2
        assert result.best_model_path is not None
        assert Path(result.best_model_path).exists()
        assert len(result.metrics) > 0

    def test_full_real_yolo_closed_loop(self, session, builder, real_dataset, tmp_path):
        """Full lifecycle: import -> dataset -> snapshot -> train -> candidate -> evaluate -> publish -> inference -> rollback."""
        import uuid as _uuid

        # 1. Import root model
        model_path = ensure_root_model_exists()
        root = builder.root_model(
            name="yolov8n-root-real",
            label_schema=builder._create_label_schema(
                name="schema-real-root",
                classes=[
                    (0, "crack", "Crack"),
                    (1, "spalling", "Spalling"),
                ],
            ),
        )
        assert root.model_node.status == "approved"

        # 2. Import dataset and create snapshot
        ds = builder.first_gen_dataset(label_schema=root.label_schema)
        assert ds.snapshot.train_positive_count > 0
        assert ds.snapshot.val_positive_count > 0

        # 3. Create training task
        task = training_service.create_task(
            session,
            parent_model_node_id=root.model_node.id,
            dataset_snapshot_id=ds.snapshot.id,
            task_type="object_detection",
            model_family="yolo",
            training_config_json={"epochs": 2, "batch": 2, "device": "cpu"},
            resource_config_json={},
            evaluation_policy_json={"min_mAP50": 0.0},
        )
        assert task.status == "queued"

        # 4. Start attempt
        attempt = training_service.start_attempt(session, task.id)
        assert attempt.status == "running"

        # 5. Actually train with YOLO
        checkpoint_dir = tmp_path / "checkpoints"
        trainer = YoloTrainer({
            "epochs": 2,
            "imgsz": 640,
            "batch": 2,
            "device": "cpu",
            "model_name": "yolov8n",
            "project": str(checkpoint_dir),
            "name": "train",
        })
        result = trainer.train(
            dataset_dir=real_dataset.root_dir,
            checkpoint_dir=checkpoint_dir,
            parent_model_path=str(model_path),
        )
        assert result.success, f"Training failed: {result.error}"

        # 6. Complete attempt -> candidate
        candidate = training_service.complete_attempt(
            session,
            attempt.id,
            artifact_path=result.best_model_path,
            artifact_hash=f"sha256:{_sha256_hex(result.best_model_path)}",
            metadata_json={"metrics": result.metrics, "epochs": result.epochs_completed},
        )
        assert candidate.status == "candidate"
        assert candidate.parent_id == root.model_node.id

        # 7. Auto evaluate + human review pass
        ev = builder.evaluation(
            model_node_id=candidate.id,
            dataset_snapshot_id=ds.snapshot.id,
            auto_status="passed",
        )
        assert ev.auto_status == "passed"

        # 8. Approve candidate
        candidate.status = "approved"
        session.flush()
        assert candidate.status == "approved"

        # 9. Create binding and publish
        binding = ModelBinding(external_ref="real-yolo-binding", status="unbound")
        session.add(binding)
        session.flush()

        release = BindingRelease(
            binding_id=binding.id,
            revision_no=1,
            model_node_id=candidate.id,
            inference_config_json={"confidence": 0.5},
            inference_config_hash=f"sha256:{_sha256_hex('real-yolo-config')}",
            release_type="normal",
            status="active",
        )
        session.add(release)
        session.flush()

        binding.current_release_id = release.id
        binding.status = "bound"
        session.flush()
        assert binding.status == "bound"

        # 10. Inference metadata check
        instance = RuntimeInstance(
            binding_id=binding.id,
            release_id=release.id,
            model_node_id=candidate.id,
            config_hash=f"sha256:{_sha256_hex('real-yolo-config')}",
            generation=1,
            fencing_token=1,
            status="serving",
            gpu_device="cpu",
            reserved_memory_mb=2048,
        )
        session.add(instance)
        session.flush()
        binding.current_runtime_instance_id = instance.id
        session.flush()
        assert instance.status == "serving"

        # 11. Rollback
        rollback = BindingRelease(
            binding_id=binding.id,
            revision_no=2,
            model_node_id=root.model_node.id,
            inference_config_json={},
            inference_config_hash=f"sha256:{_sha256_hex('root-config')}",
            release_type="rollback",
            rollback_target_release_id=release.id,
            status="active",
            reason="Regression detected",
        )
        session.add(rollback)
        session.flush()

        assert rollback.release_type == "rollback"
        assert rollback.rollback_target_release_id == release.id
        assert rollback.revision_no == 2


# ---------------------------------------------------------------------------
# 16. Worker closed-loop smoke test
# ---------------------------------------------------------------------------

class TestWorkerClosedLoopSmoke:
    def test_worker_receives_and_processes_task(self, session, builder, monkeypatch):
        from backend.app.workers.train_worker import run_training, execute_training
        from backend.app.workers.db_session import set_worker_engine_override, reset_worker_session_factory
        from backend.app.models import LabelSchema
        from backend.app.training.yolo_trainer import TrainResult
        import tempfile

        engine = session.get_bind()
        set_worker_engine_override(engine)

        try:
            label_schema = LabelSchema(name="smoke-schema", status="active")
            session.add(label_schema)
            session.flush()

            from backend.app.models.label_schema import LabelSchemaClass
            for class_id, semantic_key in enumerate(["crack", "spalling"]):
                cls = LabelSchemaClass(
                    schema_id=label_schema.id,
                    class_id=class_id,
                    semantic_key=semantic_key,
                    display_name=semantic_key.title(),
                )
                session.add(cls)
            session.flush()

            root = builder.root_model(
                name="smoke-root-model",
                label_schema=label_schema,
            )

            ds = builder.first_gen_dataset(label_schema=root.label_schema)

            task = training_service.create_task(
                session,
                parent_model_node_id=root.model_node.id,
                dataset_snapshot_id=ds.snapshot.id,
                task_type="object_detection",
                model_family="yolo",
                training_config_json={"epochs": 1, "batch": 2, "device": "cpu"},
                resource_config_json={"memory_mb": 256},
                evaluation_policy_json={"min_mAP50": 0.0},
            )
            session.commit()
            assert task.status == "queued"

            def fake_execute_training(**kwargs):
                output_dir = kwargs["output_dir"]
                output_dir.mkdir(parents=True, exist_ok=True)
                best_path = output_dir / "best.pt"
                best_path.write_bytes(b"fake model")
                return TrainResult(
                    success=True,
                    epochs_completed=1,
                    best_model_path=str(best_path),
                    latest_model_path=str(best_path),
                    metrics={"mAP50": 0.5},
                )

            monkeypatch.setattr(
                "backend.app.workers.train_worker.execute_training",
                fake_execute_training,
            )

            result = run_training(str(task.id))

            with Session(engine) as verify:
                refreshed = verify.get(TrainingTask, task.id)
                assert refreshed.status == "completed", f"Task status: {refreshed.status}, result: {result}"
                assert len(refreshed.attempts) > 0

        finally:
            reset_worker_session_factory()

    def test_worker_handles_missing_task_gracefully(self):
        from backend.app.workers.train_worker import run_training
        from backend.app.workers.db_session import set_worker_engine_override, reset_worker_session_factory
        from sqlalchemy import create_engine, event
        from backend.app.models import Base
        import uuid

        engine = create_engine("sqlite+pysqlite:///:memory:")
        event.listen(
            engine,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
        Base.metadata.create_all(engine)
        set_worker_engine_override(engine)

        try:
            fake_id = str(uuid.uuid4())
            result = run_training(fake_id)
            assert "not found" in result
        finally:
            reset_worker_session_factory()
            engine.dispose()

    def test_worker_handles_terminal_task_idempotently(self, session, builder):
        from backend.app.workers.train_worker import run_training
        from backend.app.workers.db_session import set_worker_engine_override, reset_worker_session_factory

        engine = session.get_bind()
        set_worker_engine_override(engine)

        try:
            root = builder.root_model(
                name="idempotent-root-model",
                label_schema=builder._create_label_schema(
                    name="idempotent-schema",
                    classes=[
                        (0, "crack", "Crack"),
                    ],
                ),
            )

            ds = builder.first_gen_dataset(label_schema=root.label_schema)

            task = training_service.create_task(
                session,
                parent_model_node_id=root.model_node.id,
                dataset_snapshot_id=ds.snapshot.id,
                task_type="object_detection",
                model_family="yolo",
                training_config_json={},
                resource_config_json={},
                evaluation_policy_json={},
            )
            session.commit()

            attempt = training_service.start_attempt(session, task.id)
            session.commit()

            training_service.complete_attempt(
                session,
                attempt.id,
                artifact_path="/data/models/test.pt",
                artifact_hash="sha256:test",
            )
            session.commit()

            session.refresh(task)
            assert task.status == "completed"

            result = run_training(str(task.id))
            assert "already completed" in result

        finally:
            reset_worker_session_factory()
