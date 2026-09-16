from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import DateTime, create_engine, event, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models import (
    Base,
    BindingRelease,
    Checkpoint,
    Dataset,
    DatasetSnapshot,
    Evaluation,
    GPUResource,
    GPUResourceLease,
    LabelSchema,
    LabelSchemaClass,
    ModelBinding,
    ModelNode,
    ModelResidencyPlan,
    OperationLog,
    RuntimeInstance,
    TrainingAttempt,
    TrainingTask,
)


@pytest.fixture()
def session() -> Session:
    database_url = os.environ.get("TEST_DATABASE_URL", "sqlite+pysqlite:///:memory:")
    engine = create_engine(database_url)
    if engine.dialect.name == "sqlite":
        event.listen(
            engine,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
    Base.metadata.create_all(engine)
    with Session(engine) as db_session:
        yield db_session
    if engine.dialect.name == "sqlite":
        engine.dispose()
    else:
        Base.metadata.drop_all(engine)
        engine.dispose()


def make_schema(session: Session, name: str = "detector") -> LabelSchema:
    schema = LabelSchema(name=name)
    session.add(schema)
    session.flush()
    return schema


def make_model(
    session: Session,
    *,
    parent: ModelNode | None = None,
    training_attempt_id: object | None = None,
) -> ModelNode:
    model = ModelNode(
        parent=parent,
        artifact_path=f"models/{uuid4()}.pt",
        artifact_hash=f"sha256:{uuid4().hex}",
        task_type="object_detection",
        model_family="yolo",
        training_attempt_id=training_attempt_id,
        status="approved",
        metadata_json={"format": "pt"},
    )
    session.add(model)
    session.flush()
    return model


def make_binding_release(
    session: Session,
    binding: ModelBinding,
    model: ModelNode,
    revision: int,
) -> BindingRelease:
    release = BindingRelease(
        binding=binding,
        revision_no=revision,
        model_node=model,
        inference_config_json={"confidence": 0.5},
        inference_config_hash=f"sha256:{uuid4().hex}",
        release_type="normal",
        status="pending",
    )
    session.add(release)
    session.flush()
    return release


def test_database_models_can_be_created(session: Session) -> None:
    assert session.execute(select(LabelSchema)).scalars().all() == []


def test_root_and_parent_child_model_nodes(session: Session) -> None:
    root = make_model(session)
    child = make_model(session, parent=root)
    session.commit()

    assert child.parent_id == root.id
    assert child.parent.id == root.id
    assert root.parent_id is None
    assert root.label_schema_id is None


def test_label_schema_lineage_and_class_uniqueness(session: Session) -> None:
    parent = make_schema(session, "base")
    parent.classes.append(
        LabelSchemaClass(class_id=0, semantic_key="person", display_name="Person")
    )
    child = LabelSchema(name="extended", parent_schema=parent)
    child.classes.append(
        LabelSchemaClass(class_id=0, semantic_key="person", display_name="Person")
    )
    child.classes.append(
        LabelSchemaClass(class_id=1, semantic_key="vehicle", display_name="Vehicle")
    )
    session.add(child)
    session.commit()

    assert child.parent_schema_id == parent.id
    assert {item.semantic_key for item in child.classes} == {"person", "vehicle"}

    duplicate = LabelSchemaClass(
        schema_id=child.id,
        class_id=1,
        semantic_key="other-vehicle",
        display_name="Other vehicle",
    )
    session.add(duplicate)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

def test_dataset_snapshot_references_logical_dataset_and_schema(
    session: Session,
) -> None:
    schema = make_schema(session)
    dataset = Dataset(name="training-images")
    snapshot = DatasetSnapshot(
        dataset=dataset,
        label_schema=schema,
        train_manifest_json=[{"path": "train/a.jpg"}],
        val_manifest_json=[{"path": "val/a.jpg"}],
        test_manifest_json=[{"path": "test/a.jpg"}],
        manifest_path="snapshots/s1/manifest.json",
        manifest_hash="sha256:manifest",
        train_positive_count=1,
        val_positive_count=1,
        test_positive_count=1,
        train_negative_count=0,
        val_negative_count=0,
        test_negative_count=0,
    )
    session.add(snapshot)
    session.commit()

    assert snapshot.dataset_id == dataset.id
    assert snapshot.label_schema_id == schema.id
    assert snapshot.dataset.id == dataset.id


def test_one_running_or_recovering_attempt_per_training_task(session: Session) -> None:
    dataset = Dataset(name="training-images")
    snapshot = DatasetSnapshot(
        dataset=dataset,
        label_schema=make_schema(session, "training-schema"),
        manifest_path="snapshots/s1/manifest.json",
        manifest_hash="sha256:manifest",
    )
    task = TrainingTask(
        dataset_snapshot=snapshot,
        task_type="object_detection",
        model_family="yolo",
        training_config_json={"epochs": 1},
        evaluation_policy_json={"mAP50": 0.5},
        status="queued",
    )
    session.add(task)
    session.commit()
    session.add_all(
        [
            TrainingAttempt(
                task=task,
                attempt_no=1,
                status="running",
                lease_token=uuid4(),
                fencing_token=1,
            ),
            TrainingAttempt(
                task=task,
                attempt_no=2,
                status="running",
                lease_token=uuid4(),
                fencing_token=2,
            ),
        ]
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.expire_all()
    task = session.get(TrainingTask, task.id)
    assert task is not None

    first = TrainingAttempt(
        task=task,
        attempt_no=1,
        status="recovering",
        lease_token=uuid4(),
        fencing_token=1,
    )
    second = TrainingAttempt(
        task=task,
        attempt_no=2,
        status="recovering",
        lease_token=uuid4(),
        fencing_token=2,
    )
    session.add_all([first, second])
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_dataset_snapshot_requires_a_label_schema(session: Session) -> None:
    snapshot = DatasetSnapshot(
        dataset=Dataset(name="without-schema"),
        manifest_path="snapshots/missing-schema/manifest.json",
        manifest_hash="sha256:manifest",
    )
    session.add(snapshot)
    with pytest.raises(IntegrityError):
        session.commit()


def test_training_output_belongs_to_a_specific_attempt(session: Session) -> None:
    assert "output_model_node_id" not in TrainingTask.__table__.c
    assert "output_model_node_id" not in TrainingAttempt.__table__.c
    assert "training_attempt_id" in ModelNode.__table__.c
    assert hasattr(TrainingAttempt, "output_model_node")


def test_release_model_identity_is_required_by_runtime_and_residency(
    session: Session,
) -> None:
    release_model = make_model(session)
    other_model = make_model(session)
    binding = ModelBinding(status="unbound")
    session.add(binding)
    session.flush()
    release = make_binding_release(session, binding, release_model, 1)
    session.commit()

    session.add(
        RuntimeInstance(
            binding_id=binding.id,
            release_id=release.id,
            model_node=other_model,
            config_hash="mismatch",
            generation=1,
            fencing_token=1,
            status="serving",
            gpu_device="0",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(
        ModelResidencyPlan(
            binding_id=binding.id,
            release_id=release.id,
            model_node=other_model,
            gpu_device="0",
            reserved_memory_mb=128,
            desired_state="resident",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    valid_plan = ModelResidencyPlan(
        binding_id=binding.id,
        release_id=release.id,
        model_node_id=release_model.id,
        gpu_device="0",
        reserved_memory_mb=128,
        desired_state="resident",
    )
    session.add(valid_plan)
    session.commit()
    assert valid_plan.binding.id == binding.id
    assert valid_plan.release.id == release.id
    assert valid_plan.model_node.id == release_model.id


def make_snapshot(session: Session, name: str = "sealed-dataset") -> DatasetSnapshot:
    schema = make_schema(session, f"{name}-schema")
    snapshot = DatasetSnapshot(
        dataset=Dataset(name=name),
        label_schema=schema,
        train_manifest_json=[{"path": "train/a.jpg"}],
        val_manifest_json=[{"path": "val/a.jpg"}],
        test_manifest_json=[{"path": "test/a.jpg"}],
        manifest_path=f"snapshots/{name}/manifest.json",
        manifest_hash="sha256:manifest",
        train_positive_count=1,
        val_positive_count=2,
        test_positive_count=3,
        train_negative_count=4,
        val_negative_count=5,
        test_negative_count=6,
        quality_json={"warning": False},
        source_path=f"/imports/{name}",
    )
    session.add(snapshot)
    session.commit()
    return snapshot


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_manifest_json", [{"path": "changed.jpg"}]),
        ("val_manifest_json", [{"path": "changed.jpg"}]),
        ("test_manifest_json", [{"path": "changed.jpg"}]),
        ("label_schema_id", uuid4()),
        ("manifest_path", "snapshots/changed.json"),
        ("manifest_hash", "sha256:changed"),
        ("train_positive_count", 9),
        ("val_negative_count", 9),
        ("quality_json", {"warning": True}),
        ("source_path", "/imports/changed"),
    ],
)
def test_dataset_snapshot_sealed_fields_cannot_change(
    session: Session, field: str, value: object
) -> None:
    snapshot = make_snapshot(session, f"sealed-{field}")
    setattr(snapshot, field, value)
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parent_id", uuid4()),
        ("label_schema_id", uuid4()),
        ("dataset_snapshot_id", uuid4()),
        ("training_attempt_id", uuid4()),
        ("artifact_path", "models/changed.pt"),
        ("artifact_hash", "sha256:changed"),
        ("framework", "changed-framework"),
        ("artifact_format", "onnx"),
        ("metadata_json", {"changed": True}),
        ("task_type", "classification"),
        ("model_family", "other-family"),
    ],
)
def test_model_node_sealed_fields_cannot_change(
    session: Session, field: str, value: object
) -> None:
    model = make_model(session)
    session.commit()
    setattr(model, field, value)
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()


def test_evaluation_attempt_status_and_release_revision_constraints(
    session: Session,
) -> None:
    schema = make_schema(session, "evaluation-schema")
    snapshot = DatasetSnapshot(
        dataset=Dataset(name="evaluation-dataset"),
        label_schema=schema,
        manifest_path="snapshots/evaluation/manifest.json",
        manifest_hash="sha256:evaluation",
    )
    model = make_model(session)
    session.add(snapshot)
    session.flush()
    session.add(
        Evaluation(
            model_node=model,
            dataset_snapshot=snapshot,
            attempt_status="invalid",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    binding = ModelBinding(status="unbound")
    session.add(binding)
    session.flush()
    session.add(
        BindingRelease(
            binding=binding,
            revision_no=0,
            model_node=model,
            inference_config_json={},
            inference_config_hash="sha256:config",
            release_type="normal",
            status="pending",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_rollback_target_must_belong_to_the_same_binding(session: Session) -> None:
    model = make_model(session)
    first_binding = ModelBinding(status="unbound")
    second_binding = ModelBinding(status="unbound")
    session.add_all([first_binding, second_binding])
    session.flush()
    target = make_binding_release(session, first_binding, model, 1)
    valid_rollback = BindingRelease(
        binding=first_binding,
        revision_no=2,
        model_node=model,
        inference_config_json={},
        inference_config_hash="sha256:rollback",
        release_type="rollback",
        rollback_target_release_id=target.id,
        status="pending",
    )
    session.add(valid_rollback)
    session.commit()
    assert valid_rollback.rollback_target_release_id == target.id
    assert valid_rollback.rollback_target.id == target.id

    invalid_rollback = BindingRelease(
        binding=second_binding,
        revision_no=1,
        model_node=model,
        inference_config_json={},
        inference_config_hash="sha256:invalid-rollback",
        release_type="rollback",
        rollback_target_release_id=target.id,
        status="pending",
    )
    session.add(invalid_rollback)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_gpu_lease_owner_is_unique_while_active(session: Session) -> None:
    owner_id = uuid4()
    expires = datetime.now(timezone.utc) + timedelta(minutes=5)
    resource = GPUResource(
        gpu_device="0", capacity_memory_mb=1024, reserved_memory_mb=0
    )
    session.add(resource)
    session.commit()
    session.add(
        GPUResourceLease(
            resource=resource,
            gpu_device="0",
            owner_type="training_attempt",
            owner_id=owner_id,
            reserved_memory_mb=128,
            lease_token=uuid4(),
            fencing_token=1,
            lease_expires_at=expires,
            status="active",
        )
    )
    session.commit()
    session.add(
        GPUResourceLease(
            resource=resource,
            gpu_device="0",
            owner_type="training_attempt",
            owner_id=owner_id,
            reserved_memory_mb=256,
            lease_token=uuid4(),
            fencing_token=2,
            lease_expires_at=expires,
            status="active",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_lease_and_lifecycle_timestamps_are_timezone_aware() -> None:
    timestamp_names = {
        "created_at",
        "updated_at",
        "heartbeat_at",
        "lease_expires_at",
        "started_at",
        "finished_at",
        "stopped_at",
    }
    for table in Base.metadata.tables.values():
        for column in table.columns:
            if column.name in timestamp_names:
                assert isinstance(column.type, DateTime)
                assert column.type.timezone is True, f"{table.name}.{column.name}"


def test_label_schema_definition_fields_are_immutable_when_sealed(
    session: Session,
) -> None:
    schema = make_schema(session, "sealed-label-schema")
    label_class = LabelSchemaClass(
        schema=schema,
        class_id=1,
        semantic_key="vehicle",
        display_name="Vehicle",
        description="A vehicle",
    )
    session.add(label_class)
    session.commit()

    schema.definition_sealed = True
    session.commit()

    schema.name = "changed-name"
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()

    label_class.class_id = 2
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()

    label_class.semantic_key = "different-key"
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()

    label_class.display_name = "Changed vehicle"
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()

    label_class.description = "Changed description"
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()

    other_schema = make_schema(session, "other-sealed-schema")
    label_class.schema_id = other_schema.id
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()

    session.delete(label_class)
    with pytest.raises(ValueError, match="sealed"):
        session.commit()
    session.rollback()


def test_referenced_label_schema_cannot_be_modified_or_have_classes_deleted(
    session: Session,
) -> None:
    schema = make_schema(session, "referenced-schema")
    dataset = Dataset(name="referenced-dataset")
    snapshot = DatasetSnapshot(
        dataset=dataset,
        label_schema=schema,
        manifest_path="snapshots/ref/manifest.json",
        manifest_hash="sha256:ref",
    )
    label_class = LabelSchemaClass(
        schema=schema,
        class_id=0,
        semantic_key="person",
        display_name="Person",
    )
    session.add_all([snapshot, label_class])
    session.commit()

    schema.name = "changed-name"
    with pytest.raises(ValueError, match="referenced"):
        session.commit()
    session.rollback()

    label_class.display_name = "Changed Person"
    with pytest.raises(ValueError, match="referenced"):
        session.commit()
    session.rollback()

    session.delete(label_class)
    with pytest.raises(ValueError, match="referenced"):
        session.commit()
    session.rollback()


def test_sealed_label_schema_cannot_be_unsealed_and_changed_in_one_flush(
    session: Session,
) -> None:
    schema = make_schema(session, "one-way-seal")
    session.commit()
    schema.definition_sealed = True
    session.commit()

    schema.definition_sealed = False
    schema.name = "changed-after-unseal"
    with pytest.raises(ValueError, match="can only be enabled once"):
        session.flush()
    session.rollback()


def test_policy_and_report_fields_are_immutable_after_creation(session: Session) -> None:
    snapshot = make_snapshot(session, "policy-freeze")
    task = TrainingTask(
        dataset_snapshot=snapshot,
        task_type="object_detection",
        model_family="yolo",
        training_config_json={"epochs": 1},
        evaluation_policy_json={"mAP50": 0.5},
        status="queued",
    )
    session.add(task)
    session.commit()

    task.evaluation_policy_json["thresholds"] = {"mAP50": 0.5}
    task.evaluation_policy_json["thresholds"]["mAP50"] = 0.9
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()


    model = make_model(session)
    evaluation = Evaluation(
        model_node=model,
        dataset_snapshot=snapshot,
        evaluation_policy_json={"mAP50": 0.5},
        report_path="reports/evaluation.json",
        report_hash="sha256:report",
    )
    session.add(evaluation)
    session.commit()
    assert evaluation.model_node.id == model.id
    assert evaluation.dataset_snapshot.id == snapshot.id

    evaluation.evaluation_policy_json["mAP50"] = 0.9
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()

    evaluation.report_path = "reports/changed.json"
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()

    evaluation.report_hash = "sha256:changed"
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()


def test_evaluation_metrics_can_be_written_once_after_pending_creation(
    session: Session,
) -> None:
    snapshot = make_snapshot(session, "metrics-once")
    model = make_model(session)
    evaluation = Evaluation(
        model_node=model,
        dataset_snapshot=snapshot,
        evaluation_policy_json={"original": True},
    )
    session.add(evaluation)
    session.commit()

    evaluation.auto_metrics_json["per_class"] = {"vehicle": {"mAP50": 0.8}}
    session.commit()
    assert evaluation.auto_metrics_json["per_class"]["vehicle"]["mAP50"] == 0.8

    evaluation.auto_metrics_json["per_class"]["vehicle"]["mAP50"] = 0.9
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()


def test_non_pending_empty_evaluation_metrics_cannot_be_filled(
    session: Session,
) -> None:
    snapshot = make_snapshot(session, "metrics-non-pending")
    model = make_model(session)
    evaluation = Evaluation(
        model_node=model,
        dataset_snapshot=snapshot,
        auto_status="passed",
        auto_metrics_json={},
    )
    session.add(evaluation)
    session.commit()

    evaluation.auto_metrics_json["mAP50"] = 0.8
    with pytest.raises(ValueError, match="pending"):
        session.commit()
    session.rollback()


def test_sealed_label_schema_rejects_new_classes(session: Session) -> None:
    schema = make_schema(session, "sealed-no-append")
    session.commit()
    schema.definition_sealed = True
    session.commit()

    schema.classes.append(
        LabelSchemaClass(class_id=1, semantic_key="vehicle", display_name="Vehicle")
    )
    with pytest.raises(ValueError, match="sealed"):
        session.commit()
    session.rollback()


def test_binding_release_definition_fields_are_immutable(session: Session) -> None:
    model = make_model(session)
    replacement_model = make_model(session)
    first_binding = ModelBinding(status="unbound")
    second_binding = ModelBinding(status="unbound")
    session.add_all([first_binding, second_binding])
    session.flush()
    release = make_binding_release(session, first_binding, model, 1)
    session.commit()

    mutations = [
        ("binding_id", second_binding.id),
        ("model_node_id", replacement_model.id),
        ("revision_no", 2),
        ("rollback_target_release_id", uuid4()),
        ("inference_config_hash", "sha256:changed"),
        ("release_type", "rollback"),
        ("reason", "changed reason"),
    ]
    for field, value in mutations:
        setattr(release, field, value)
        with pytest.raises(ValueError, match="immutable"):
            session.commit()
        session.rollback()

    release.inference_config_json["confidence"] = 0.9
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()


def test_service_contract_current_pointer_activation_gate() -> None:
    assert "fk_binding_current_release" in {
        constraint.name for constraint in ModelBinding.__table__.constraints
    }
    assert "fk_binding_current_runtime" in {
        constraint.name for constraint in ModelBinding.__table__.constraints
    }
    assert "active" in str(BindingRelease.__table__.c.status.type) or any(
        "active" in str(constraint.sqltext)
        for constraint in BindingRelease.__table__.constraints
        if hasattr(constraint, "sqltext")
    )
    assert any(
        "serving" in str(constraint.sqltext)
        for constraint in RuntimeInstance.__table__.constraints
        if hasattr(constraint, "sqltext")
    )


def test_service_contract_model_and_checkpoint_cross_table_gates() -> None:
    assert ModelNode.__table__.c.dataset_snapshot_id.foreign_keys
    assert ModelNode.__table__.c.label_schema_id.foreign_keys
    assert Checkpoint.__table__.c.dataset_snapshot_id.foreign_keys
    assert TrainingTask.__table__.c.dataset_snapshot_id.foreign_keys


def test_gpu_capacity_is_enforced_and_released(session: Session) -> None:
    resource = GPUResource(
        gpu_device="0", capacity_memory_mb=100, reserved_memory_mb=0
    )
    session.add(resource)
    session.commit()

    first = GPUResourceLease(
        resource=resource,
        gpu_device="0",
        owner_type="attempt",
        owner_id=uuid4(),
        reserved_memory_mb=80,
        lease_token=uuid4(),
        fencing_token=1,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        status="active",
    )
    session.add(first)
    session.commit()
    assert resource.reserved_memory_mb == 80

    second = GPUResourceLease(
        resource=resource,
        gpu_device="0",
        owner_type="attempt",
        owner_id=uuid4(),
        reserved_memory_mb=30,
        lease_token=uuid4(),
        fencing_token=2,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        status="active",
    )
    session.add(second)
    with pytest.raises(ValueError, match="capacity"):
        session.commit()
    session.rollback()

    first.status = "released"
    session.commit()
    session.add(second)
    session.commit()
    assert resource.reserved_memory_mb == 30

    resource.capacity_memory_mb = 20
    with pytest.raises(ValueError, match="capacity"):
        session.commit()
    session.rollback()

    other_resource = GPUResource(
        gpu_device="1", capacity_memory_mb=100, reserved_memory_mb=0
    )
    session.add(other_resource)
    session.commit()
    first.gpu_resource_id = other_resource.id
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()

    first.gpu_device = "1"
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()


def test_gpu_resource_and_lease_memory_values_cannot_be_negative(
    session: Session,
) -> None:
    negative_resource = GPUResource(
        gpu_device="negative", capacity_memory_mb=-1, reserved_memory_mb=0
    )
    session.add(negative_resource)
    with pytest.raises((IntegrityError, ValueError)):
        session.commit()
    session.rollback()


def test_gpu_new_resource_and_lease_are_capacity_checked_in_one_flush(
    session: Session,
) -> None:
    resource = GPUResource(
        gpu_device="same-flush", capacity_memory_mb=10, reserved_memory_mb=0
    )
    lease = GPUResourceLease(
        resource=resource,
        gpu_device="same-flush",
        owner_type="attempt",
        owner_id=uuid4(),
        reserved_memory_mb=11,
        lease_token=uuid4(),
        fencing_token=1,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        status="active",
    )
    session.add_all([resource, lease])
    with pytest.raises(ValueError, match="capacity"):
        session.flush()
    session.rollback()

    resource = GPUResource(
        gpu_device="nonnegative", capacity_memory_mb=100, reserved_memory_mb=0
    )
    session.add(resource)
    session.commit()
    negative_lease = GPUResourceLease(
        resource=resource,
        gpu_device="nonnegative",
        owner_type="attempt",
        owner_id=uuid4(),
        reserved_memory_mb=-1,
        actual_memory_mb=-1,
        lease_token=uuid4(),
        fencing_token=1,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        status="active",
    )
    session.add(negative_lease)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    mismatched_device = GPUResourceLease(
        resource=resource,
        gpu_device="wrong-device",
        owner_type="attempt",
        owner_id=uuid4(),
        reserved_memory_mb=1,
        lease_token=uuid4(),
        fencing_token=2,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        status="active",
    )
    session.add(mismatched_device)
    with pytest.raises(ValueError, match="device"):
        session.commit()
    session.rollback()


def test_checkpoint_identity_fields_are_immutable(session: Session) -> None:
    snapshot = make_snapshot(session, "checkpoint-identity")
    other_snapshot = make_snapshot(session, "checkpoint-other")
    task = TrainingTask(
        dataset_snapshot=snapshot,
        task_type="object_detection",
        model_family="yolo",
        training_config_json={},
        resource_config_json={},
        evaluation_policy_json={},
    )
    session.add(task)
    session.commit()
    attempt = TrainingAttempt(task=task, attempt_no=1, status="completed")
    other_attempt = TrainingAttempt(task=task, attempt_no=2, status="completed")
    session.add_all([attempt, other_attempt])
    session.commit()
    checkpoint = Checkpoint(
        attempt=attempt,
        dataset_snapshot=snapshot,
        parent_artifact_hash="sha256:parent",
        training_config_hash="sha256:config",
        epoch=1,
        artifact_path="checkpoints/identity.pt",
        artifact_hash="sha256:checkpoint",
        metrics_json={},
    )
    session.add(checkpoint)
    session.commit()

    other_checkpoint = Checkpoint(
        attempt=other_attempt,
        dataset_snapshot=snapshot,
        parent_artifact_hash="sha256:other-parent",
        training_config_hash="sha256:other-config",
        epoch=1,
        artifact_path="checkpoints/other.pt",
        artifact_hash="sha256:other-checkpoint",
        metrics_json={},
    )
    session.add(other_checkpoint)
    session.commit()
    attempt.latest_checkpoint_id = other_checkpoint.id
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    checkpoint.attempt_id = other_attempt.id
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()
    checkpoint.dataset_snapshot_id = other_snapshot.id
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()
    checkpoint.parent_artifact_hash = "sha256:changed"
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()


def test_training_attempt_task_and_checkpoint_metrics_are_immutable(
    session: Session,
) -> None:
    snapshot = make_snapshot(session, "attempt-task-identity")
    first_task = TrainingTask(
        dataset_snapshot=snapshot,
        task_type="object_detection",
        model_family="yolo",
        training_config_json={},
        resource_config_json={},
        evaluation_policy_json={},
    )
    second_task = TrainingTask(
        dataset_snapshot=snapshot,
        task_type="object_detection",
        model_family="yolo",
        training_config_json={},
        resource_config_json={},
        evaluation_policy_json={},
    )
    session.add_all([first_task, second_task])
    session.commit()
    attempt = TrainingAttempt(task=first_task, attempt_no=1, status="completed")
    session.add(attempt)
    session.commit()
    attempt.task_id = second_task.id
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()

    checkpoint = Checkpoint(
        attempt=attempt,
        dataset_snapshot=snapshot,
        parent_artifact_hash="sha256:parent",
        training_config_hash="sha256:config",
        epoch=1,
        artifact_path="checkpoints/metrics.pt",
        artifact_hash="sha256:metrics",
        metrics_json={"loss": {"box": 0.1}},
    )
    session.add(checkpoint)
    session.commit()
    checkpoint.metrics_json["loss"]["box"] = 0.2
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()


def test_evaluation_identity_fields_are_immutable(session: Session) -> None:
    snapshot = make_snapshot(session, "evaluation-identity")
    other_snapshot = make_snapshot(session, "evaluation-other")
    model = make_model(session)
    other_model = make_model(session)
    evaluation = Evaluation(
        model_node=model,
        dataset_snapshot=snapshot,
        evaluation_policy_json={"original": True},
    )
    session.add(evaluation)
    session.commit()

    evaluation.model_node_id = other_model.id
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()
    evaluation.dataset_snapshot_id = other_snapshot.id
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()
def test_service_contract_rechecks_current_pointer_status_changes() -> None:
    assert any(
        constraint.name == "fk_binding_current_release"
        for constraint in ModelBinding.__table__.constraints
    )


def test_runtime_identity_fields_are_immutable(session: Session) -> None:
    model = make_model(session)
    binding = ModelBinding(status="unbound")
    session.add(binding)
    session.flush()
    release = make_binding_release(session, binding, model, 1)
    runtime = RuntimeInstance(
        binding=binding,
        release=release,
        model_node=model,
        config_hash="original",
        generation=1,
        fencing_token=1,
        status="ready",
        gpu_device="0",
        reserved_memory_mb=128,
    )
    session.add(runtime)
    session.commit()

    mutations = [
        ("binding_id", uuid4()),
        ("release_id", uuid4()),
        ("model_node_id", uuid4()),
        ("config_hash", "changed"),
        ("generation", 2),
        ("fencing_token", 2),
        ("gpu_device", "1"),
        ("reserved_memory_mb", 256),
    ]
    for field, value in mutations:
        setattr(runtime, field, value)
        with pytest.raises(ValueError, match="immutable"):
            session.commit()
        session.rollback()


def test_nonnegative_constraints_are_present_for_all_runtime_counters() -> None:
    expected = {
        "label_schema_classes": "class_id >= 0",
        "training_attempts": "attempt_no >= 0",
        "training_attempts_retry": "retry_count >= 0",
        "training_attempts_epoch": "current_epoch >= 0",
        "checkpoints": "epoch >= 0",
        "evaluations_attempt": "attempt_no >= 0",
        "evaluations_retry": "retry_count >= 0",
        "runtime_generation": "generation >= 0",
        "runtime_reserved": "reserved_memory_mb >= 0",
        "runtime_actual": "actual_memory_mb IS NULL OR actual_memory_mb >= 0",
        "residency_reserved": "reserved_memory_mb >= 0",
    }
    constraint_text = " ".join(
        str(constraint.sqltext)
        for table in Base.metadata.tables.values()
        for constraint in table.constraints
        if hasattr(constraint, "sqltext")
    )
    for text in expected.values():
        assert text in constraint_text


def test_checkpoint_composite_fk_names_and_restrict_match_migration() -> None:
    checkpoint_fks = {
        constraint.name: constraint
        for constraint in TrainingAttempt.__table__.constraints
        if constraint.name
        in {
            "fk_training_attempt_latest_checkpoint_same_attempt",
            "fk_training_attempt_best_checkpoint_same_attempt",
        }
    }
    assert {
        "fk_training_attempt_latest_checkpoint_same_attempt",
        "fk_training_attempt_best_checkpoint_same_attempt",
    } <= set(checkpoint_fks)
    for constraint in checkpoint_fks.values():
        assert [element.parent.name for element in constraint.elements] in [
            ["id", "latest_checkpoint_id"],
            ["id", "best_checkpoint_id"],
        ]
        assert all(element.ondelete == "RESTRICT" for element in constraint.elements)
    assert any(
        constraint.name == "fk_binding_current_runtime"
        for constraint in ModelBinding.__table__.constraints
    )

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parent_model_node_id", uuid4()),
        ("dataset_snapshot_id", uuid4()),
        ("task_type", "classification"),
        ("model_family", "other-family"),
        ("training_config_json", {"optimizer": {"lr": 0.01}}),
        ("resource_config_json", {"gpu": {"memory_mb": 1024}}),
        ("evaluation_policy_json", {"thresholds": {"mAP50": 0.9}}),
    ],
)
def test_training_task_definition_fields_are_immutable(
    session: Session, field: str, value: object
) -> None:
    snapshot = make_snapshot(session, f"task-sealed-{field}")
    task = TrainingTask(
        dataset_snapshot=snapshot,
        task_type="object_detection",
        model_family="yolo",
        training_config_json={"optimizer": {"lr": 0.1}},
        resource_config_json={"gpu": {"memory_mb": 512}},
        evaluation_policy_json={"thresholds": {"mAP50": 0.5}},
        status="queued",
    )
    session.add(task)
    session.commit()
    setattr(task, field, value)
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()


def test_nested_manifest_and_model_metadata_changes_are_immutable(
    session: Session,
) -> None:
    snapshot = make_snapshot(session, "nested-sealed")
    snapshot.train_manifest_json[0]["path"] = "changed.jpg"
    with pytest.raises(ValueError, match="immutable"):
        session.flush()
    session.rollback()

    model = ModelNode(
        artifact_path=f"models/{uuid4()}.pt",
        artifact_hash=f"sha256:{uuid4().hex}",
        task_type="object_detection",
        model_family="yolo",
        status="approved",
        metadata_json={"format": "pt", "runtime": {"labels": ["old"]}},
    )
    session.add(model)
    session.commit()
    model.metadata_json["runtime"]["labels"].append("new")
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()


def test_evaluation_report_can_be_written_once_from_null(session: Session) -> None:
    snapshot = make_snapshot(session, "report-once")
    model = make_model(session)
    evaluation = Evaluation(model_node=model, dataset_snapshot=snapshot)
    session.add(evaluation)
    session.commit()

    evaluation.report_path = "reports/first.json"
    evaluation.report_hash = "sha256:first"
    session.commit()

    evaluation.report_path = "reports/second.json"
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()


def test_training_attempt_output_is_one_to_one_and_consistent(
    session: Session,
) -> None:
    snapshot = make_snapshot(session, "attempt-output")
    task = TrainingTask(
        dataset_snapshot=snapshot,
        task_type="object_detection",
        model_family="yolo",
        training_config_json={},
        evaluation_policy_json={},
        status="queued",
    )
    session.add(task)
    session.commit()
    attempt = TrainingAttempt(task=task, attempt_no=1, status="completed")
    other_attempt = TrainingAttempt(task=task, attempt_no=2, status="completed")
    session.add_all([attempt, other_attempt])
    session.flush()

    output_model = make_model(session, training_attempt_id=attempt.id)
    session.commit()
    assert attempt.output_model_node.id == output_model.id

    output_model.training_attempt_id = other_attempt.id
    with pytest.raises(ValueError, match="immutable"):
        session.commit()
    session.rollback()

    second_output = ModelNode(
        training_attempt_id=attempt.id,
        artifact_path=f"models/{uuid4()}.pt",
        artifact_hash=f"sha256:{uuid4().hex}",
        task_type="object_detection",
        model_family="yolo",
        status="approved",
        metadata_json={"format": "pt"},
    )
    session.add(second_output)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_operation_log_schema_and_fields(session: Session) -> None:
    request_id = uuid4()
    resource_id = uuid4()
    log = OperationLog(
        operation_type="model.imported",
        actor="admin",
        request_id=request_id,
        resource_type="model_node",
        resource_id=resource_id,
        status="success",
        summary_json={"artifact": "model.pt"},
    )
    session.add(log)
    session.commit()

    assert log.id is not None
    assert log.request_id == request_id
    assert log.resource_id == resource_id
    assert log.summary_json == {"artifact": "model.pt"}
    assert {
        "operation_type",
        "actor",
        "request_id",
        "resource_type",
        "resource_id",
        "status",
        "summary_json",
        "error_summary",
        "created_at",
    } <= set(inspect(OperationLog).columns.keys())


def test_task_and_residency_updated_at_have_onupdate() -> None:
    assert TrainingTask.__table__.c.updated_at.onupdate is not None
    assert ModelResidencyPlan.__table__.c.updated_at.onupdate is not None


def test_lineage_relationships_use_database_restrict_delete() -> None:
    assert LabelSchema.parent_schema.property.passive_deletes is True
    assert LabelSchema.child_schemas.property.passive_deletes is True
    assert ModelNode.children.property.passive_deletes is True


def test_binding_release_and_runtime_composite_references(session: Session) -> None:
    model = make_model(session)
    binding = ModelBinding(status="unbound")
    session.add(binding)
    session.flush()
    release = make_binding_release(session, binding, model, 1)
    release.status = "active"
    runtime = RuntimeInstance(
        binding=binding,
        release=release,
        model_node=model,
        config_hash=release.inference_config_hash,
        generation=1,
        fencing_token=1,
        status="serving",
        gpu_device="0",
    )
    session.add(runtime)
    session.commit()
    runtime_id = runtime.id

    binding.current_release_id = release.id
    binding.current_runtime_instance_id = runtime_id
    session.commit()
    assert binding.current_release.id == release.id
    assert binding.current_runtime_instance.id == runtime.id

    invalid_binding = ModelBinding(
        current_release_id=release.id,
        current_runtime_instance_id=runtime.id,
        status="bound",
    )
    session.add(invalid_binding)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_one_serving_and_one_runtime_candidate_per_binding(session: Session) -> None:
    model = make_model(session)
    binding = ModelBinding(status="unbound")
    session.add(binding)
    session.flush()
    serving_release = make_binding_release(session, binding, model, 1)
    candidate_release = make_binding_release(session, binding, model, 2)
    session.add_all(
        [
            RuntimeInstance(
                binding=binding,
                release=serving_release,
                model_node=model,
                config_hash="serving",
                generation=1,
                fencing_token=1,
                status="serving",
                gpu_device="0",
            ),
            RuntimeInstance(
                binding=binding,
                release=candidate_release,
                model_node=model,
                config_hash="candidate",
                generation=2,
                fencing_token=2,
                status="preparing",
                gpu_device="0",
            ),
        ]
    )
    session.commit()

    session.add(
        RuntimeInstance(
            binding=binding,
            release=serving_release,
            model_node=model,
            config_hash="second-serving",
            generation=3,
            fencing_token=3,
            status="serving",
            gpu_device="0",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    session.add(
        RuntimeInstance(
            binding=binding,
            release=candidate_release,
            model_node=model,
            config_hash="second-candidate",
            generation=4,
            fencing_token=4,
            status="ready",
            gpu_device="0",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
