from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from backend.app.models import (
    Base,
    Dataset,
    DatasetSnapshot,
    Evaluation,
    LabelSchema,
    LabelSchemaClass,
    ModelNode,
)
from backend.app.services.model_diff import (
    DiffReport,
    diff_models,
    get_model_history,
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


def _make_schema(session: Session, name: str, class_specs: list[tuple[int, str, str]]) -> LabelSchema:
    schema = LabelSchema(name=name)
    session.add(schema)
    session.flush()
    for class_id, semantic_key, display_name in class_specs:
        cls = LabelSchemaClass(
            schema=schema,
            class_id=class_id,
            semantic_key=semantic_key,
            display_name=display_name,
        )
        session.add(cls)
    session.flush()
    return schema


def _make_model(
    session: Session,
    *,
    task_type: str = "object_detection",
    model_family: str = "yolo",
    artifact_path: str = "models/a.pt",
    artifact_hash: str = "sha256:a",
    framework: str = "pytorch",
    artifact_format: str = "pt",
    status: str = "candidate",
    metadata_json: dict | None = None,
    label_schema_id: uuid4 | None = None,
    parent_id: uuid4 | None = None,
) -> ModelNode:
    model = ModelNode(
        task_type=task_type,
        model_family=model_family,
        artifact_path=artifact_path,
        artifact_hash=artifact_hash,
        framework=framework,
        artifact_format=artifact_format,
        status=status,
        metadata_json=metadata_json or {},
        label_schema_id=label_schema_id,
        parent_id=parent_id,
    )
    session.add(model)
    session.flush()
    return model


def _make_snapshot(session: Session) -> DatasetSnapshot:
    uid = uuid4().hex[:8]
    schema = LabelSchema(name=f"snap-schema-{uid}")
    session.add(schema)
    session.flush()
    ds = Dataset(name=f"snap-ds-{uid}", metadata_json={})
    session.add(ds)
    session.flush()
    snap = DatasetSnapshot(
        dataset_id=ds.id,
        label_schema_id=schema.id,
        manifest_path=f"manifests/test-{uid}.json",
        manifest_hash=f"sha256:test-{uid}",
    )
    session.add(snap)
    session.flush()
    return snap


def _make_evaluation(
    session: Session,
    model_node_id,
    dataset_snapshot_id=None,
    metrics: dict | None = None,
) -> Evaluation:
    if dataset_snapshot_id is None:
        snapshot = _make_snapshot(session)
        dataset_snapshot_id = snapshot.id
    ev = Evaluation(
        model_node_id=model_node_id,
        dataset_snapshot_id=dataset_snapshot_id,
        auto_status="passed",
        human_status="pending",
        evaluation_policy_json={},
        auto_metrics_json=metrics or {},
    )
    session.add(ev)
    session.flush()
    return ev


class TestDiffModelsMetrics:
    def test_same_model_returns_empty_diff(self, session: Session) -> None:
        schema = _make_schema(session, "s1", [(0, "person", "Person")])
        m = _make_model(session, label_schema_id=schema.id)
        report = diff_models(session, m.id, m.id)
        assert report.is_identical is True
        assert report.config_diffs == []
        assert report.metrics_diffs == []
        assert report.label_diffs == []

    def test_different_task_type_detected(self, session: Session) -> None:
        m1 = _make_model(session, task_type="object_detection")
        m2 = _make_model(session, task_type="classification")
        report = diff_models(session, m1.id, m2.id)
        assert report.is_identical is False
        assert any(d.field == "task_type" for d in report.config_diffs)

    def test_different_model_family_detected(self, session: Session) -> None:
        m1 = _make_model(session, model_family="yolo")
        m2 = _make_model(session, model_family="resnet")
        report = diff_models(session, m1.id, m2.id)
        assert any(d.field == "model_family" for d in report.config_diffs)

    def test_different_framework_detected(self, session: Session) -> None:
        m1 = _make_model(session, framework="pytorch")
        m2 = _make_model(session, framework="tensorflow")
        report = diff_models(session, m1.id, m2.id)
        assert any(d.field == "framework" for d in report.config_diffs)

    def test_different_artifact_format_detected(self, session: Session) -> None:
        m1 = _make_model(session, artifact_format="pt")
        m2 = _make_model(session, artifact_format="onnx")
        report = diff_models(session, m1.id, m2.id)
        assert any(d.field == "artifact_format" for d in report.config_diffs)

    def test_different_status_detected(self, session: Session) -> None:
        m1 = _make_model(session, status="candidate")
        m2 = _make_model(session, status="approved")
        report = diff_models(session, m1.id, m2.id)
        assert any(d.field == "status" for d in report.config_diffs)


class TestDiffModelsLabels:
    def test_different_label_schemas_detected(self, session: Session) -> None:
        s1 = _make_schema(session, "s1", [(0, "person", "Person")])
        s2 = _make_schema(session, "s2", [(0, "vehicle", "Vehicle")])
        m1 = _make_model(session, label_schema_id=s1.id)
        m2 = _make_model(session, label_schema_id=s2.id)
        report = diff_models(session, m1.id, m2.id)
        assert report.label_diffs != []

    def test_extra_classes_in_second_schema(self, session: Session) -> None:
        s1 = _make_schema(session, "s1", [(0, "person", "Person")])
        s2 = _make_schema(session, "s2", [(0, "person", "Person"), (1, "car", "Car")])
        m1 = _make_model(session, label_schema_id=s1.id)
        m2 = _make_model(session, label_schema_id=s2.id)
        report = diff_models(session, m1.id, m2.id)
        assert any(d.change == "added" for d in report.label_diffs)

    def test_missing_classes_in_second_schema(self, session: Session) -> None:
        s1 = _make_schema(session, "s1", [(0, "person", "Person"), (1, "car", "Car")])
        s2 = _make_schema(session, "s2", [(0, "person", "Person")])
        m1 = _make_model(session, label_schema_id=s1.id)
        m2 = _make_model(session, label_schema_id=s2.id)
        report = diff_models(session, m1.id, m2.id)
        assert any(d.change == "removed" for d in report.label_diffs)

    def test_renamed_class_detected(self, session: Session) -> None:
        s1 = _make_schema(session, "s1", [(0, "person", "Person")])
        s2 = _make_schema(session, "s2", [(0, "pedestrian", "Pedestrian")])
        m1 = _make_model(session, label_schema_id=s1.id)
        m2 = _make_model(session, label_schema_id=s2.id)
        report = diff_models(session, m1.id, m2.id)
        assert any(d.change == "renamed" for d in report.label_diffs)

    def test_same_labels_returns_empty(self, session: Session) -> None:
        s = _make_schema(session, "s", [(0, "person", "Person")])
        m1 = _make_model(session, label_schema_id=s.id)
        m2 = _make_model(session, label_schema_id=s.id)
        report = diff_models(session, m1.id, m2.id)
        assert report.label_diffs == []


class TestDiffModelsMetricsEval:
    def test_different_eval_metrics_detected(self, session: Session) -> None:
        m1 = _make_model(session)
        m2 = _make_model(session)
        _make_evaluation(session, m1.id, metrics={"mAP": 0.85, "recall": 0.9})
        _make_evaluation(session, m2.id, metrics={"mAP": 0.72, "recall": 0.8})
        report = diff_models(session, m1.id, m2.id)
        assert report.metrics_diffs != []

    def test_same_eval_metrics_returns_empty(self, session: Session) -> None:
        m1 = _make_model(session)
        m2 = _make_model(session)
        _make_evaluation(session, m1.id, metrics={"mAP": 0.85})
        _make_evaluation(session, m2.id, metrics={"mAP": 0.85})
        report = diff_models(session, m1.id, m2.id)
        assert report.metrics_diffs == []

    def test_metric_only_in_one_model(self, session: Session) -> None:
        m1 = _make_model(session)
        m2 = _make_model(session)
        _make_evaluation(session, m1.id, metrics={"mAP": 0.85})
        _make_evaluation(session, m2.id, metrics={"precision": 0.9})
        report = diff_models(session, m1.id, m2.id)
        assert any(d.change == "added" for d in report.metrics_diffs)
        assert any(d.change == "removed" for d in report.metrics_diffs)


class TestDiffModelsNotFound:
    def test_first_model_not_found(self, session: Session) -> None:
        with pytest.raises(ValueError, match="not found"):
            diff_models(session, uuid4(), uuid4())

    def test_second_model_not_found(self, session: Session) -> None:
        m = _make_model(session)
        with pytest.raises(ValueError, match="not found"):
            diff_models(session, m.id, uuid4())


class TestGetModelHistory:
    def test_single_model_history(self, session: Session) -> None:
        m = _make_model(session)
        history = get_model_history(session, m.id)
        assert len(history) == 1
        assert history[0].id == m.id

    def test_parent_child_history(self, session: Session) -> None:
        m1 = _make_model(session)
        m2 = _make_model(session, parent_id=m1.id)
        m3 = _make_model(session, parent_id=m2.id)
        history = get_model_history(session, m3.id)
        ids = [h.id for h in history]
        assert m1.id in ids
        assert m2.id in ids
        assert m3.id in ids

    def test_history_not_found(self, session: Session) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_model_history(session, uuid4())


class TestDiffReportStructure:
    def test_report_has_required_fields(self, session: Session) -> None:
        m = _make_model(session)
        report = diff_models(session, m.id, m.id)
        assert hasattr(report, "base_model_id")
        assert hasattr(report, "compare_model_id")
        assert hasattr(report, "is_identical")
        assert hasattr(report, "config_diffs")
        assert hasattr(report, "metrics_diffs")
        assert hasattr(report, "label_diffs")
        assert hasattr(report, "summary")

    def test_diff_item_has_field_old_new(self, session: Session) -> None:
        m1 = _make_model(session, framework="pytorch")
        m2 = _make_model(session, framework="tensorflow")
        report = diff_models(session, m1.id, m2.id)
        item = next(d for d in report.config_diffs if d.field == "framework")
        assert item.old_value == "pytorch"
        assert item.new_value == "tensorflow"
