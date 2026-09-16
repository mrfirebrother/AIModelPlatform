from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

from backend.app.evaluation.metrics import (
    compute_mAP,
    compute_per_class_metrics,
    compute_precision_recall,
)
from backend.app.evaluation.policy import EvaluationPolicy, validate_policy_for_run
from backend.app.evaluation.report import generate_evaluation_report, write_report
from backend.app.models import (
    Base,
    Dataset,
    DatasetSnapshot,
    Evaluation,
    LabelSchema,
    LabelSchemaClass,
    ModelNode,
)
from backend.app.services.evaluation_service import (
    create_evaluation,
    get_evaluation,
    list_pending_evaluations,
    mark_evaluation_failed,
    mark_evaluation_running,
    record_human_review,
    record_metrics,
    write_report_path,
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


def _make_model(session: Session, snapshot: DatasetSnapshot) -> ModelNode:
    model = ModelNode(
        dataset_snapshot_id=snapshot.id,
        artifact_path=f"models/{uuid4()}.pt",
        artifact_hash=f"sha256:{uuid4().hex}",
        task_type="object_detection",
        model_family="yolo",
        status="candidate",
        metadata_json={},
    )
    session.add(model)
    session.flush()
    return model


class TestCreateEvaluation:
    def test_creates_pending_evaluation(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        model = _make_model(session, snapshot)
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        ev = create_evaluation(
            session,
            model_node_id=model.id,
            dataset_snapshot_id=snapshot.id,
            policy=policy,
        )
        session.commit()
        assert ev.auto_status == "pending"
        assert ev.attempt_status == "pending"
        assert ev.evaluation_policy_json["min_mAP50"] == 0.5

    def test_policy_frozen_on_creation(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        model = _make_model(session, snapshot)
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        ev = create_evaluation(
            session,
            model_node_id=model.id,
            dataset_snapshot_id=snapshot.id,
            policy=policy,
        )
        session.commit()
        with pytest.raises(ValueError, match="immutable"):
            ev.evaluation_policy_json = {"min_mAP50": 0.9}
            session.commit()


class TestRecordMetrics:
    def test_metrics_written_once(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        model = _make_model(session, snapshot)
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        ev = create_evaluation(
            session,
            model_node_id=model.id,
            dataset_snapshot_id=snapshot.id,
            policy=policy,
        )
        session.commit()

        mark_evaluation_running(session, ev.id)
        session.commit()

        metrics = {
            "mAP50": 0.6,
            "mAP50_95": 0.4,
            "precision": 0.7,
            "recall": 0.6,
            "per_class": {0: {"precision": 0.7, "recall": 0.6}},
        }
        record_metrics(session, ev.id, metrics, passed=True)
        session.commit()

        session.refresh(ev)
        assert ev.auto_status == "passed"
        assert ev.auto_metrics_json["mAP50"] == 0.6

    def test_metrics_cannot_be_overwritten(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        model = _make_model(session, snapshot)
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        ev = create_evaluation(
            session,
            model_node_id=model.id,
            dataset_snapshot_id=snapshot.id,
            policy=policy,
        )
        session.commit()
        mark_evaluation_running(session, ev.id)
        session.commit()
        record_metrics(
            session, ev.id, {"mAP50": 0.6, "precision": 0.7, "recall": 0.6}, passed=True
        )
        session.commit()

        with pytest.raises(ValueError, match="Cannot record metrics"):
            record_metrics(
                session, ev.id, {"mAP50": 0.3, "precision": 0.3, "recall": 0.3}, passed=True
            )
            session.commit()


class TestWriteReportPath:
    def test_report_path_frozen_after_write(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        model = _make_model(session, snapshot)
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        ev = create_evaluation(
            session,
            model_node_id=model.id,
            dataset_snapshot_id=snapshot.id,
            policy=policy,
        )
        session.commit()
        mark_evaluation_running(session, ev.id)
        session.commit()
        record_metrics(
            session, ev.id, {"mAP50": 0.6, "precision": 0.7, "recall": 0.6}, passed=True
        )
        session.commit()
        write_report_path(session, ev.id, "/reports/ev1.json", "sha256:abc")
        session.commit()

        session.refresh(ev)
        assert ev.report_path == "/reports/ev1.json"
        assert ev.report_hash == "sha256:abc"

        with pytest.raises(ValueError, match="immutable"):
            ev.report_path = "/reports/ev2.json"
            session.commit()


class TestHumanReviewIsGone:
    """人工复核（以及它依赖的通过/未通过判定）已按需求移除。

    平台没有审核机制：评测只产出指标，没有人负责批准/拒绝，因此这条链路不该再存在。
    这里锁住"它确实没了"，避免以后有人又把它加回来造成误读。
    """

    def test_service_rejects_a_review_call(self, session: Session) -> None:
        with pytest.raises(ValueError, match="人工复核已移除"):
            record_human_review(
                session, uuid4(), reviewer="admin", conclusion="approved"
            )

    def test_review_endpoint_is_not_registered(self) -> None:
        from backend.app.main import create_app

        paths = {getattr(route, "path", "") for route in create_app().routes}
        assert not any(path.endswith("/review") for path in paths)


class TestEvaluationServiceFlow:
    def test_full_evaluation_flow(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        model = _make_model(session, snapshot)
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        ev = create_evaluation(
            session,
            model_node_id=model.id,
            dataset_snapshot_id=snapshot.id,
            policy=policy,
        )
        session.commit()
        assert ev.auto_status == "pending"

        mark_evaluation_running(session, ev.id)
        session.commit()
        session.refresh(ev)
        assert ev.auto_status == "running"
        assert ev.attempt_status == "running"

        metrics = {
            "mAP50": 0.65,
            "mAP50_95": 0.45,
            "precision": 0.75,
            "recall": 0.65,
            "per_class": {0: {"precision": 0.75, "recall": 0.65}},
        }
        record_metrics(session, ev.id, metrics, passed=True)
        session.commit()
        session.refresh(ev)
        assert ev.auto_status == "passed"
        assert ev.auto_metrics_json["mAP50"] == 0.65


    def test_evaluation_failed_auto_status(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        model = _make_model(session, snapshot)
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        ev = create_evaluation(
            session,
            model_node_id=model.id,
            dataset_snapshot_id=snapshot.id,
            policy=policy,
        )
        session.commit()
        mark_evaluation_running(session, ev.id)
        session.commit()
        record_metrics(
            session, ev.id, {"mAP50": 0.2, "precision": 0.3, "recall": 0.2}, passed=False
        )
        session.commit()
        session.refresh(ev)
        assert ev.auto_status == "failed"

    def test_list_pending_evaluations(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        model = _make_model(session, snapshot)
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        ev1 = create_evaluation(
            session,
            model_node_id=model.id,
            dataset_snapshot_id=snapshot.id,
            policy=policy,
        )
        ev2 = create_evaluation(
            session,
            model_node_id=model.id,
            dataset_snapshot_id=snapshot.id,
            policy=policy,
        )
        session.commit()

        pending = list_pending_evaluations(session)
        assert len(pending) == 2

        mark_evaluation_running(session, ev1.id)
        session.commit()
        pending = list_pending_evaluations(session)
        assert len(pending) == 1

    def test_mark_evaluation_failed(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        model = _make_model(session, snapshot)
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        ev = create_evaluation(
            session,
            model_node_id=model.id,
            dataset_snapshot_id=snapshot.id,
            policy=policy,
        )
        session.commit()
        mark_evaluation_running(session, ev.id)
        session.commit()
        mark_evaluation_failed(session, ev.id, "GPU OOM")
        session.commit()
        session.refresh(ev)
        assert ev.auto_status == "failed"
        assert ev.last_error == "GPU OOM"

    def test_get_evaluation(self, session: Session) -> None:
        snapshot = _make_snapshot(session)
        model = _make_model(session, snapshot)
        policy = EvaluationPolicy(
            min_mAP50=0.5,
            min_precision=0.4,
            min_recall=0.4,
            max_regression_ratio=0.1,
            require_per_class_coverage=True,
        )
        ev = create_evaluation(
            session,
            model_node_id=model.id,
            dataset_snapshot_id=snapshot.id,
            policy=policy,
        )
        session.commit()
        fetched = get_evaluation(session, ev.id)
        assert fetched is not None
        assert fetched.id == ev.id
        assert get_evaluation(session, uuid4()) is None
