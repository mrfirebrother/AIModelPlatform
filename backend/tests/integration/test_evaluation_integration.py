from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from backend.app.evaluation.yolo_evaluator import EvalResult, YoloEvaluator
from backend.app.evaluation.policy import EvaluationPolicy, validate_policy_for_run
from backend.app.evaluation.metrics import compute_mAP, compute_per_class_metrics
from backend.app.models import (
    Base,
    Dataset,
    DatasetSnapshot,
    Evaluation,
    LabelSchema,
    ModelNode,
)
from backend.app.services.evaluation_service import (
    create_evaluation,
    mark_evaluation_running,
    record_metrics,
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
    from uuid import uuid4

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
    from uuid import uuid4

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


class TestYoloEvaluatorIntegration:
    def test_full_evaluation_flow(self, session: Session, tmp_path: Path) -> None:
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

        evaluator = YoloEvaluator()
        with patch("backend.app.evaluation.yolo_evaluator.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "fire"}

            mock_result = MagicMock()
            mock_boxes = MagicMock()
            mock_boxes.xyxy = [[0, 0, 10, 10], [20, 20, 30, 30]]
            mock_boxes.conf = [0.95, 0.85]
            mock_boxes.cls = [0, 0]
            mock_result.boxes = mock_boxes
            mock_model.return_value = [mock_result]

            MockYOLO.return_value = mock_model
            evaluator.load_model(str(tmp_path / "model.pt"))

            images = [MagicMock()]
            ground_truths = [
                [
                    {"class_id": 0, "bbox": [0, 0, 10, 10]},
                    {"class_id": 0, "bbox": [20, 20, 30, 30]},
                ]
            ]

            eval_result = evaluator.evaluate(
                dataset_images=images,
                dataset_ground_truths=ground_truths,
                class_ids=[0],
            )

            assert eval_result.success is True
            assert eval_result.mAP50 == pytest.approx(1.0)

            metrics_dict = {
                "mAP50": eval_result.mAP50,
                "mAP50_95": eval_result.mAP50_95,
                "precision": eval_result.precision,
                "recall": eval_result.recall,
                "per_class": eval_result.per_class,
            }
            policy_validation = validate_policy_for_run(
                policy,
                metrics_dict,
                test_classes={0},
                schema_classes={0},
                parent_metrics=None,
            )
            assert policy_validation.passed is True

            record_metrics(session, ev.id, metrics_dict, passed=True)
            session.commit()
            session.refresh(ev)
            assert ev.auto_status == "passed"
            assert ev.auto_metrics_json["mAP50"] == pytest.approx(1.0)

    def test_evaluation_fails_policy_threshold(
        self, session: Session, tmp_path: Path
    ) -> None:
        snapshot = _make_snapshot(session)
        model = _make_model(session, snapshot)
        policy = EvaluationPolicy(
            min_mAP50=0.8,
            min_precision=0.8,
            min_recall=0.8,
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

        evaluator = YoloEvaluator()
        with patch("backend.app.evaluation.yolo_evaluator.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "fire"}

            mock_result = MagicMock()
            mock_boxes = MagicMock()
            mock_boxes.xyxy = [[50, 50, 60, 60]]
            mock_boxes.conf = [0.9]
            mock_boxes.cls = [0]
            mock_result.boxes = mock_boxes
            mock_model.return_value = [mock_result]

            MockYOLO.return_value = mock_model
            evaluator.load_model(str(tmp_path / "model.pt"))

            images = [MagicMock()]
            ground_truths = [[{"class_id": 0, "bbox": [0, 0, 10, 10]}]]

            eval_result = evaluator.evaluate(
                dataset_images=images,
                dataset_ground_truths=ground_truths,
                class_ids=[0],
            )

            metrics_dict = {
                "mAP50": eval_result.mAP50,
                "mAP50_95": eval_result.mAP50_95,
                "precision": eval_result.precision,
                "recall": eval_result.recall,
                "per_class": eval_result.per_class,
            }
            policy_validation = validate_policy_for_run(
                policy,
                metrics_dict,
                test_classes={0},
                schema_classes={0},
                parent_metrics=None,
            )
            assert policy_validation.passed is False
            assert len(policy_validation.regression_failures) > 0

            record_metrics(session, ev.id, metrics_dict, passed=False)
            session.commit()
            session.refresh(ev)
            assert ev.auto_status == "failed"

    def test_per_class_metrics_calculation(self, tmp_path: Path) -> None:
        evaluator = YoloEvaluator()
        with patch("backend.app.evaluation.yolo_evaluator.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "fire", 1: "smoke"}

            mock_result = MagicMock()
            mock_boxes = MagicMock()
            mock_boxes.xyxy = [[0, 0, 10, 10], [20, 20, 30, 30]]
            mock_boxes.conf = [0.95, 0.85]
            mock_boxes.cls = [0, 1]
            mock_result.boxes = mock_boxes
            mock_model.return_value = [mock_result]

            MockYOLO.return_value = mock_model
            evaluator.load_model(str(tmp_path / "model.pt"))

            images = [MagicMock()]
            ground_truths = [
                [
                    {"class_id": 0, "bbox": [0, 0, 10, 10]},
                    {"class_id": 1, "bbox": [20, 20, 30, 30]},
                ]
            ]

            eval_result = evaluator.evaluate(
                dataset_images=images,
                dataset_ground_truths=ground_truths,
                class_ids=[0, 1],
            )

            assert eval_result.success is True
            assert 0 in eval_result.per_class
            assert 1 in eval_result.per_class
            assert eval_result.per_class[0]["precision"] == pytest.approx(1.0)
            assert eval_result.per_class[1]["recall"] == pytest.approx(1.0)

    def test_empty_dataset_evaluation(self, tmp_path: Path) -> None:
        evaluator = YoloEvaluator()
        with patch("backend.app.evaluation.yolo_evaluator.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "fire"}
            MockYOLO.return_value = mock_model

            evaluator.load_model(str(tmp_path / "model.pt"))
            result = evaluator.evaluate(
                dataset_images=[],
                dataset_ground_truths=[],
                class_ids=[0],
            )
            assert result.success is True
            assert result.num_images == 0
            assert result.mAP50 == 0.0
