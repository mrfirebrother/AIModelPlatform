from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models import (
    Base,
    Evaluation,
    LabelSchema,
    LabelSchemaClass,
    ModelNode,
)
from backend.app.repositories.model_repository import (
    create_model_node,
    get_model_node,
    list_child_models,
    list_root_models,
    validate_label_schema_compatibility,
)
from backend.app.services.label_schema_service import (
    create_child_schema,
    create_root_schema,
    seal_schema,
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


def _make_evaluation(session: Session, model: ModelNode) -> Evaluation:
    schema = LabelSchema(name=f"eval-schema-{uuid4().hex[:8]}")
    session.add(schema)
    session.flush()
    from backend.app.models import DatasetSnapshot, Dataset

    dataset = Dataset(name=f"eval-ds-{uuid4().hex[:8]}")
    snapshot = DatasetSnapshot(
        dataset=dataset,
        label_schema=schema,
        manifest_path=f"snapshots/{uuid4()}/manifest.json",
        manifest_hash=f"sha256:{uuid4().hex}",
    )
    session.add(snapshot)
    session.flush()
    evaluation = Evaluation(
        model_node=model,
        dataset_snapshot=snapshot,
        auto_status="passed",
        evaluation_policy_json={"mAP50": 0.5},
    )
    session.add(evaluation)
    session.flush()
    return evaluation


class TestModelNodeCRUD:
    def test_create_root_model(self, session: Session) -> None:
        model = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/root.pt",
            artifact_hash="sha256:root",
            status="approved",
        )
        assert model.parent_id is None
        assert model.status == "approved"
        assert model.task_type == "object_detection"

    def test_create_child_model(self, session: Session) -> None:
        root = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/root.pt",
            artifact_hash="sha256:root",
            status="approved",
        )
        child = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/child.pt",
            artifact_hash="sha256:child",
            parent_id=root.id,
            status="candidate",
        )
        assert child.parent_id == root.id
        assert child.parent.id == root.id

    def test_get_model_node(self, session: Session) -> None:
        model = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/get.pt",
            artifact_hash="sha256:get",
            status="approved",
        )
        session.commit()
        fetched = get_model_node(session, model.id)
        assert fetched is not None
        assert fetched.id == model.id

    def test_list_root_models(self, session: Session) -> None:
        root1 = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/r1.pt",
            artifact_hash="sha256:r1",
            status="approved",
        )
        root2 = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/r2.pt",
            artifact_hash="sha256:r2",
            status="approved",
        )
        child = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/c1.pt",
            artifact_hash="sha256:c1",
            parent_id=root1.id,
            status="candidate",
        )
        session.commit()
        roots = list_root_models(session)
        assert len(roots) == 2
        root_ids = {r.id for r in roots}
        assert root1.id in root_ids
        assert root2.id in root_ids

    def test_list_child_models(self, session: Session) -> None:
        root = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/parent.pt",
            artifact_hash="sha256:parent",
            status="approved",
        )
        child1 = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/c1.pt",
            artifact_hash="sha256:c1",
            parent_id=root.id,
            status="candidate",
        )
        child2 = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/c2.pt",
            artifact_hash="sha256:c2",
            parent_id=root.id,
            status="candidate",
        )
        session.commit()
        children = list_child_models(session, root.id)
        assert len(children) == 2
        child_ids = {c.id for c in children}
        assert child1.id in child_ids
        assert child2.id in child_ids


class TestModelStatusTransitions:
    def test_root_model_approved_directly(self, session: Session) -> None:
        model = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/approved.pt",
            artifact_hash="sha256:approved",
            status="approved",
        )
        assert model.status == "approved"

    def test_candidate_can_be_approved(self, session: Session) -> None:
        model = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/candidate.pt",
            artifact_hash="sha256:candidate",
            status="candidate",
        )
        model.status = "approved"
        session.commit()
        assert model.status == "approved"


class TestLabelSchemaCompatibility:
    def test_model_with_matching_schema_compatible(self, session: Session) -> None:
        schema = create_root_schema(
            session, "detector", classes=[(0, "person", "Person")]
        )
        seal_schema(session, schema.id)
        model = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/compat.pt",
            artifact_hash="sha256:compat",
            label_schema_id=schema.id,
            status="approved",
        )
        session.commit()
        compatible = validate_label_schema_compatibility(session, model.id, schema.id)
        assert compatible is True

    def test_model_without_schema_compatible_with_any(self, session: Session) -> None:
        schema = create_root_schema(
            session, "any-schema", classes=[(0, "person", "Person")]
        )
        seal_schema(session, schema.id)
        model = create_model_node(
            session,
            task_type="object_detection",
            model_family="yolo",
            artifact_path="models/no-schema.pt",
            artifact_hash="sha256:no-schema",
            status="approved",
        )
        session.commit()
        compatible = validate_label_schema_compatibility(session, model.id, schema.id)
        assert compatible is True


class TestChildSchemaInheritanceOnModels:
    def test_first_generation_task_creates_new_schema(self, session: Session) -> None:
        root_schema = create_root_schema(
            session, "base", classes=[(0, "person", "Person")]
        )
        seal_schema(session, root_schema.id)

        child_schema = create_child_schema(
            session,
            parent_schema_id=root_schema.id,
            name="extended",
            new_classes=[(1, "vehicle", "Vehicle")],
        )
        seal_schema(session, child_schema.id)
        assert child_schema.parent_schema_id == root_schema.id
        assert len(child_schema.classes) == 2

    def test_subsequent_child_cannot_remove_parent_classes(
        self, session: Session
    ) -> None:
        root_schema = create_root_schema(
            session, "base-classes", classes=[(0, "person", "Person")]
        )
        seal_schema(session, root_schema.id)
        child1 = create_child_schema(
            session,
            parent_schema_id=root_schema.id,
            name="child1",
            new_classes=[(1, "vehicle", "Vehicle")],
        )
        seal_schema(session, child1.id)
        child2 = create_child_schema(
            session,
            parent_schema_id=child1.id,
            name="child2",
            new_classes=[(2, "animal", "Animal")],
        )
        session.commit()
        class_keys = {c.semantic_key for c in child2.classes}
        assert "person" in class_keys
        assert "vehicle" in class_keys
        assert "animal" in class_keys
