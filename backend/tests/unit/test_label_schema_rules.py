from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models import Base, LabelSchema, LabelSchemaClass, ModelNode


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


def _make_schema(session: Session, name: str) -> LabelSchema:
    schema = LabelSchema(name=name)
    session.add(schema)
    session.flush()
    return schema


def _make_class(
    session: Session,
    schema: LabelSchema,
    class_id: int,
    semantic_key: str,
    display_name: str,
) -> LabelSchemaClass:
    cls = LabelSchemaClass(
        schema=schema,
        class_id=class_id,
        semantic_key=semantic_key,
        display_name=display_name,
    )
    session.add(cls)
    session.flush()
    return cls


class TestRootSchemaCreation:
    def test_root_schema_has_no_parent(self, session: Session) -> None:
        schema = _make_schema(session, "root")
        assert schema.parent_schema_id is None

    def test_root_schema_can_have_classes(self, session: Session) -> None:
        schema = _make_schema(session, "root")
        cls = _make_class(session, schema, 0, "person", "Person")
        session.commit()
        assert cls.schema_id == schema.id
        assert schema.classes[0].semantic_key == "person"


class TestChildSchemaInheritance:
    def test_child_schema_must_reference_parent(self, session: Session) -> None:
        parent = _make_schema(session, "parent")
        parent_cls = _make_class(session, parent, 0, "person", "Person")
        session.commit()

        child = LabelSchema(name="child", parent_schema=parent)
        child_cls = LabelSchemaClass(
            schema=child,
            class_id=0,
            semantic_key="person",
            display_name="Person",
        )
        child_cls2 = LabelSchemaClass(
            schema=child,
            class_id=1,
            semantic_key="vehicle",
            display_name="Vehicle",
        )
        session.add_all([child, child_cls, child_cls2])
        session.commit()

        assert child.parent_schema_id == parent.id
        assert {c.semantic_key for c in child.classes} == {"person", "vehicle"}

    def test_child_schema_reuses_parent_class_ids(self, session: Session) -> None:
        parent = _make_schema(session, "parent")
        _make_class(session, parent, 0, "person", "Person")
        session.commit()

        child = LabelSchema(name="child", parent_schema=parent)
        child_cls = LabelSchemaClass(
            schema=child,
            class_id=0,
            semantic_key="person",
            display_name="Person",
        )
        session.add_all([child, child_cls])
        session.commit()
        assert child.parent_schema_id == parent.id


class TestClassIdUniqueness:
    def test_duplicate_class_id_same_schema_rejected(self, session: Session) -> None:
        schema = _make_schema(session, "dup-test")
        _make_class(session, schema, 0, "person", "Person")
        duplicate = LabelSchemaClass(
            schema=schema,
            class_id=0,
            semantic_key="other",
            display_name="Other",
        )
        session.add(duplicate)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

    def test_duplicate_semantic_key_same_schema_rejected(self, session: Session) -> None:
        schema = _make_schema(session, "dup-semantic")
        _make_class(session, schema, 0, "person", "Person")
        duplicate = LabelSchemaClass(
            schema=schema,
            class_id=1,
            semantic_key="person",
            display_name="Person Copy",
        )
        session.add(duplicate)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


class TestSemanticKeyImmutability:
    def test_semantic_key_cannot_change_when_sealed(self, session: Session) -> None:
        schema = _make_schema(session, "sealed-semantic")
        cls = _make_class(session, schema, 0, "person", "Person")
        schema.definition_sealed = True
        session.commit()

        cls.semantic_key = "changed"
        with pytest.raises(ValueError, match="immutable"):
            session.commit()
        session.rollback()

    def test_semantic_key_cannot_change_when_referenced(self, session: Session) -> None:
        schema = _make_schema(session, "referenced-semantic")
        cls = _make_class(session, schema, 0, "person", "Person")
        model = ModelNode(
            artifact_path="models/test.pt",
            artifact_hash="sha256:test",
            task_type="object_detection",
            model_family="yolo",
            label_schema_id=schema.id,
            status="approved",
            metadata_json={},
        )
        session.add(model)
        session.commit()

        cls.semantic_key = "changed"
        with pytest.raises(ValueError, match="referenced"):
            session.commit()
        session.rollback()


class TestSchemaArchival:
    def test_archived_schema_still_allows_classes_at_db_level(self, session: Session) -> None:
        schema = _make_schema(session, "archive-test")
        schema.status = "archived"
        session.commit()

        cls = _make_class(session, schema, 0, "person", "Person")
        session.commit()
        assert cls.semantic_key == "person"

    def test_sealed_schema_rejects_new_classes(self, session: Session) -> None:
        schema = _make_schema(session, "sealed-append")
        schema.definition_sealed = True
        session.commit()

        with pytest.raises(ValueError, match="sealed"):
            _make_class(session, schema, 0, "person", "Person")
        session.rollback()


class TestChildSchemaCannotOverrideParentClasses:
    def test_child_cannot_change_parent_class_semantic_key(self, session: Session) -> None:
        parent = _make_schema(session, "parent-override")
        _make_class(session, parent, 0, "person", "Person")
        session.commit()

        child = LabelSchema(name="child", parent_schema=parent)
        child_cls = LabelSchemaClass(
            schema=child,
            class_id=0,
            semantic_key="different",
            display_name="Different",
        )
        session.add_all([child, child_cls])
        session.commit()
        assert child_cls.semantic_key == "different"
