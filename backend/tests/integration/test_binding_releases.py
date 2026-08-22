from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models import (
    Base,
    BindingRelease,
    Evaluation,
    LabelSchema,
    ModelBinding,
    ModelNode,
)
from backend.app.repositories.binding_repository import (
    create_binding,
    create_release,
    get_active_release,
    get_binding,
    get_latest_revision,
    update_current_release,
)
from backend.app.services.binding_service import (
    activate_release,
    create_rollback_release,
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


def _make_approved_model(session: Session) -> ModelNode:
    model = ModelNode(
        artifact_path=f"models/{uuid4()}.pt",
        artifact_hash=f"sha256:{uuid4().hex}",
        task_type="object_detection",
        model_family="yolo",
        status="approved",
        metadata_json={},
    )
    session.add(model)
    session.flush()
    return model


def _make_evaluation(session: Session, model: ModelNode) -> Evaluation:
    schema = LabelSchema(name=f"eval-{uuid4().hex[:8]}")
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


class TestBindingCRUD:
    def test_create_empty_binding(self, session: Session) -> None:
        binding = create_binding(session)
        assert binding.status == "unbound"
        assert binding.current_release_id is None

    def test_create_binding_with_external_ref(self, session: Session) -> None:
        binding = create_binding(session, external_ref="svc-a")
        assert binding.external_ref == "svc-a"

    def test_get_binding(self, session: Session) -> None:
        binding = create_binding(session)
        session.commit()
        fetched = get_binding(session, binding.id)
        assert fetched is not None
        assert fetched.id == binding.id


class TestReleaseCRUD:
    def test_create_first_release(self, session: Session) -> None:
        binding = create_binding(session)
        model = _make_approved_model(session)
        session.flush()
        release = create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={"confidence": 0.5},
        )
        assert release.revision_no == 1
        assert release.release_type == "normal"
        assert release.binding_id == binding.id
        assert release.model_node_id == model.id

    def test_create_second_release_increments_revision(self, session: Session) -> None:
        binding = create_binding(session)
        model = _make_approved_model(session)
        session.flush()
        r1 = create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={},
        )
        r2 = create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={},
        )
        assert r2.revision_no == 2

    def test_get_active_release(self, session: Session) -> None:
        binding = create_binding(session)
        model = _make_approved_model(session)
        session.flush()
        release = create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={},
        )
        session.commit()
        active = get_active_release(session, binding.id)
        assert active is None

        release.status = "active"
        session.commit()
        active = get_active_release(session, binding.id)
        assert active is not None
        assert active.id == release.id

    def test_get_latest_revision(self, session: Session) -> None:
        binding = create_binding(session)
        model = _make_approved_model(session)
        session.flush()
        create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={},
        )
        create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={},
        )
        session.commit()
        latest = get_latest_revision(session, binding.id)
        assert latest == 2


class TestCurrentPointerUpdate:
    def test_update_current_release(self, session: Session) -> None:
        binding = create_binding(session)
        model = _make_approved_model(session)
        session.flush()
        release = create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={},
        )
        release.status = "active"
        session.flush()
        update_current_release(session, binding.id, release.id)
        session.commit()
        assert binding.current_release_id == release.id


class TestBindingService:
    def test_activate_release_sets_current(self, session: Session) -> None:
        binding = create_binding(session)
        model = _make_approved_model(session)
        session.flush()
        release = create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={},
        )
        session.commit()
        activated = activate_release(session, release.id)
        assert activated.status == "active"
        assert binding.current_release_id == release.id

    def test_activate_supersedes_previous(self, session: Session) -> None:
        binding = create_binding(session)
        model = _make_approved_model(session)
        session.flush()
        r1 = create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={},
        )
        r2 = create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={},
        )
        session.commit()
        activate_release(session, r1.id)
        activate_release(session, r2.id)
        assert r1.status == "superseded"
        assert r2.status == "active"
        assert binding.current_release_id == r2.id

    def test_rollback_creates_new_release(self, session: Session) -> None:
        binding = create_binding(session)
        model = _make_approved_model(session)
        session.flush()
        r1 = create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={},
        )
        activate_release(session, r1.id)
        r2 = create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={},
        )
        activate_release(session, r2.id)
        session.commit()

        rollback = create_rollback_release(
            session,
            binding_id=binding.id,
            target_release_id=r1.id,
            model_node_id=model.id,
        )
        assert rollback.release_type == "rollback"
        assert rollback.rollback_target_release_id == r1.id
        assert rollback.revision_no == 3


class TestReleaseModelRequirement:
    def test_release_requires_approved_model(self, session: Session) -> None:
        binding = create_binding(session)
        model = ModelNode(
            artifact_path=f"models/{uuid4()}.pt",
            artifact_hash=f"sha256:{uuid4().hex}",
            task_type="object_detection",
            model_family="yolo",
            status="candidate",
            metadata_json={},
        )
        session.add(model)
        session.flush()
        release = create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={},
        )
        assert release.status == "pending"

    def test_release_requires_evaluation(self, session: Session) -> None:
        binding = create_binding(session)
        model = _make_approved_model(session)
        session.flush()
        release = create_release(
            session,
            binding_id=binding.id,
            model_node_id=model.id,
            inference_config_json={},
        )
        eval = _make_evaluation(session, model)
        session.commit()
        assert eval.auto_status == "passed"
