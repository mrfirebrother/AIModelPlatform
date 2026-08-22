from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from backend.app.models import BindingRelease, ModelBinding, ModelNode, RuntimeInstance
from backend.app.services.binding_service import activate_release, create_rollback_release
from backend.app.services.release_service import ReleaseService


@pytest.fixture
def mock_session():
    session = MagicMock(spec=Session)
    session.flush = MagicMock()
    session.commit = MagicMock()
    return session


@pytest.fixture
def mock_binding():
    binding = MagicMock(spec=ModelBinding)
    binding.id = uuid4()
    binding.status = "bound"
    return binding


@pytest.fixture
def mock_release():
    release = MagicMock(spec=BindingRelease)
    release.id = uuid4()
    release.binding_id = uuid4()
    release.model_node_id = uuid4()
    release.revision_no = 1
    release.status = "pending"
    release.release_type = "normal"
    release.inference_config_json = {"temperature": 0.7}
    return release


@pytest.fixture
def mock_model():
    model = MagicMock(spec=ModelNode)
    model.id = uuid4()
    model.status = "approved"
    return model


@pytest.fixture
def release_service():
    return ReleaseService()


class TestPublishRelease:
    def test_publish_pending_release(
        self, release_service, mock_session, mock_binding, mock_release, mock_model
    ):
        mock_session.get.side_effect = lambda cls, id: {
            ModelBinding: mock_binding,
            BindingRelease: mock_release,
            ModelNode: mock_model,
        }.get(cls)

        result = release_service.publish_release(mock_session, mock_release.id)
        assert result.status == "active"
        assert mock_release.status == "active"
        mock_session.flush.assert_called()

    def test_publish_fails_for_non_approved_model(
        self, release_service, mock_session, mock_binding, mock_release
    ):
        model = MagicMock(spec=ModelNode)
        model.status = "pending"
        mock_session.get.side_effect = lambda cls, id: {
            ModelBinding: mock_binding,
            BindingRelease: mock_release,
            ModelNode: model,
        }.get(cls)

        with pytest.raises(ValueError, match="non-approved model"):
            release_service.publish_release(mock_session, mock_release.id)


class TestRollbackRelease:
    def test_create_rollback(
        self, release_service, mock_session, mock_binding, mock_release, mock_model
    ):
        target_release = MagicMock(spec=BindingRelease)
        target_release.id = uuid4()
        target_release.binding_id = mock_binding.id
        target_release.inference_config_json = {"temperature": 0.5}

        mock_session.get.side_effect = lambda cls, id: {
            ModelBinding: mock_binding,
            BindingRelease: target_release,
            ModelNode: mock_model,
        }.get(cls)

        result = release_service.create_rollback(
            mock_session,
            binding_id=mock_binding.id,
            target_release_id=target_release.id,
            model_node_id=mock_model.id,
        )
        assert result.release_type == "rollback"
        assert result.rollback_target_release_id == target_release.id
        mock_session.flush.assert_called()

    def test_rollback_creates_new_revision(
        self, release_service, mock_session, mock_binding, mock_model
    ):
        old_release = MagicMock(spec=BindingRelease)
        old_release.id = uuid4()
        old_release.binding_id = mock_binding.id
        old_release.revision_no = 2
        old_release.inference_config_json = {"temperature": 0.5}

        mock_session.get.side_effect = lambda cls, id: {
            ModelBinding: mock_binding,
            BindingRelease: old_release,
            ModelNode: mock_model,
        }.get(cls)

        mock_session.execute.return_value.scalar_one_or_none.return_value = 2

        result = release_service.create_rollback(
            mock_session,
            binding_id=mock_binding.id,
            target_release_id=old_release.id,
            model_node_id=mock_model.id,
        )
        assert result.revision_no == 3


class TestStatusTransition:
    def test_valid_transitions(self, release_service):
        valid = [
            ("pending", "preparing"),
            ("preparing", "active"),
            ("pending", "active"),
        ]
        for from_status, to_status in valid:
            assert release_service.is_valid_transition(from_status, to_status) is True

    def test_invalid_transitions(self, release_service):
        invalid = [
            ("active", "pending"),
            ("failed", "active"),
            ("superseded", "pending"),
        ]
        for from_status, to_status in invalid:
            assert release_service.is_valid_transition(from_status, to_status) is False


class TestBindingConstraint:
    def test_one_serving_per_binding(self, release_service, mock_session, mock_binding):
        serving = MagicMock(spec=RuntimeInstance)
        serving.status = "serving"
        mock_session.execute.return_value.scalars.return_value.first.return_value = serving

        result = release_service.has_serving_instance(mock_session, mock_binding.id)
        assert result is True

    def test_one_candidate_per_binding(self, release_service, mock_session, mock_binding):
        candidate = MagicMock(spec=RuntimeInstance)
        candidate.status = "loading"
        mock_session.execute.return_value.scalars.return_value.first.return_value = candidate

        result = release_service.has_candidate_instance(mock_session, mock_binding.id)
        assert result is True
