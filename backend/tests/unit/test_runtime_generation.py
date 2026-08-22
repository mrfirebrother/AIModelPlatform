from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from backend.app.models import BindingRelease, ModelBinding, RuntimeInstance
from backend.app.runtime.model_manager import ModelManager
from backend.app.runtime.runtime_instance import RuntimeInstanceManager


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.flush = MagicMock()
    return session


@pytest.fixture
def mock_binding():
    binding = MagicMock(spec=ModelBinding)
    binding.id = uuid4()
    return binding


@pytest.fixture
def mock_release():
    release = MagicMock(spec=BindingRelease)
    release.id = uuid4()
    release.binding_id = uuid4()
    release.model_node_id = uuid4()
    release.inference_config_json = {"temperature": 0.7}
    release.inference_config_hash = "sha256:test"
    release.status = "pending"
    return release


@pytest.fixture
def model_manager():
    return ModelManager()


@pytest.fixture
def runtime_manager():
    return RuntimeInstanceManager()


class TestModelManager:
    def test_load_model_returns_gpu_device(self, model_manager):
        gpu = model_manager.load_model("test_config")
        assert gpu is not None
        assert isinstance(gpu, str)

    def test_unload_model_success(self, model_manager):
        gpu = model_manager.load_model("test_config")
        result = model_manager.unload_model(gpu)
        assert result is True

    def test_get_memory_usage(self, model_manager):
        gpu = model_manager.load_model("test_config")
        memory = model_manager.get_memory_usage(gpu)
        assert memory >= 0


class TestRuntimeInstanceManager:
    def test_create_instance(self, runtime_manager, mock_session, mock_binding, mock_release):
        instance = runtime_manager.create_instance(
            session=mock_session,
            binding_id=mock_binding.id,
            release_id=mock_release.id,
            model_node_id=mock_release.model_node_id,
            config_hash=mock_release.inference_config_hash,
            generation=0,
            fencing_token=0,
            gpu_device="gpu:0",
        )
        assert instance is not None
        assert instance.status == "loading"
        mock_session.flush.assert_called()

    def test_transition_status(self, runtime_manager, mock_session):
        instance = MagicMock(spec=RuntimeInstance)
        instance.status = "loading"
        instance.id = uuid4()
        instance.binding_id = uuid4()
        mock_session.get.return_value = instance

        runtime_manager.transition_status(mock_session, instance.id, "preparing")
        assert instance.status == "preparing"
        mock_session.flush.assert_called()

    def test_transition_invalid(self, runtime_manager, mock_session):
        instance = MagicMock(spec=RuntimeInstance)
        instance.status = "stopped"
        instance.id = uuid4()
        instance.binding_id = uuid4()
        mock_session.get.return_value = instance

        with pytest.raises(ValueError, match="Invalid transition"):
            runtime_manager.transition_status(mock_session, instance.id, "serving")

    def test_get_serving_instance(self, runtime_manager, mock_session, mock_binding):
        instance = MagicMock(spec=RuntimeInstance)
        instance.binding_id = mock_binding.id
        instance.status = "serving"
        mock_session.execute.return_value.scalars.return_value.first.return_value = instance

        result = runtime_manager.get_serving_instance(mock_session, mock_binding.id)
        assert result == instance

    def test_get_candidate_instance(self, runtime_manager, mock_session, mock_binding):
        instance = MagicMock(spec=RuntimeInstance)
        instance.binding_id = mock_binding.id
        instance.status = "loading"
        mock_session.execute.return_value.scalars.return_value.first.return_value = instance

        result = runtime_manager.get_candidate_instance(mock_session, mock_binding.id)
        assert result == instance


class TestGenerationFencing:
    def test_create_request_fencing(self, runtime_manager, mock_session):
        instance = MagicMock(spec=RuntimeInstance)
        instance.fencing_token = 5
        instance.generation = 2
        instance.status = "serving"
        instance.id = uuid4()

        def mock_get(cls, id):
            return instance

        mock_session.get = mock_get

        token = runtime_manager.validate_request_fencing(mock_session, instance.id, 5)
        assert token == 5

    def test_validate_request_fencing_rejects_stale(self, runtime_manager, mock_session):
        instance = MagicMock(spec=RuntimeInstance)
        instance.fencing_token = 5
        instance.generation = 2
        instance.status = "serving"
        instance.id = uuid4()

        def mock_get(cls, id):
            return instance

        mock_session.get = mock_get

        with pytest.raises(ValueError, match="Stale fencing token"):
            runtime_manager.validate_request_fencing(mock_session, instance.id, 4)
