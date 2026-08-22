from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class _InstanceState:
    """Tracks state of a single inference instance."""

    instance_id: UUID
    model_key: str | None = None
    model_path: str | None = None
    running: bool = False


class InferenceWorker:
    """Manages YOLO inference instances: start/stop/health for serving models."""

    def __init__(self) -> None:
        self._instances: dict[UUID, _InstanceState] = {}

    def start(self, instance_id: UUID, model_path: str | None = None) -> None:
        """Start inference serving for an instance."""
        if instance_id in self._instances and self._instances[instance_id].running:
            logger.warning("Instance %s already running", instance_id)
            return

        state = _InstanceState(
            instance_id=instance_id,
            model_key=str(instance_id),
            model_path=model_path,
            running=True,
        )
        self._instances[instance_id] = state
        logger.info("Started inference instance %s", instance_id)

    def stop(self, instance_id: UUID) -> None:
        """Stop inference serving for an instance."""
        state = self._instances.get(instance_id)
        if state is not None:
            state.running = False
            logger.info("Stopped inference instance %s", instance_id)

    def is_running(self, instance_id: UUID) -> bool:
        """Check if an instance is currently running."""
        state = self._instances.get(instance_id)
        return state is not None and state.running

    def health_check(self, instance_id: UUID) -> bool:
        """Check if the worker is healthy for a given instance."""
        return self.is_running(instance_id)
