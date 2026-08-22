from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import RuntimeInstance


VALID_TRANSITIONS = {
    "loading": {"preparing", "failed"},
    "preparing": {"ready", "failed"},
    "ready": {"serving", "draining", "failed"},
    "serving": {"draining", "stopped", "failed"},
    "draining": {"stopped", "failed"},
    "stopped": set(),
    "failed": set(),
}


class RuntimeInstanceManager:
    """Manages the lifecycle of runtime instances."""

    def create_instance(
        self,
        session: Session,
        *,
        binding_id: UUID,
        release_id: UUID,
        model_node_id: UUID,
        config_hash: str,
        generation: int,
        fencing_token: int,
        gpu_device: str,
    ) -> RuntimeInstance:
        """Create a new runtime instance in loading state."""
        instance = RuntimeInstance(
            binding_id=binding_id,
            release_id=release_id,
            model_node_id=model_node_id,
            config_hash=config_hash,
            generation=generation,
            fencing_token=fencing_token,
            status="loading",
            gpu_device=gpu_device,
            reserved_memory_mb=0,
        )
        session.add(instance)
        session.flush()
        return instance

    def transition_status(
        self, session: Session, instance_id: UUID, new_status: str
    ) -> RuntimeInstance:
        """Transition a runtime instance to a new status."""
        instance = session.get(RuntimeInstance, instance_id)
        if instance is None:
            raise ValueError(f"RuntimeInstance {instance_id} not found")
        allowed = VALID_TRANSITIONS.get(instance.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition from {instance.status} to {new_status}"
            )
        instance.status = new_status
        session.flush()
        return instance

    def get_serving_instance(
        self, session: Session, binding_id: UUID
    ) -> RuntimeInstance | None:
        """Get the serving instance for a binding."""
        stmt = select(RuntimeInstance).where(
            RuntimeInstance.binding_id == binding_id,
            RuntimeInstance.status == "serving",
        )
        return session.execute(stmt).scalars().first()

    def get_candidate_instance(
        self, session: Session, binding_id: UUID
    ) -> RuntimeInstance | None:
        """Get the candidate (non-serving, non-terminal) instance for a binding."""
        stmt = select(RuntimeInstance).where(
            RuntimeInstance.binding_id == binding_id,
            RuntimeInstance.status.in_(["loading", "preparing", "ready"]),
        )
        return session.execute(stmt).scalars().first()

    def validate_request_fencing(
        self, session: Session, instance_id: UUID, request_fencing_token: int
    ) -> int:
        """Validate the request fencing token against the instance."""
        instance = session.get(RuntimeInstance, instance_id)
        if instance is None:
            raise ValueError(f"RuntimeInstance {instance_id} not found")
        if instance.fencing_token != request_fencing_token:
            raise ValueError(
                f"Stale fencing token: expected {instance.fencing_token}, "
                f"got {request_fencing_token}"
            )
        return instance.fencing_token
