from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.models import BindingRelease, ModelBinding, ModelNode
from backend.app.repositories.binding_repository import (
    create_release,
    get_active_release,
    update_current_release,
)
from backend.app.runtime.runtime_instance import RuntimeInstanceManager


def activate_release(session: Session, release_id: UUID) -> BindingRelease:
    release = session.get(BindingRelease, release_id)
    if release is None:
        raise ValueError(f"Release {release_id} not found")
    model = session.get(ModelNode, release.model_node_id)
    if model is None:
        raise ValueError(f"Model node {release.model_node_id} not found")
    if model.status != "approved":
        raise ValueError(
            f"Cannot activate release for non-approved model (status={model.status})"
        )
    previous_active = get_active_release(session, release.binding_id)
    if previous_active is not None and previous_active.id != release_id:
        previous_active.status = "superseded"
    release.status = "active"
    update_current_release(session, release.binding_id, release_id)
    session.flush()

    # Single-process platform: stand up the serving instance synchronously rather
    # than through a separate runtime manager. Drain the previous serving
    # instance, create a fresh one for this release, and hand it to the binding
    # so `/api/v1/infer` has something to serve.
    manager = RuntimeInstanceManager()
    binding = session.get(ModelBinding, release.binding_id)
    if binding is None:
        raise ValueError(f"Binding {release.binding_id} not found")
    previous_instance = manager.get_serving_instance(session, release.binding_id)
    generation = 1
    if previous_instance is not None:
        generation = previous_instance.generation + 1
        manager.transition_status(session, previous_instance.id, "draining")
        manager.transition_status(session, previous_instance.id, "stopped")
    instance = manager.create_instance(
        session,
        binding_id=release.binding_id,
        release_id=release.id,
        model_node_id=release.model_node_id,
        config_hash=release.inference_config_hash or "",
        generation=generation,
        fencing_token=generation,
        gpu_device="0",
    )
    for target in ("preparing", "ready", "serving"):
        manager.transition_status(session, instance.id, target)
    binding.current_runtime_instance_id = instance.id
    session.flush()
    return release


def create_rollback_release(
    session: Session,
    *,
    binding_id: UUID,
    target_release_id: UUID,
    model_node_id: UUID,
    reason: str | None = None,
) -> BindingRelease:
    binding = session.get(ModelBinding, binding_id)
    if binding is None:
        raise ValueError(f"Binding {binding_id} not found")
    target_release = session.get(BindingRelease, target_release_id)
    if target_release is None:
        raise ValueError(f"Target release {target_release_id} not found")
    if target_release.binding_id != binding_id:
        raise ValueError("Target release must belong to the same binding")
    model = session.get(ModelNode, model_node_id)
    if model is None:
        raise ValueError(f"Model node {model_node_id} not found")
    if model.status != "approved":
        raise ValueError(
            f"Cannot rollback to non-approved model (status={model.status})"
        )
    release = create_release(
        session,
        binding_id=binding_id,
        model_node_id=model_node_id,
        inference_config_json=dict(target_release.inference_config_json),
        reason=reason,
        release_type="rollback",
        rollback_target_release_id=target_release_id,
    )
    session.flush()
    return release
