from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from backend.app.models import BindingRelease, ModelBinding, ModelNode
from backend.app.repositories.binding_repository import (
    create_release,
    get_active_release,
    update_current_release,
)


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
