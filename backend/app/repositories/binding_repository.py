from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models import BindingRelease, ModelBinding


def create_binding(
    session: Session,
    *,
    external_ref: str | None = None,
) -> ModelBinding:
    binding = ModelBinding(
        external_ref=external_ref,
        status="unbound",
    )
    session.add(binding)
    session.flush()
    return binding


def get_binding(session: Session, binding_id: UUID) -> ModelBinding | None:
    return session.get(ModelBinding, binding_id)


def list_bindings(session: Session) -> list[ModelBinding]:
    return list(session.execute(select(ModelBinding)).scalars().all())


def get_active_release(session: Session, binding_id: UUID) -> BindingRelease | None:
    stmt = select(BindingRelease).where(
        BindingRelease.binding_id == binding_id,
        BindingRelease.status == "active",
    )
    return session.execute(stmt).scalars().first()


def get_latest_revision(session: Session, binding_id: UUID) -> int:
    stmt = select(func.max(BindingRelease.revision_no)).where(
        BindingRelease.binding_id == binding_id
    )
    result = session.execute(stmt).scalar_one_or_none()
    return result if result is not None else 0


def create_release(
    session: Session,
    *,
    binding_id: UUID,
    model_node_id: UUID,
    inference_config_json: dict[str, Any] | None = None,
    reason: str | None = None,
    release_type: str = "normal",
    rollback_target_release_id: UUID | None = None,
) -> BindingRelease:
    binding = session.get(ModelBinding, binding_id)
    if binding is None:
        raise ValueError(f"Binding {binding_id} not found")
    next_revision = get_latest_revision(session, binding_id) + 1
    config_json = inference_config_json or {}
    config_hash = _compute_hash(config_json)
    release = BindingRelease(
        binding_id=binding_id,
        revision_no=next_revision,
        model_node_id=model_node_id,
        inference_config_json=config_json,
        inference_config_hash=config_hash,
        release_type=release_type,
        rollback_target_release_id=rollback_target_release_id,
        status="pending",
        reason=reason,
    )
    session.add(release)
    session.flush()
    return release


def update_current_release(
    session: Session,
    binding_id: UUID,
    release_id: UUID,
) -> None:
    binding = session.get(ModelBinding, binding_id)
    if binding is None:
        raise ValueError(f"Binding {binding_id} not found")
    binding.current_release_id = release_id
    session.flush()


def _compute_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{h}"
