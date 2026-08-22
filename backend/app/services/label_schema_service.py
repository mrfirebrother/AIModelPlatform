from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import (
    LabelSchema,
    LabelSchemaClass,
    ModelNode,
    DatasetSnapshot,
)
from backend.app.models.base import ImmutableFieldError


def create_root_schema(
    session: Session,
    name: str,
    *,
    description: str | None = None,
    classes: list[tuple[int, str, str]] | None = None,
) -> LabelSchema:
    schema = LabelSchema(name=name, description=description)
    session.add(schema)
    session.flush()
    if classes:
        for class_id, semantic_key, display_name in classes:
            cls = LabelSchemaClass(
                schema_id=schema.id,
                class_id=class_id,
                semantic_key=semantic_key,
                display_name=display_name,
            )
            session.add(cls)
        session.flush()
    return schema


def create_child_schema(
    session: Session,
    *,
    parent_schema_id: UUID,
    name: str,
    description: str | None = None,
    new_classes: list[tuple[int, str, str]] | None = None,
) -> LabelSchema:
    parent = session.get(LabelSchema, parent_schema_id)
    if parent is None:
        raise ValueError(f"Parent schema {parent_schema_id} not found")
    if parent.status == "archived":
        raise ValueError("Cannot create child from archived schema")
    child = LabelSchema(name=name, description=description, parent_schema=parent)
    session.add(child)
    session.flush()
    parent_classes = (
        session.execute(
            select(LabelSchemaClass).where(LabelSchemaClass.schema_id == parent_schema_id)
        )
        .scalars()
        .all()
    )
    seen_keys: set[str] = set()
    seen_ids: set[int] = set()
    for pc in parent_classes:
        if pc.semantic_key not in seen_keys:
            cls = LabelSchemaClass(
                schema_id=child.id,
                class_id=pc.class_id,
                semantic_key=pc.semantic_key,
                display_name=pc.display_name,
                description=pc.description,
            )
            session.add(cls)
            seen_keys.add(pc.semantic_key)
            seen_ids.add(pc.class_id)
    if new_classes:
        for class_id, semantic_key, display_name in new_classes:
            if semantic_key in seen_keys:
                continue
            if class_id in seen_ids:
                max_id = max(seen_ids) + 1 if seen_ids else 0
                class_id = max_id
            cls = LabelSchemaClass(
                schema_id=child.id,
                class_id=class_id,
                semantic_key=semantic_key,
                display_name=display_name,
            )
            session.add(cls)
            seen_keys.add(semantic_key)
            seen_ids.add(class_id)
    session.flush()
    return child


def seal_schema(session: Session, schema_id: UUID) -> LabelSchema:
    schema = session.get(LabelSchema, schema_id)
    if schema is None:
        raise ValueError(f"Schema {schema_id} not found")
    schema.definition_sealed = True
    session.flush()
    return schema


def archive_schema(session: Session, schema_id: UUID) -> LabelSchema:
    schema = session.get(LabelSchema, schema_id)
    if schema is None:
        raise ValueError(f"Schema {schema_id} not found")
    if _schema_is_referenced(session, schema_id):
        raise ImmutableFieldError("Cannot archive referenced schema; use seal instead")
    schema.status = "archived"
    session.flush()
    return schema


def _schema_is_referenced(session: Session, schema_id: UUID) -> bool:
    model_ref = session.execute(
        select(ModelNode.id).where(ModelNode.label_schema_id == schema_id)
    ).scalar_one_or_none()
    if model_ref is not None:
        return True
    snapshot_ref = session.execute(
        select(DatasetSnapshot.id).where(DatasetSnapshot.label_schema_id == schema_id)
    ).scalar_one_or_none()
    if snapshot_ref is not None:
        return True
    return False
