from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import LabelSchema, LabelSchemaClass, ModelNode


def create_model_node(
    session: Session,
    *,
    task_type: str,
    model_family: str,
    artifact_path: str,
    artifact_hash: str,
    name: str = "",
    parent_id: UUID | None = None,
    label_schema_id: UUID | None = None,
    dataset_snapshot_id: UUID | None = None,
    training_attempt_id: UUID | None = None,
    framework: str = "pytorch",
    artifact_format: str = "pt",
    status: str = "candidate",
    metadata_json: dict[str, Any] | None = None,
) -> ModelNode:
    if parent_id is not None:
        parent = session.get(ModelNode, parent_id)
        if parent is None:
            raise ValueError(f"Parent model node {parent_id} not found")
    if label_schema_id is not None:
        schema = session.get(LabelSchema, label_schema_id)
        if schema is None:
            raise ValueError(f"Label schema {label_schema_id} not found")
    model = ModelNode(
        name=name,
        task_type=task_type,
        model_family=model_family,
        artifact_path=artifact_path,
        artifact_hash=artifact_hash,
        parent_id=parent_id,
        label_schema_id=label_schema_id,
        dataset_snapshot_id=dataset_snapshot_id,
        training_attempt_id=training_attempt_id,
        framework=framework,
        artifact_format=artifact_format,
        status=status,
        metadata_json=metadata_json or {},
    )
    session.add(model)
    session.flush()
    return model


def get_model_node(session: Session, model_id: UUID) -> ModelNode | None:
    return session.get(ModelNode, model_id)


def list_root_models(session: Session) -> list[ModelNode]:
    stmt = select(ModelNode).where(ModelNode.parent_id.is_(None))
    return list(session.execute(stmt).scalars().all())


def list_child_models(session: Session, parent_id: UUID) -> list[ModelNode]:
    stmt = select(ModelNode).where(ModelNode.parent_id == parent_id)
    return list(session.execute(stmt).scalars().all())


def validate_label_schema_compatibility(
    session: Session,
    model_id: UUID,
    schema_id: UUID,
) -> bool:
    model = session.get(ModelNode, model_id)
    if model is None:
        raise ValueError(f"Model node {model_id} not found")
    schema = session.get(LabelSchema, schema_id)
    if schema is None:
        raise ValueError(f"Label schema {schema_id} not found")
    if model.label_schema_id is None:
        return True
    if model.label_schema_id == schema_id:
        return True
    model_schema_classes = (
        session.execute(
            select(LabelSchemaClass.semantic_key).where(
                LabelSchemaClass.schema_id == model.label_schema_id
            )
        )
        .scalars()
        .all()
    )
    schema_classes = (
        session.execute(
            select(LabelSchemaClass.semantic_key).where(
                LabelSchemaClass.schema_id == schema_id
            )
        )
        .scalars()
        .all()
    )
    model_keys = set(model_schema_classes)
    schema_keys = set(schema_classes)
    return model_keys == schema_keys
