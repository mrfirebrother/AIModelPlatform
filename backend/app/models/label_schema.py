from __future__ import annotations

from datetime import datetime
from typing import List
from uuid import UUID

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, event, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from .base import Base, ImmutableFieldError, TimestampMixin, UUIDPrimaryKeyMixin, install_immutable_guard


class LabelSchema(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "label_schemas"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'archived')", name="label_schema_status"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    parent_schema_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("label_schemas.id", ondelete="RESTRICT")
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    definition_sealed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    parent_schema: Mapped[LabelSchema | None] = relationship(
        remote_side="LabelSchema.id",
        back_populates="child_schemas",
        passive_deletes=True,
    )
    child_schemas: Mapped[List[LabelSchema]] = relationship(
        back_populates="parent_schema", passive_deletes=True
    )
    classes: Mapped[List[LabelSchemaClass]] = relationship(
        back_populates="schema",
        cascade="save-update, merge",
        passive_deletes=True,
        order_by="LabelSchemaClass.class_id",
    )


class LabelSchemaClass(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "label_schema_classes"
    __table_args__ = (
        UniqueConstraint("schema_id", "class_id", name="uq_label_class_schema_class_id"),
        UniqueConstraint("schema_id", "semantic_key", name="uq_label_class_schema_semantic"),
        CheckConstraint("class_id >= 0", name="label_class_id_nonnegative"),
    )

    schema_id: Mapped[UUID] = mapped_column(
        ForeignKey("label_schemas.id", ondelete="RESTRICT"), nullable=False
    )
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    semantic_key: Mapped[str] = mapped_column(
        String(255), nullable=False, info={"immutable": True}
    )
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    definition_sealed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    schema: Mapped[LabelSchema] = relationship(back_populates="classes")


install_immutable_guard(
    LabelSchema,
    {"name", "description", "parent_schema_id"},
    seal_field="definition_sealed",
)


@event.listens_for(LabelSchemaClass, "before_delete")
def _reject_class_delete_from_sealed_schema(
    mapper: object, connection: object, target: LabelSchemaClass
) -> None:
    sealed = target.definition_sealed
    if target.schema_id is not None:
        sealed = sealed or bool(
            connection.execute(
                select(LabelSchema.definition_sealed).where(
                    LabelSchema.id == target.schema_id
                )
            ).scalar_one_or_none()
        )
        sealed = sealed or _class_schema_is_referenced(connection, target)
    if sealed:
        raise ImmutableFieldError(
            "LabelSchema is sealed; sealed or referenced label classes cannot be deleted"
        )
install_immutable_guard(
    LabelSchemaClass,
    {"schema_id", "class_id", "semantic_key", "display_name", "description"},
    seal_check=lambda target: bool(
        target.definition_sealed
        or (target.schema is not None and target.schema.definition_sealed)
    ),
)


def _schema_is_referenced(connection: object, schema_id: UUID) -> bool:
    from .dataset import DatasetSnapshot
    from .model_node import ModelNode
    tables_and_columns = [
        (DatasetSnapshot, DatasetSnapshot.label_schema_id),
        (ModelNode, ModelNode.label_schema_id),
    ]
    for table_model, column in tables_and_columns:
        if connection.execute(
            select(table_model.id).where(column == schema_id)
        ).scalar_one_or_none() is not None:
            return True
    return False


def _class_schema_is_referenced(connection: object, class_target: LabelSchemaClass) -> bool:
    if class_target.schema_id is not None:
        return _schema_is_referenced(connection, class_target.schema_id)
    return False


@event.listens_for(Session, "before_flush")
def _reject_referenced_label_changes(
    session: Session, flush_context: object, instances: object
) -> None:
    for target in session.dirty:
        if isinstance(target, (LabelSchema, LabelSchemaClass)):
            schema_id = (
                target.id
                if isinstance(target, LabelSchema)
                else target.schema_id
            )
            if schema_id is not None and _schema_is_referenced(session, schema_id):
                if isinstance(target, LabelSchema):
                    raise ImmutableFieldError(
                        "LabelSchema is referenced; schema definitions cannot be modified"
                    )
                else:
                    raise ImmutableFieldError(
                        "LabelSchema is referenced; class definitions cannot be modified"
                    )
    for target in session.deleted:
        if isinstance(target, (LabelSchema, LabelSchemaClass)):
            schema_id = (
                target.id
                if isinstance(target, LabelSchema)
                else target.schema_id
            )
            if schema_id is not None and _schema_is_referenced(session, schema_id):
                if isinstance(target, LabelSchema):
                    raise ImmutableFieldError(
                        "LabelSchema is referenced; schema cannot be deleted"
                    )
                else:
                    raise ImmutableFieldError(
                        "LabelSchema is referenced; class cannot be deleted"
                    )


@event.listens_for(LabelSchemaClass, "before_insert")
def _reject_class_insert_into_sealed_schema(
    mapper: object, connection: object, target: LabelSchemaClass
) -> None:
    if target.definition_sealed:
        raise ImmutableFieldError(
            "LabelSchema is sealed; create a child schema before adding classes"
        )
    if target.schema_id is None:
        return
    sealed = connection.execute(
        select(LabelSchema.definition_sealed).where(LabelSchema.id == target.schema_id)
    ).scalar_one_or_none()
    if sealed:
        raise ImmutableFieldError(
            "LabelSchema is sealed; create a child schema before adding classes"
        )
