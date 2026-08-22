from __future__ import annotations

from datetime import datetime
from typing import Any, List
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UUIDPrimaryKeyMixin, install_immutable_guard, json_column


class ModelBinding(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "model_bindings"
    __table_args__ = (
        ForeignKeyConstraint(
            ["id", "current_release_id"],
            ["model_binding_releases.binding_id", "model_binding_releases.id"],
            name="fk_binding_current_release",
        ),
        ForeignKeyConstraint(
            ["id", "current_runtime_instance_id"],
            ["runtime_instances.binding_id", "runtime_instances.id"],
            name="fk_binding_current_runtime",
        ),
        CheckConstraint("status IN ('unbound', 'bound', 'archived')", name="binding_status"),
    )

    external_ref: Mapped[str | None] = mapped_column(String(255))
    current_release_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    current_runtime_instance_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="unbound")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    releases: Mapped[List[BindingRelease]] = relationship(
        back_populates="binding", foreign_keys="BindingRelease.binding_id"
    )
    runtime_instances: Mapped[List["RuntimeInstance"]] = relationship(
        back_populates="binding", foreign_keys="RuntimeInstance.binding_id"
    )
    current_release: Mapped[BindingRelease | None] = relationship(
        foreign_keys=[current_release_id],
        primaryjoin="and_(ModelBinding.id == BindingRelease.binding_id, ModelBinding.current_release_id == BindingRelease.id)",
        post_update=True,
    )
    current_runtime_instance: Mapped["RuntimeInstance | None"] = relationship(
        foreign_keys=[current_runtime_instance_id],
        primaryjoin="and_(ModelBinding.id == RuntimeInstance.binding_id, ModelBinding.current_runtime_instance_id == RuntimeInstance.id)",
        post_update=True,
    )


class BindingRelease(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "model_binding_releases"
    __table_args__ = (
        UniqueConstraint("binding_id", "id", name="uq_binding_release_binding_id_id"),
        UniqueConstraint(
            "binding_id",
            "id",
            "model_node_id",
            name="uq_binding_release_binding_id_model_node_id",
        ),
        UniqueConstraint("binding_id", "revision_no", name="uq_binding_release_revision"),
        CheckConstraint("revision_no > 0", name="release_revision_positive"),
        ForeignKeyConstraint(
            ["binding_id", "rollback_target_release_id"],
            ["model_binding_releases.binding_id", "model_binding_releases.id"],
            name="fk_release_rollback_target",
        ),
        CheckConstraint("release_type IN ('normal', 'rollback')", name="release_type"),
        CheckConstraint(
            "status IN ('pending', 'preparing', 'active', 'failed', 'superseded')",
            name="release_status",
        ),
        CheckConstraint(
            "(release_type = 'rollback' AND rollback_target_release_id IS NOT NULL) OR "
            "(release_type = 'normal' AND rollback_target_release_id IS NULL)",
            name="release_rollback_target_required",
        ),
    )

    binding_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_bindings.id", ondelete="RESTRICT"), nullable=False
    )
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False)
    model_node_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_nodes.id", ondelete="RESTRICT"), nullable=False
    )
    inference_config_json: Mapped[dict[str, Any]] = json_column()
    inference_config_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    release_type: Mapped[str] = mapped_column(String(32), nullable=False, default="normal")
    rollback_target_release_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    binding: Mapped[ModelBinding] = relationship(
        back_populates="releases", foreign_keys=[binding_id]
    )
    model_node: Mapped["ModelNode"] = relationship("ModelNode")
    rollback_target: Mapped["BindingRelease | None"] = relationship(
        "BindingRelease",
        primaryjoin="foreign(BindingRelease.rollback_target_release_id) == remote(BindingRelease.id)",
        foreign_keys=[rollback_target_release_id],
        remote_side="BindingRelease.id",
        viewonly=True,
    )


install_immutable_guard(
    BindingRelease,
    {
        "binding_id",
        "revision_no",
        "model_node_id",
        "rollback_target_release_id",
        "inference_config_json",
        "inference_config_hash",
        "release_type",
        "reason",
    },
)
