from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    UniqueConstraint,
    Uuid,
    Index,
    text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UUIDPrimaryKeyMixin, install_immutable_guard


class RuntimeInstance(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "runtime_instances"
    __table_args__ = (
        UniqueConstraint("binding_id", "id", name="uq_runtime_binding_id_id"),
        ForeignKeyConstraint(
            ["binding_id", "release_id", "model_node_id"],
            [
                "model_binding_releases.binding_id",
                "model_binding_releases.id",
                "model_binding_releases.model_node_id",
            ],
            name="fk_runtime_binding_release",
        ),
        CheckConstraint(
            "status IN ('loading', 'preparing', 'ready', 'serving', 'draining', 'stopped', 'failed')",
            name="runtime_status",
        ),
        CheckConstraint(
            "generation >= 0 AND reserved_memory_mb >= 0 "
            "AND (actual_memory_mb IS NULL OR actual_memory_mb >= 0)",
            name="runtime_memory_generation_nonnegative",
        ),
        Index(
            "uq_runtime_serving_binding",
            "binding_id",
            unique=True,
            postgresql_where=text("status = 'serving'"),
            sqlite_where=text("status = 'serving'"),
        ),
        Index(
            "uq_runtime_candidate_binding",
            "binding_id",
            unique=True,
            postgresql_where=text("status IN ('loading', 'preparing', 'ready')"),
            sqlite_where=text("status IN ('loading', 'preparing', 'ready')"),
        ),
    )

    binding_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_bindings.id", ondelete="RESTRICT"), nullable=False
    )
    release_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    model_node_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_nodes.id", ondelete="RESTRICT"), nullable=False
    )
    config_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    fencing_token: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="loading")
    gpu_device: Mapped[str] = mapped_column(String(64), nullable=False)
    reserved_memory_mb: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    actual_memory_mb: Mapped[int | None] = mapped_column(Integer)
    worker_id: Mapped[str | None] = mapped_column(String(255))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_reason: Mapped[str | None] = mapped_column(String(2048))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    binding: Mapped["ModelBinding"] = relationship(
        back_populates="runtime_instances", foreign_keys=[binding_id]
    )
    release: Mapped["BindingRelease"] = relationship(
        "BindingRelease", overlaps="binding,runtime_instances"
    )
    model_node: Mapped["ModelNode"] = relationship(
        "ModelNode", overlaps="release"
    )


install_immutable_guard(
    RuntimeInstance,
    {
        "binding_id",
        "release_id",
        "model_node_id",
        "config_hash",
        "generation",
        "fencing_token",
        "gpu_device",
        "reserved_memory_mb",
    },
)
