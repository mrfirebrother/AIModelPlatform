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
    Uuid,
    Index,
    text,
    func,
    event,
    inspect,
    select,
    true,
)
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from .base import (
    Base,
    ImmutableFieldError,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    install_immutable_guard,
)


class GPUResource(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "gpu_resources"

    gpu_device: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    capacity_memory_mb: Mapped[int] = mapped_column(Integer, nullable=False)
    reserved_memory_mb: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    __table_args__ = (
        CheckConstraint(
            "capacity_memory_mb >= 0 AND reserved_memory_mb >= 0 "
            "AND reserved_memory_mb <= capacity_memory_mb",
            name="gpu_resource_memory_nonnegative",
        ),
    )
    leases: Mapped[list[GPUResourceLease]] = relationship(
        back_populates="resource", passive_deletes=True
    )


class GPUResourceLease(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "gpu_resource_leases"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'released', 'expired', 'failed')",
            name="gpu_lease_status",
        ),
        CheckConstraint(
            "reserved_memory_mb >= 0 AND (actual_memory_mb IS NULL OR actual_memory_mb >= 0)",
            name="gpu_lease_memory_nonnegative",
        ),
        Index(
            "uq_gpu_lease_owner_active",
            "owner_type",
            "owner_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),
    )

    gpu_device: Mapped[str] = mapped_column(String(64), nullable=False)
    gpu_resource_id: Mapped[UUID] = mapped_column(
        ForeignKey("gpu_resources.id", ondelete="RESTRICT"), nullable=False
    )
    owner_type: Mapped[str] = mapped_column(String(64), nullable=False)
    owner_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    reserved_memory_mb: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_memory_mb: Mapped[int | None] = mapped_column(Integer)
    lease_token: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    fencing_token: Mapped[int] = mapped_column(BigInteger, nullable=False)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    resource: Mapped[GPUResource] = relationship(back_populates="leases")


install_immutable_guard(GPUResourceLease, {"gpu_resource_id", "gpu_device"})


@event.listens_for(Session, "before_flush")
def _enforce_gpu_capacity(
    session: Session, flush_context: object, instances: object
) -> None:
    leases = [
        target
        for target in session.new.union(session.dirty)
        if isinstance(target, GPUResourceLease)
    ]
    resource_objects: dict[object, GPUResource] = {}

    def add_resource(resource: GPUResource | None) -> None:
        if resource is not None:
            resource_objects[resource.id or id(resource)] = resource

    for lease in leases:
        resource = lease.resource
        if resource is not None:
            add_resource(resource)
        elif lease.gpu_resource_id is not None:
            with session.no_autoflush:
                resource = session.get(
                    GPUResource, lease.gpu_resource_id, with_for_update=True
                )
            add_resource(resource)

        resource_history = inspect(lease).attrs.gpu_resource_id.history
        for old_resource_id in resource_history.deleted:
            with session.no_autoflush:
                add_resource(session.get(GPUResource, old_resource_id))

        if resource is not None and lease.gpu_device != resource.gpu_device:
            raise ImmutableFieldError(
                "GPU lease gpu_device must match its resource and cannot migrate"
            )
        if resource_history.deleted and (lease.status or "active") == "active":
            raise ImmutableFieldError("active GPU lease cannot migrate resources")

    for target in session.new.union(session.dirty):
        if isinstance(target, GPUResource):
            add_resource(target)

    for resource_id, resource in resource_objects.items():
        for lease in leases:
            if (
                lease.resource is resource
                and lease.gpu_resource_id is None
                and resource.id is not None
            ):
                lease.gpu_resource_id = resource.id
        changed_ids = {
            lease.id
            for lease in leases
            if lease.id is not None
            and (lease.gpu_resource_id or (lease.resource.id if lease.resource else None))
            == resource.id
        }
        with session.no_autoflush:
            locked_resource = session.scalar(
                select(GPUResource)
                .where(GPUResource.id == resource_id)
                .with_for_update()
            )
            if locked_resource is not None:
                resource = locked_resource
            persisted_total = session.scalar(
                select(func.coalesce(func.sum(GPUResourceLease.reserved_memory_mb), 0))
                .where(
                    GPUResourceLease.gpu_resource_id == resource.id,
                    GPUResourceLease.status == "active",
                    ~GPUResourceLease.id.in_(changed_ids) if changed_ids else true(),
                )
            )
        pending_total = sum(
            lease.reserved_memory_mb
            for lease in leases
            if (lease.gpu_resource_id or (lease.resource.id if lease.resource else None))
            == resource.id
            and (lease.status or "active") == "active"
        )
        total = int(persisted_total or 0) + pending_total
        if total > resource.capacity_memory_mb:
            raise ImmutableFieldError(
                f"GPU {resource.gpu_device} reservation exceeds capacity"
            )
        resource.reserved_memory_mb = total


class ModelResidencyPlan(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "model_residency_plans"
    __table_args__ = (
        CheckConstraint("desired_state IN ('resident', 'removed')", name="residency_desired_state"),
        CheckConstraint(
            "reserved_memory_mb >= 0", name="residency_reserved_memory_nonnegative"
        ),
        ForeignKeyConstraint(
            ["binding_id", "release_id", "model_node_id"],
            [
                "model_binding_releases.binding_id",
                "model_binding_releases.id",
                "model_binding_releases.model_node_id",
            ],
            name="fk_residency_binding_release",
        ),
    )

    binding_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_bindings.id", ondelete="RESTRICT"), nullable=False
    )
    release_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    model_node_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_nodes.id", ondelete="RESTRICT"), nullable=False
    )
    gpu_device: Mapped[str] = mapped_column(String(64), nullable=False)
    reserved_memory_mb: Mapped[int] = mapped_column(Integer, nullable=False)
    desired_state: Mapped[str] = mapped_column(String(32), nullable=False, default="resident")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_by: Mapped[str | None] = mapped_column(String(255))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    binding: Mapped["ModelBinding"] = relationship("ModelBinding")
    release: Mapped["BindingRelease"] = relationship(
        "BindingRelease", overlaps="binding"
    )
    model_node: Mapped["ModelNode"] = relationship(
        "ModelNode", overlaps="release"
    )
