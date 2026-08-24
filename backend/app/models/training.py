from __future__ import annotations

from datetime import datetime
from typing import Any, List
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    Index,
    text,
    func,
)
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UUIDPrimaryKeyMixin, install_immutable_guard, json_column


class TrainingTask(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "training_tasks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="training_task_status",
        ),
    )

    parent_model_node_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("model_nodes.id", ondelete="RESTRICT")
    )
    dataset_snapshot_id: Mapped[UUID] = mapped_column(
        ForeignKey("dataset_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    target_binding_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("model_bindings.id", ondelete="RESTRICT")
    )
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    model_family: Mapped[str] = mapped_column(String(128), nullable=False)
    training_config_json: Mapped[dict[str, Any]] = json_column()
    resource_config_json: Mapped[dict[str, Any]] = json_column()
    evaluation_policy_json: Mapped[dict[str, Any]] = json_column()
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    cancellation_requested: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=False, server_default=sa.text("0")
    )
    created_by: Mapped[str | None] = mapped_column(String(255))
    failure_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    parent_model_node: Mapped["ModelNode | None"] = relationship(
        foreign_keys=[parent_model_node_id]
    )
    dataset_snapshot: Mapped["DatasetSnapshot"] = relationship("DatasetSnapshot")
    target_binding: Mapped["ModelBinding | None"] = relationship("ModelBinding")
    attempts: Mapped[List[TrainingAttempt]] = relationship(
        back_populates="task", passive_deletes=True
    )


class TrainingAttempt(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "training_attempts"
    __table_args__ = (
        UniqueConstraint("task_id", "attempt_no", name="uq_training_attempt_task_no"),
        CheckConstraint(
            "status IN ('pending', 'running', 'recovering', 'completed', 'failed', 'cancelled')",
            name="training_attempt_status",
        ),
        CheckConstraint(
            "attempt_no >= 0 AND current_epoch >= 0 AND retry_count >= 0",
            name="training_attempt_counters_nonnegative",
        ),
        ForeignKeyConstraint(
            ["id", "latest_checkpoint_id"],
            ["checkpoints.attempt_id", "checkpoints.id"],
            name="fk_training_attempt_latest_checkpoint_same_attempt",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["id", "best_checkpoint_id"],
            ["checkpoints.attempt_id", "checkpoints.id"],
            name="fk_training_attempt_best_checkpoint_same_attempt",
            ondelete="RESTRICT",
        ),
        Index(
            "uq_training_attempt_active_task",
            "task_id",
            unique=True,
            postgresql_where=text("status IN ('running', 'recovering')"),
            sqlite_where=text("status IN ('running', 'recovering')"),
        ),
    )

    task_id: Mapped[UUID] = mapped_column(
        ForeignKey("training_tasks.id", ondelete="RESTRICT"), nullable=False
    )
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    gpu_device: Mapped[str | None] = mapped_column(String(64))
    current_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    log_path: Mapped[str | None] = mapped_column(String(1024))
    latest_checkpoint_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    best_checkpoint_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    lease_token: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    fencing_token: Mapped[int | None] = mapped_column(BigInteger)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    task: Mapped[TrainingTask] = relationship(back_populates="attempts")
    output_model_node: Mapped["ModelNode | None"] = relationship(
        "ModelNode",
        back_populates="training_attempt",
        foreign_keys="ModelNode.training_attempt_id",
        primaryjoin="TrainingAttempt.id == ModelNode.training_attempt_id",
        uselist=False,
        viewonly=True,
    )
    checkpoints: Mapped[List[Checkpoint]] = relationship(
        back_populates="attempt", foreign_keys="Checkpoint.attempt_id"
    )


install_immutable_guard(TrainingAttempt, {"task_id", "attempt_no"})


install_immutable_guard(
    TrainingTask,
    {
        "parent_model_node_id",
        "dataset_snapshot_id",
        "task_type",
        "model_family",
        "training_config_json",
        "resource_config_json",
        "evaluation_policy_json",
    },
)


class Checkpoint(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "checkpoints"
    __table_args__ = (
        UniqueConstraint("attempt_id", "id", name="uq_checkpoint_attempt_id_id"),
        CheckConstraint("epoch >= 0", name="checkpoint_epoch_nonnegative"),
    )

    attempt_id: Mapped[UUID] = mapped_column(
        ForeignKey("training_attempts.id", ondelete="RESTRICT"), nullable=False
    )
    parent_artifact_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_snapshot_id: Mapped[UUID] = mapped_column(
        ForeignKey("dataset_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    training_config_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    artifact_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    metrics_json: Mapped[dict[str, Any]] = json_column()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    attempt: Mapped[TrainingAttempt] = relationship(
        back_populates="checkpoints", foreign_keys=[attempt_id]
    )
    dataset_snapshot: Mapped["DatasetSnapshot"] = relationship("DatasetSnapshot")


install_immutable_guard(
    Checkpoint,
    {
        "attempt_id",
        "dataset_snapshot_id",
        "parent_artifact_hash",
        "training_config_hash",
        "epoch",
        "artifact_path",
        "artifact_hash",
        "metrics_json",
    },
)
