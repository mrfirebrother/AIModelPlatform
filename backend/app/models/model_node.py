from __future__ import annotations

from datetime import datetime
from typing import Any, List
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, UUIDPrimaryKeyMixin, install_immutable_guard, json_column


class ModelNode(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "model_nodes"
    __table_args__ = (
        CheckConstraint(
            "status IN ('candidate', 'approved', 'rejected', 'archived')",
            name="model_node_status",
        ),
        UniqueConstraint("training_attempt_id", name="uq_model_node_training_attempt"),
        UniqueConstraint(
            "id",
            "training_attempt_id",
            name="uq_model_node_id_training_attempt",
        ),
        UniqueConstraint("code", name="uq_model_node_code"),
    )

    code: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    parent_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("model_nodes.id", ondelete="RESTRICT")
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    label_schema_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("label_schemas.id", ondelete="RESTRICT")
    )
    dataset_snapshot_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("dataset_snapshots.id", ondelete="RESTRICT")
    )
    training_attempt_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("training_attempts.id", ondelete="RESTRICT")
    )
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    model_family: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    artifact_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    framework: Mapped[str] = mapped_column(String(64), nullable=False, default="pytorch")
    artifact_format: Mapped[str] = mapped_column(String(32), nullable=False, default="pt")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="candidate")
    metadata_json: Mapped[dict[str, Any]] = json_column()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    parent: Mapped[ModelNode | None] = relationship(
        remote_side="ModelNode.id", back_populates="children"
    )
    children: Mapped[List[ModelNode]] = relationship(
        back_populates="parent", passive_deletes=True
    )
    label_schema: Mapped["LabelSchema | None"] = relationship("LabelSchema")
    dataset_snapshot: Mapped["DatasetSnapshot | None"] = relationship("DatasetSnapshot")
    training_attempt: Mapped["TrainingAttempt | None"] = relationship(
        foreign_keys=[training_attempt_id],
        back_populates="output_model_node",
    )


install_immutable_guard(
    ModelNode,
    {
        "code",
        "parent_id",
        "label_schema_id",
        "dataset_snapshot_id",
        "training_attempt_id",
        "artifact_path",
        "artifact_hash",
        "framework",
        "artifact_format",
        "metadata_json",
        "task_type",
        "model_family",
    },
)
