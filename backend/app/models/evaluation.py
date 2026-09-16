from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UUIDPrimaryKeyMixin, install_immutable_guard, json_column


class Evaluation(UUIDPrimaryKeyMixin, Base):
    """一次模型评测。

    平台没有审核机制：这里**没有**人工复核字段（原 human_status / reviewer / human_conclusion /
    human_comments / reviewed_at 已由迁移 ``0005`` 删除）。``auto_status`` 是评测流水线状态
    （pending → running → passed/failed），供 worker、调度器与 model_diff 使用，不是"通过/未通过"判定。
    """

    __tablename__ = "evaluations"
    __table_args__ = (
        CheckConstraint(
            "auto_status IN ('pending', 'running', 'passed', 'failed')",
            name="evaluation_auto_status",
        ),
        CheckConstraint(
            "attempt_status IN ('pending', 'running', 'passed', 'failed', 'retrying', 'cancelled')",
            name="evaluation_attempt_status",
        ),
        CheckConstraint(
            "attempt_no >= 0 AND retry_count >= 0",
            name="evaluation_counters_nonnegative",
        ),
    )

    model_node_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_nodes.id", ondelete="RESTRICT"), nullable=False
    )
    dataset_snapshot_id: Mapped[UUID] = mapped_column(
        ForeignKey("dataset_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    auto_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    evaluation_policy_json: Mapped[dict[str, Any]] = json_column()
    auto_metrics_json: Mapped[dict[str, Any]] = json_column()
    report_path: Mapped[str | None] = mapped_column(String(1024))
    report_hash: Mapped[str | None] = mapped_column(String(255))
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    lease_token: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    fencing_token: Mapped[int | None] = mapped_column(BigInteger)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    model_node: Mapped["ModelNode"] = relationship("ModelNode")
    dataset_snapshot: Mapped["DatasetSnapshot"] = relationship("DatasetSnapshot")


install_immutable_guard(
    Evaluation,
    {
        "model_node_id",
        "dataset_snapshot_id",
        "evaluation_policy_json",
        "auto_metrics_json",
        "report_path",
        "report_hash",
    },
    allow_initial_null_fields={"report_path", "report_hash"},
    allow_initial_empty_fields={"auto_metrics_json"},
    allow_initial_empty_check=lambda target: target.auto_status in ("pending", "running"),
)
