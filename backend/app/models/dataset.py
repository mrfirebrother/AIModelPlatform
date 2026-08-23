from __future__ import annotations

from datetime import datetime
from typing import Any, List
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin, UUIDPrimaryKeyMixin, install_immutable_guard, json_column


class Dataset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    source_path: Mapped[str | None] = mapped_column(String(1024))
    metadata_json: Mapped[dict[str, Any]] = json_column()

    snapshots: Mapped[List[DatasetSnapshot]] = relationship(
        back_populates="dataset", passive_deletes=True
    )


class DatasetSnapshot(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "dataset_snapshots"
    __table_args__ = (
        CheckConstraint(
            "train_positive_count >= 0 AND val_positive_count >= 0 AND test_positive_count >= 0",
            name="dataset_snapshot_positive_counts",
        ),
        CheckConstraint(
            "train_negative_count >= 0 AND val_negative_count >= 0 AND test_negative_count >= 0",
            name="dataset_snapshot_negative_counts",
        ),
    )

    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="RESTRICT"), nullable=False
    )
    label_schema_id: Mapped[UUID] = mapped_column(
        ForeignKey("label_schemas.id", ondelete="RESTRICT"), nullable=False
    )
    train_manifest_json: Mapped[list[dict[str, Any]]] = json_column(list)
    val_manifest_json: Mapped[list[dict[str, Any]]] = json_column(list)
    test_manifest_json: Mapped[list[dict[str, Any]]] = json_column(list)
    manifest_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    train_positive_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    val_positive_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    test_positive_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    train_negative_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    val_negative_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    test_negative_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quality_json: Mapped[dict[str, Any]] = json_column()
    source_path: Mapped[str | None] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    dataset: Mapped[Dataset] = relationship(back_populates="snapshots")
    label_schema: Mapped["LabelSchema"] = relationship("LabelSchema")


install_immutable_guard(
    DatasetSnapshot,
    {
        "dataset_id",
        "label_schema_id",
        "train_manifest_json",
        "val_manifest_json",
        "test_manifest_json",
        "manifest_path",
        "manifest_hash",
        "train_positive_count",
        "val_positive_count",
        "test_positive_count",
        "train_negative_count",
        "val_negative_count",
        "test_negative_count",
        "quality_json",
        "source_path",
    },
)
