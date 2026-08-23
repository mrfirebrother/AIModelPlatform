from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, UUIDPrimaryKeyMixin, json_column


class BackupRecord(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "backup_records"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    backup_type: Mapped[str] = mapped_column(String(32), nullable=False, default="full")
    scope_json: Mapped[dict[str, Any]] = json_column()
    artifact_path: Mapped[str | None] = mapped_column(String(1024))
    artifact_hash: Mapped[str | None] = mapped_column(String(255))
    metadata_json: Mapped[dict[str, Any]] = json_column()
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    verified: Mapped[bool] = mapped_column(default=False)
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    verify_hash: Mapped[str | None] = mapped_column(String(255))
    restore_target_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("backup_records.id", ondelete="SET NULL"), nullable=True
    )
    error_summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
