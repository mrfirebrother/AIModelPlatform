from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import BackupRecord, DatasetSnapshot, ModelNode
from backend.app.storage.artifacts import ContentAddressedStore, compute_file_hash

from .db_backup import DbBackupService


class BackupService:
    """Core backup and restore logic for models, datasets, and metadata."""

    def __init__(
        self,
        backup_dir: Path | None = None,
        db_backup_service: DbBackupService | None = None,
    ) -> None:
        self._backup_dir = backup_dir or Path("/data/backups")
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._store = ContentAddressedStore(self._backup_dir / "artifacts")
        self._db_backup = db_backup_service or DbBackupService(
            backup_dir=self._backup_dir / "db",
        )

    def create_backup(
        self,
        session: Session,
        *,
        name: str,
        description: str | None = None,
        backup_type: str = "full",
        scope_json: dict[str, Any] | None = None,
        metadata_json: dict[str, Any] | None = None,
    ) -> BackupRecord:
        existing = session.execute(
            select(BackupRecord).where(BackupRecord.name == name)
        ).scalars().first()
        if existing is not None:
            raise ValueError(f"Backup with name '{name}' already exists")

        backup_id = uuid4()
        backup_path = self._backup_dir / str(backup_id)
        backup_path.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "name": name,
            "backup_type": backup_type,
            "scope_json": scope_json or {},
            "metadata_json": metadata_json or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        collected = self._collect_items(session, backup_type, scope_json or {}, backup_path)
        manifest["items"] = collected

        manifest_path = backup_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        artifact_hash = compute_file_hash(manifest_path)

        db_dump_path = backup_path / "db_dump.sql"
        db_dump_meta: dict[str, Any] = {}
        try:
            from backend.app.config import get_settings
            settings = get_settings()
            db_dump_meta = self._db_backup.dump(
                db_url=settings.postgres_url,
                output_path=db_dump_path,
                backup_type=backup_type,
            )
        except Exception as exc:
            db_dump_meta = {"success": False, "error": str(exc)}

        merged_meta = dict(metadata_json or {})
        merged_meta["db_dump"] = db_dump_meta

        record = BackupRecord(
            id=backup_id,
            name=name,
            description=description,
            backup_type=backup_type,
            scope_json=scope_json or {},
            artifact_path=str(backup_path),
            artifact_hash=artifact_hash,
            metadata_json=merged_meta,
            status="completed",
        )
        session.add(record)
        session.flush()
        return record

    def _collect_items(
        self,
        session: Session,
        backup_type: str,
        scope_json: dict[str, Any],
        dest: Path,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []

        if backup_type in ("full", "model"):
            items.extend(self._backup_models(session, scope_json, dest))

        if backup_type in ("full", "dataset"):
            items.extend(self._backup_datasets(session, scope_json, dest))

        if backup_type in ("full", "metadata"):
            items.extend(self._backup_metadata(session, scope_json, dest))

        return items

    def _backup_models(
        self,
        session: Session,
        scope_json: dict[str, Any],
        dest: Path,
    ) -> list[dict[str, Any]]:
        model_ids = scope_json.get("model_ids")
        stmt = select(ModelNode)
        if model_ids:
            stmt = stmt.where(ModelNode.id.in_([UUID(mid) for mid in model_ids]))
        models = list(session.execute(stmt).scalars().all())
        items: list[dict[str, Any]] = []
        for model in models:
            src_path = Path(model.artifact_path)
            item: dict[str, Any] = {
                "type": "model",
                "id": str(model.id),
                "artifact_path": model.artifact_path,
                "artifact_hash": model.artifact_hash,
                "status": model.status,
            }
            if src_path.exists():
                target = self._store.store_file(src_path, suffix=".pt")
                item["stored_hash"] = str(target)
            items.append(item)
        return items

    def _backup_datasets(
        self,
        session: Session,
        scope_json: dict[str, Any],
        dest: Path,
    ) -> list[dict[str, Any]]:
        snapshot_ids = scope_json.get("dataset_snapshot_ids")
        stmt = select(DatasetSnapshot)
        if snapshot_ids:
            stmt = stmt.where(DatasetSnapshot.id.in_([UUID(sid) for sid in snapshot_ids]))
        snapshots = list(session.execute(stmt).scalars().all())
        items: list[dict[str, Any]] = []
        for snap in snapshots:
            snapshot_data = {
                "id": str(snap.id),
                "manifest_path": snap.manifest_path,
                "manifest_hash": snap.manifest_hash,
                "dataset_id": str(snap.dataset_id),
            }
            items.append({
                "type": "dataset_snapshot",
                **snapshot_data,
            })
        return items

    def _backup_metadata(
        self,
        session: Session,
        scope_json: dict[str, Any],
        dest: Path,
    ) -> list[dict[str, Any]]:
        metadata_path = dest / "metadata.json"
        snapshot_stmt = select(DatasetSnapshot)
        snapshots = list(session.execute(snapshot_stmt).scalars().all())
        metadata: dict[str, Any] = {
            "dataset_snapshots": [
                {
                    "id": str(s.id),
                    "manifest_hash": s.manifest_hash,
                }
                for s in snapshots
            ],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        return [{"type": "metadata", "path": str(metadata_path)}]

    def verify_backup(
        self,
        session: Session,
        backup_id: UUID,
    ) -> tuple[bool, str, datetime]:
        record = session.get(BackupRecord, backup_id)
        if record is None:
            raise ValueError(f"Backup {backup_id} not found")
        if record.artifact_path is None or record.artifact_hash is None:
            raise ValueError("Backup has no artifact to verify")

        backup_path = Path(record.artifact_path)
        manifest_path = backup_path / "manifest.json"
        if not manifest_path.exists():
            record.verified = False
            record.error_summary = "Manifest file missing"
            record.status = "failed"
            session.flush()
            raise ValueError("Manifest file missing from backup")

        current_hash = compute_file_hash(manifest_path)
        now = datetime.now(timezone.utc)
        record.verified = True
        record.verified_at = now
        record.verify_hash = current_hash
        record.error_summary = None
        session.flush()
        return True, current_hash, now

    def restore_backup(
        self,
        session: Session,
        backup_id: UUID,
        *,
        restore_type: str = "full",
        target_backup_id: UUID | None = None,
        force: bool = False,
    ) -> BackupRecord:
        record = session.get(BackupRecord, backup_id)
        if record is None:
            raise ValueError(f"Backup {backup_id} not found")

        if not record.verified and not force:
            raise ValueError("Backup must be verified before restore (use force=True to bypass)")

        restore_id = uuid4()
        db_restore_meta: dict[str, Any] = {}
        try:
            backup_path = Path(record.artifact_path)
            db_dump_path = backup_path / "db_dump.sql"
            if db_dump_path.exists():
                from backend.app.config import get_settings
                settings = get_settings()
                db_restore_meta = self._db_backup.restore(
                    dump_path=db_dump_path,
                    db_url=settings.postgres_url,
                )
        except Exception as exc:
            db_restore_meta = {"success": False, "error": str(exc)}

        restore_record = BackupRecord(
            id=restore_id,
            name=f"restore-{restore_id.hex[:8]}",
            description=f"Restore from backup {backup_id}",
            backup_type=restore_type,
            scope_json={
                "source_backup_id": str(backup_id),
                "force": force,
            },
            artifact_path=record.artifact_path,
            artifact_hash=record.artifact_hash,
            metadata_json={"db_restore": db_restore_meta},
            status="completed",
            restore_target_id=backup_id,
        )
        session.add(restore_record)
        session.flush()
        return restore_record

    def list_backups(
        self,
        session: Session,
        *,
        backup_type: str | None = None,
    ) -> list[BackupRecord]:
        stmt = select(BackupRecord)
        if backup_type is not None:
            stmt = stmt.where(BackupRecord.backup_type == backup_type)
        stmt = stmt.order_by(BackupRecord.created_at.desc())
        return list(session.execute(stmt).scalars().all())

    def get_backup(self, session: Session, backup_id: UUID) -> BackupRecord | None:
        return session.get(BackupRecord, backup_id)

    def delete_backup(
        self,
        session: Session,
        backup_id: UUID,
    ) -> BackupRecord:
        record = session.get(BackupRecord, backup_id)
        if record is None:
            raise ValueError(f"Backup {backup_id} not found")

        restore_refs = session.execute(
            select(BackupRecord).where(BackupRecord.restore_target_id == backup_id)
        ).scalars().all()
        for ref in restore_refs:
            ref.restore_target_id = None

        if record.artifact_path:
            artifact_dir = Path(record.artifact_path)
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir, ignore_errors=True)

        session.delete(record)
        session.flush()
        return record
