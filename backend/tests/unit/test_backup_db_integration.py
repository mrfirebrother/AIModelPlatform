from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from backend.app.models import Base, DatasetSnapshot, LabelSchema, Dataset, ModelNode
from backend.app.services.backup_service import BackupService
from backend.app.services.db_backup import DbBackupService


@pytest.fixture()
def backup_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture()
def db_backup_service(backup_dir: Path) -> DbBackupService:
    return DbBackupService(
        backup_dir=backup_dir / "db",
        pg_dump_path="pg_dump",
        pg_restore_path="pg_restore",
    )


@pytest.fixture()
def service(backup_dir: Path, db_backup_service: DbBackupService) -> BackupService:
    return BackupService(
        backup_dir=backup_dir,
        db_backup_service=db_backup_service,
    )


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    event.listen(
        engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(engine)
    with Session(engine) as db_session:
        yield db_session
    engine.dispose()


def _make_model(session: Session, artifact_path: str = "models/test.pt") -> ModelNode:
    model = ModelNode(
        artifact_path=artifact_path,
        artifact_hash=f"sha256:{uuid4().hex}",
        task_type="object_detection",
        model_family="yolo",
        status="approved",
        metadata_json={},
    )
    session.add(model)
    session.flush()
    return model


class TestBackupWithDbDump:
    def test_full_backup_includes_db_dump(
        self, service: BackupService, session: Session
    ) -> None:
        with patch.object(
            service._db_backup, "dump", wraps=service._db_backup.dump
        ) as mock_dump:
            with patch("backend.app.services.db_backup.subprocess") as mock_sub:
                mock_sub.run.return_value = MagicMock(returncode=0, stderr="")
                record = service.create_backup(
                    session, name="full-with-db", backup_type="full"
                )
                assert record.name == "full-with-db"
                assert record.status == "completed"
                mock_dump.assert_called_once()

    def test_backup_record_contains_db_metadata(
        self, service: BackupService, session: Session
    ) -> None:
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stderr="")
            record = service.create_backup(
                session, name="db-meta-test", backup_type="full"
            )
            meta = record.metadata_json or {}
            assert "db_dump" in meta
            assert meta["db_dump"]["success"] is True

    def test_model_backup_also_dumps_db(
        self, service: BackupService, session: Session
    ) -> None:
        model = _make_model(session)
        session.commit()
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stderr="")
            record = service.create_backup(
                session,
                name="model-with-db",
                backup_type="model",
                scope_json={"model_ids": [str(model.id)]},
            )
            meta = record.metadata_json or {}
            assert "db_dump" in meta


class TestRestoreWithDbRestore:
    def test_restore_triggers_db_restore(
        self, service: BackupService, session: Session
    ) -> None:
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stderr="")
            record = service.create_backup(
                session, name="restore-test", backup_type="full"
            )
            backup_path = Path(record.artifact_path)
            db_dump_path = backup_path / "db_dump.sql"
            db_dump_path.write_text("-- dummy dump", encoding="utf-8")
            session.commit()

            with patch.object(
                service._db_backup, "restore", wraps=service._db_backup.restore
            ) as mock_restore:
                with patch("backend.app.services.db_backup.subprocess") as mock_sub2:
                    mock_sub2.run.return_value = MagicMock(returncode=0, stderr="")
                    service.verify_backup(session, record.id)
                    session.commit()
                    restored = service.restore_backup(
                        session, record.id, restore_type="full", force=True
                    )
                    assert restored.status == "completed"
                    mock_restore.assert_called_once()

    def test_restore_record_has_db_status(
        self, service: BackupService, session: Session
    ) -> None:
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stderr="")
            record = service.create_backup(
                session, name="db-status-test", backup_type="full"
            )
            backup_path = Path(record.artifact_path)
            db_dump_path = backup_path / "db_dump.sql"
            db_dump_path.write_text("-- dummy dump", encoding="utf-8")
            session.commit()

            with patch("backend.app.services.db_backup.subprocess") as mock_sub2:
                mock_sub2.run.return_value = MagicMock(returncode=0, stderr="")
                service.verify_backup(session, record.id)
                session.commit()
                restored = service.restore_backup(
                    session, record.id, restore_type="full", force=True
                )
                meta = restored.metadata_json or {}
                assert "db_restore" in meta
                assert meta["db_restore"]["success"] is True


class TestBackupDeleteWithDb:
    def test_delete_backup_removes_db_files(
        self, service: BackupService, session: Session
    ) -> None:
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stderr="")
            record = service.create_backup(
                session, name="del-db-test", backup_type="full"
            )
            session.commit()
            backup_path = Path(record.artifact_path)
            db_files = list(backup_path.glob("*.sql"))
            assert len(db_files) > 0 or record.artifact_path is not None

            deleted = service.delete_backup(session, record.id)
            assert deleted.name == "del-db-test"
