from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from backend.app.models import Base, Dataset, DatasetSnapshot, LabelSchema, ModelNode
from backend.app.services.backup_service import BackupService


@pytest.fixture()
def backup_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture()
def service(backup_dir: Path) -> BackupService:
    return BackupService(backup_dir=backup_dir)


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


def _make_snapshot(session: Session) -> DatasetSnapshot:
    schema = LabelSchema(name=f"schema-{uuid4().hex[:8]}")
    session.add(schema)
    session.flush()
    dataset = Dataset(name=f"ds-{uuid4().hex[:8]}")
    session.add(dataset)
    session.flush()
    snap = DatasetSnapshot(
        dataset_id=dataset.id,
        label_schema_id=schema.id,
        manifest_path=f"snapshots/{uuid4()}/manifest.json",
        manifest_hash=f"sha256:{uuid4().hex}",
    )
    session.add(snap)
    session.flush()
    return snap


class TestBackupCreate:
    def test_create_full_backup(self, service: BackupService, session: Session) -> None:
        record = service.create_backup(session, name="full-backup-1")
        assert record.name == "full-backup-1"
        assert record.status == "completed"
        assert record.backup_type == "full"
        assert record.artifact_hash is not None

    def test_create_backup_with_description(self, service: BackupService, session: Session) -> None:
        record = service.create_backup(
            session, name="desc-backup", description="Test description"
        )
        assert record.description == "Test description"

    def test_create_duplicate_backup_name_raises(self, service: BackupService, session: Session) -> None:
        service.create_backup(session, name="dup-test")
        with pytest.raises(ValueError, match="already exists"):
            service.create_backup(session, name="dup-test")

    def test_create_model_backup(self, service: BackupService, session: Session) -> None:
        model = _make_model(session)
        session.commit()
        record = service.create_backup(
            session, name="model-backup", backup_type="model",
            scope_json={"model_ids": [str(model.id)]},
        )
        assert record.backup_type == "model"

    def test_create_dataset_backup(self, service: BackupService, session: Session) -> None:
        snap = _make_snapshot(session)
        session.commit()
        record = service.create_backup(
            session, name="ds-backup", backup_type="dataset",
            scope_json={"dataset_snapshot_ids": [str(snap.id)]},
        )
        assert record.backup_type == "dataset"


class TestBackupVerify:
    def test_verify_backup(self, service: BackupService, session: Session) -> None:
        record = service.create_backup(session, name="verify-test")
        session.commit()
        verified, verify_hash, verified_at = service.verify_backup(session, record.id)
        assert verified is True
        assert verify_hash.startswith("sha256:")
        assert verified_at is not None

    def test_verify_nonexistent_backup_raises(self, service: BackupService, session: Session) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.verify_backup(session, uuid4())


class TestBackupDelete:
    def test_delete_backup(self, service: BackupService, session: Session) -> None:
        record = service.create_backup(session, name="delete-test")
        session.commit()
        deleted = service.delete_backup(session, record.id)
        assert deleted.name == "delete-test"

    def test_delete_nonexistent_raises(self, service: BackupService, session: Session) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.delete_backup(session, uuid4())


class TestBackupList:
    def test_list_backups(self, service: BackupService, session: Session) -> None:
        service.create_backup(session, name="list-1")
        service.create_backup(session, name="list-2")
        session.commit()
        backups = service.list_backups(session)
        assert len(backups) == 2

    def test_list_backups_by_type(self, service: BackupService, session: Session) -> None:
        service.create_backup(session, name="list-full", backup_type="full")
        service.create_backup(session, name="list-model", backup_type="model")
        session.commit()
        full = service.list_backups(session, backup_type="full")
        model = service.list_backups(session, backup_type="model")
        assert len(full) == 1
        assert len(model) == 1


class TestBackupRestore:
    def test_restore_verified_backup(self, service: BackupService, session: Session) -> None:
        record = service.create_backup(session, name="restore-verified")
        service.verify_backup(session, record.id)
        session.commit()
        restored = service.restore_backup(session, record.id, restore_type="full")
        assert restored.status == "completed"
        assert restored.restore_target_id == record.id

    def test_restore_unverified_without_force_raises(self, service: BackupService, session: Session) -> None:
        record = service.create_backup(session, name="restore-unverified")
        session.commit()
        with pytest.raises(ValueError, match="verified"):
            service.restore_backup(session, record.id, force=False)

    def test_restore_nonexistent_raises(self, service: BackupService, session: Session) -> None:
        with pytest.raises(ValueError, match="not found"):
            service.restore_backup(session, uuid4())
