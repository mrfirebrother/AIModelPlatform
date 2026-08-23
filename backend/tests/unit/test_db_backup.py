from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.app.services.db_backup import DbBackupService


@pytest.fixture()
def backup_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture()
def service(backup_dir: Path) -> DbBackupService:
    return DbBackupService(
        backup_dir=backup_dir,
        pg_dump_path="pg_dump",
        pg_restore_path="pg_restore",
    )


class TestDbBackupDump:
    def test_dump_creates_sql_file(self, service: DbBackupService, backup_dir: Path) -> None:
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stderr="")
            result = service.dump(
                db_url="postgresql://user:pass@localhost:5432/testdb",
                output_path=backup_dir / "dump.sql",
            )
            assert result["success"] is True
            assert result["dump_path"] == str(backup_dir / "dump.sql")
            assert result["backup_type"] == "full"
            assert "started_at" in result
            assert "completed_at" in result

    def test_dump_returns_lsn_metadata(self, service: DbBackupService, backup_dir: Path) -> None:
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stderr="")
            result = service.dump(
                db_url="postgresql://user:pass@localhost:5432/testdb",
                output_path=backup_dir / "dump.sql",
            )
            assert "lsn" in result
            assert "timeline" in result

    def test_dump_failure_raises(self, service: DbBackupService, backup_dir: Path) -> None:
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(
                returncode=1, stderr="connection refused"
            )
            with pytest.raises(RuntimeError, match="pg_dump failed"):
                service.dump(
                    db_url="postgresql://user:pass@localhost:5432/testdb",
                    output_path=backup_dir / "dump.sql",
                )

    def test_dump_saves_metadata_json(self, service: DbBackupService, backup_dir: Path) -> None:
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stderr="")
            output = backup_dir / "dump.sql"
            result = service.dump(
                db_url="postgresql://user:pass@localhost:5432/testdb",
                output_path=output,
            )
            meta_path = output.with_suffix(".sql.meta.json")
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            assert meta["backup_type"] == "full"
            assert meta["success"] is True


class TestDbBackupRestore:
    def test_restore_calls_pg_restore(self, service: DbBackupService, backup_dir: Path) -> None:
        dump_path = backup_dir / "dump.sql"
        dump_path.write_text("CREATE TABLE test (id INT);", encoding="utf-8")
        meta = {
            "backup_type": "full",
            "success": True,
            "lsn": "0/1234567",
            "timeline": 1,
        }
        dump_path.with_suffix(".sql.meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stderr="")
            result = service.restore(
                dump_path=dump_path,
                db_url="postgresql://user:pass@localhost:5432/testdb",
            )
            assert result["success"] is True
            assert result["restored_from"] == str(dump_path)

    def test_restore_failure_raises(self, service: DbBackupService, backup_dir: Path) -> None:
        dump_path = backup_dir / "dump.sql"
        dump_path.write_text("CREATE TABLE test (id INT);", encoding="utf-8")
        meta = {"backup_type": "full", "success": True}
        dump_path.with_suffix(".sql.meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(
                returncode=1, stderr="permission denied"
            )
            with pytest.raises(RuntimeError, match="pg_restore failed"):
                service.restore(
                    dump_path=dump_path,
                    db_url="postgresql://user:pass@localhost:5432/testdb",
                )

    def test_restore_missing_file_raises(self, service: DbBackupService) -> None:
        with pytest.raises(FileNotFoundError):
            service.restore(
                dump_path=Path("/nonexistent/dump.sql"),
                db_url="postgresql://user:pass@localhost:5432/testdb",
            )


class TestDbBackupIncremental:
    def test_incremental_uses_base_lsn(self, service: DbBackupService, backup_dir: Path) -> None:
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stderr="")
            result = service.dump(
                db_url="postgresql://user:pass@localhost:5432/testdb",
                output_path=backup_dir / "incr.sql",
                backup_type="incremental",
                base_lsn="0/1234567",
            )
            assert result["backup_type"] == "incremental"
            assert result["base_lsn"] == "0/1234567"

    def test_incremental_without_base_lsn_raises(self, service: DbBackupService, backup_dir: Path) -> None:
        with pytest.raises(ValueError, match="base_lsn is required"):
            service.dump(
                db_url="postgresql://user:pass@localhost:5432/testdb",
                output_path=backup_dir / "incr.sql",
                backup_type="incremental",
            )


class TestDbBackupStatus:
    def test_get_status_returns_metadata(self, service: DbBackupService, backup_dir: Path) -> None:
        dump_path = backup_dir / "test.sql"
        dump_path.write_text("-- dummy", encoding="utf-8")
        meta = {
            "backup_type": "full",
            "success": True,
            "lsn": "0/ABCDEF",
            "timeline": 1,
            "started_at": "2025-01-01T00:00:00+00:00",
            "completed_at": "2025-01-01T00:01:00+00:00",
        }
        dump_path.with_suffix(".sql.meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        status = service.get_status(dump_path)
        assert status["backup_type"] == "full"
        assert status["lsn"] == "0/ABCDEF"

    def test_get_status_missing_metadata_raises(self, service: DbBackupService) -> None:
        with pytest.raises(FileNotFoundError):
            service.get_status(Path("/nonexistent/dump.sql"))


class TestDbBackupConsistency:
    def test_dump_records_wal_lsn(self, service: DbBackupService, backup_dir: Path) -> None:
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stderr="")
            result = service.dump(
                db_url="postgresql://user:pass@localhost:5432/testdb",
                output_path=backup_dir / "dump.sql",
            )
            assert result["lsn"] is not None

    def test_incremental_records_base_and_end_lsn(self, service: DbBackupService, backup_dir: Path) -> None:
        with patch("backend.app.services.db_backup.subprocess") as mock_sub:
            mock_sub.run.return_value = MagicMock(returncode=0, stderr="")
            result = service.dump(
                db_url="postgresql://user:pass@localhost:5432/testdb",
                output_path=backup_dir / "incr.sql",
                backup_type="incremental",
                base_lsn="0/1000000",
            )
            assert result["base_lsn"] == "0/1000000"
            assert "lsn" in result
