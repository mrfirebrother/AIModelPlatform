from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DbBackupService:
    """PostgreSQL dump/restore with incremental backup and LSN tracking."""

    def __init__(
        self,
        backup_dir: Path | None = None,
        pg_dump_path: str = "pg_dump",
        pg_restore_path: str = "pg_restore",
    ) -> None:
        self._backup_dir = backup_dir or Path("/data/db_backups")
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._pg_dump = pg_dump_path
        self._pg_restore = pg_restore_path

    def dump(
        self,
        db_url: str,
        output_path: Path,
        *,
        backup_type: str = "full",
        base_lsn: str | None = None,
    ) -> dict[str, Any]:
        if backup_type == "incremental" and not base_lsn:
            raise ValueError("base_lsn is required for incremental backups")

        started_at = datetime.now(timezone.utc)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [self._pg_dump, "--no-owner", "--no-privileges"]
        if backup_type == "incremental" and base_lsn:
            cmd.extend(["--start-pos", base_lsn])
        cmd.extend(["--file", str(output_path), db_url])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        completed_at = datetime.now(timezone.utc)

        if result.returncode != 0:
            raise RuntimeError(f"pg_dump failed (rc={result.returncode}): {result.stderr}")

        lsn = self._extract_lsn_from_output(result.stderr)
        timeline = self._extract_timeline_from_output(result.stderr)

        meta: dict[str, Any] = {
            "backup_type": backup_type,
            "success": True,
            "lsn": lsn,
            "timeline": timeline,
            "base_lsn": base_lsn,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "dump_path": str(output_path),
        }
        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def restore(
        self,
        dump_path: Path,
        db_url: str,
    ) -> dict[str, Any]:
        if not dump_path.exists():
            raise FileNotFoundError(f"Dump file not found: {dump_path}")

        meta_path = dump_path.with_suffix(dump_path.suffix + ".meta.json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        cmd = [self._pg_restore, "--no-owner", "--no-privileges", "--clean", "--if-exists"]
        cmd.extend(["--dbname", db_url, str(dump_path)])

        started_at = datetime.now(timezone.utc)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        completed_at = datetime.now(timezone.utc)

        if result.returncode != 0:
            raise RuntimeError(f"pg_restore failed (rc={result.returncode}): {result.stderr}")

        return {
            "success": True,
            "restored_from": str(dump_path),
            "backup_type": meta.get("backup_type", "unknown"),
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
        }

    def get_status(self, dump_path: Path) -> dict[str, Any]:
        meta_path = dump_path.with_suffix(dump_path.suffix + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    @staticmethod
    def _extract_lsn_from_output(stderr: str) -> str:
        for line in stderr.splitlines():
            if "LSN" in line or "lsn" in line:
                parts = line.split()
                for part in parts:
                    if "/" in part and len(part) > 3:
                        return part.strip(",.")
        return "0/0"

    @staticmethod
    def _extract_timeline_from_output(stderr: str) -> int:
        for line in stderr.splitlines():
            if "timeline" in line.lower():
                parts = line.split()
                for part in parts:
                    try:
                        return int(part.strip(",."))
                    except ValueError:
                        continue
        return 1
