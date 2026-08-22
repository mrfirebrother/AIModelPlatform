from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckpointInfo:
    checkpoint_path: Path
    epoch: int
    is_best: bool


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, name: str) -> Path:
        return self.checkpoint_dir / f"{name}.meta.json"

    def _write_meta(self, name: str, epoch: int, is_best: bool) -> None:
        meta_path = self._meta_path(name)
        meta_path.write_text(
            json.dumps({"epoch": epoch, "is_best": is_best}),
            encoding="utf-8",
        )

    def _read_meta(self, name: str) -> dict:
        meta_path = self._meta_path(name)
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {"epoch": -1, "is_best": False}

    def save_checkpoint(
        self,
        source_path: Path,
        epoch: int,
        is_best: bool = False,
    ) -> CheckpointInfo:
        if is_best:
            target = self.checkpoint_dir / "best.pt"
            shutil.copy2(source_path, target)
            self._write_meta("best", epoch, is_best)

            latest = self.checkpoint_dir / "latest.pt"
            shutil.copy2(source_path, latest)
            self._write_meta("latest", epoch, is_best)
        else:
            target = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
            shutil.copy2(source_path, target)
            self._write_meta(f"epoch_{epoch:04d}", epoch, is_best)

            latest = self.checkpoint_dir / "latest.pt"
            shutil.copy2(source_path, latest)
            self._write_meta("latest", epoch, False)

        return CheckpointInfo(
            checkpoint_path=target,
            epoch=epoch,
            is_best=is_best,
        )

    def get_best_checkpoint(self) -> CheckpointInfo | None:
        best = self.checkpoint_dir / "best.pt"
        if not best.exists():
            return None
        meta = self._read_meta("best")
        return CheckpointInfo(
            checkpoint_path=best,
            epoch=meta["epoch"],
            is_best=True,
        )

    def get_latest_checkpoint(self) -> CheckpointInfo | None:
        latest = self.checkpoint_dir / "latest.pt"
        if not latest.exists():
            return None
        meta = self._read_meta("latest")
        return CheckpointInfo(
            checkpoint_path=latest,
            epoch=meta["epoch"],
            is_best=meta.get("is_best", False),
        )
