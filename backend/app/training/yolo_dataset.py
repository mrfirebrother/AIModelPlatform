from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class DatasetPreparationCancelled(Exception):
    """Raised when a training task is cancelled while copying its snapshot."""


def manifest_source_dir(manifest_path: Path) -> str:
    """Return the ``source_dir`` a snapshot manifest roots its relative paths at."""
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001 - a broken manifest must not abort the caller
        return ""
    value = data.get("source_dir")
    return str(value) if value else ""


def with_absolute_paths(
    entries: list[dict[str, Any]], source_dir: str
) -> list[dict[str, Any]]:
    """Resolve manifest entries against *source_dir*.

    Snapshot manifests store ``image``/``label`` relative to ``source_dir`` while consumers
    read ``*_stored_path`` first, so filling those in is what makes the entries usable.
    (``train_worker`` carries an equivalent private copy of this logic.)
    """
    if not source_dir:
        return entries
    base = Path(source_dir)
    resolved: list[dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        for rel_key, stored_key in (
            ("image", "image_stored_path"),
            ("label", "label_stored_path"),
        ):
            relative = item.get(rel_key)
            if not item.get(stored_key) and relative and not Path(relative).is_absolute():
                item[stored_key] = str(base / relative)
        resolved.append(item)
    return resolved


@dataclass
class YoloDataset:
    root: Path

    @classmethod
    def from_manifest(
        cls,
        manifest: dict[str, Any],
        output_dir: Path,
        should_cancel=None,
    ) -> YoloDataset:
        output_dir.mkdir(parents=True, exist_ok=True)

        for split in ("train", "val", "test"):
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Path("") is Path(".") - truthy - which would resolve relative entries against
        # the process CWD and fail with a confusing relative FileNotFoundError.
        raw_source_dir = manifest.get("source_dir") or ""
        source_dir = Path(raw_source_dir) if raw_source_dir else None

        for split in ("train", "val", "test"):
            key = f"{split}_files"
            files = manifest.get(key, [])
            for entry in files:
                if should_cancel is not None and should_cancel():
                    raise DatasetPreparationCancelled()
                img_src = entry.get("image_stored_path")
                lbl_src = entry.get("label_stored_path")
                img_rel = entry.get("image", "")
                lbl_rel = entry.get("label", "")

                # Fall back to source_dir when stored paths are not available
                if not img_src and img_rel and source_dir:
                    img_src = str(source_dir / img_rel)
                if not lbl_src and lbl_rel and source_dir:
                    lbl_src = str(source_dir / lbl_rel)

                if img_src and img_rel:
                    dst = output_dir / "images" / split / Path(img_rel).name
                    shutil.copy2(img_src, dst)

                if lbl_src and lbl_rel:
                    dst = output_dir / "labels" / split / Path(lbl_rel).name
                    shutil.copy2(lbl_src, dst)

        nc = manifest.get("nc", 0)
        names = manifest.get("names", [])

        data_yaml = {
            "path": str(output_dir),
            "train": str((output_dir / "images" / "train").as_posix()),
            "val": str((output_dir / "images" / "val").as_posix()),
            "nc": nc,
            "names": names,
        }

        yaml_path = output_dir / "data.yaml"
        yaml_path.write_text(yaml.dump(data_yaml, default_flow_style=False), encoding="utf-8")

        return cls(root=output_dir)

    @staticmethod
    def resolve_device(requested: str) -> str:
        if requested == "cpu":
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return requested
        except ImportError:
            pass
        return "cpu"
