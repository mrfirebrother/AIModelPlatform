from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class DatasetPreparationCancelled(Exception):
    """Raised when a training task is cancelled while copying its snapshot."""


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

        source_dir = Path(manifest.get("source_dir", ""))

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
