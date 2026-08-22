from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import BinaryIO


def compute_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def compute_bytes_hash(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return f"sha256:{h.hexdigest()}"


class ContentAddressedStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _hash_to_path(self, content_hash: str, extension: str = "") -> Path:
        algorithm, _, digest = content_hash.partition(":")
        if not digest:
            digest = algorithm
            algorithm = "sha256"
        prefix = digest[:2]
        suffix = extension if extension.startswith(".") else ""
        return self.root / algorithm / prefix / f"{digest}{suffix}"

    def store_content(self, data: bytes, extension: str = "") -> Path:
        content_hash = compute_bytes_hash(data)
        target = self._hash_to_path(content_hash, extension)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return target

    def store_file(self, source: Path, extension: str = "") -> Path:
        content_hash = compute_file_hash(source)
        target = self._hash_to_path(content_hash, extension)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return target

    def resolve(self, content_hash: str, extension: str = "") -> Path | None:
        path = self._hash_to_path(content_hash, extension)
        return path if path.exists() else None

    def exists(self, content_hash: str, extension: str = "") -> bool:
        return self._hash_to_path(content_hash, extension).exists()
