from __future__ import annotations

import dataclasses
import time

from backend.app.runtime.gpu_monitor import GpuMonitor


class EvictionError(Exception):
    """Raised when no unlocked model can be evicted."""


@dataclasses.dataclass
class ModelEntry:
    """Tracks a single resident model on GPU."""

    model_name: str
    gpu_device: str
    memory_mb: int
    loaded_at: float
    last_accessed: float
    locked: bool = False


class ModelManager:
    """Manages GPU model loading, unloading, and LRU eviction.

    Integrates with GpuMonitor for memory tracking.  In CPU mode the
    monitor returns simulated telemetry so the full lifecycle can be
    exercised without real hardware.
    """

    def __init__(
        self,
        gpu_monitor: GpuMonitor | None = None,
        *,
        max_models: int = 8,
    ) -> None:
        self._monitor = gpu_monitor or GpuMonitor(cpu_mode=True)
        self._max_models = max_models
        self._models: dict[str, ModelEntry] = {}
        self._next_device = 0

    # ------------------------------------------------------------------
    # load / unload
    # ------------------------------------------------------------------

    def load(self, model_name: str, *, memory_mb: int) -> str:
        """Load a model and return its GPU device key.

        If the model is already resident, refresh its access timestamp and
        return the existing device.  When capacity is exceeded the LRU
        (least-recently-used, unlocked) model is evicted first.
        """
        existing = self._find_by_name(model_name)
        if existing is not None:
            existing.last_accessed = time.time()
            return existing.gpu_device

        self._ensure_capacity(memory_mb)
        device = self._allocate_device()
        now = time.time()
        entry = ModelEntry(
            model_name=model_name,
            gpu_device=device,
            memory_mb=memory_mb,
            loaded_at=now,
            last_accessed=now,
        )
        self._models[device] = entry
        self._monitor.reserve_memory(device_index=0, mb=memory_mb)
        return device

    def unload(self, gpu_device: str) -> bool:
        """Unload the model on *gpu_device*.  Returns True on success."""
        entry = self._models.pop(gpu_device, None)
        if entry is None:
            return False
        self._monitor.release_memory(device_index=0, mb=entry.memory_mb)
        return True

    # ------------------------------------------------------------------
    # eviction
    # ------------------------------------------------------------------

    def evict_lru(self) -> ModelEntry:
        """Evict the least-recently-used unlocked model.

        Raises EvictionError if every resident model is locked.
        """
        candidate = self.get_next_eviction_candidate()
        if candidate is None:
            raise EvictionError("No unlocked model available for eviction")
        self.unload(candidate.gpu_device)
        return candidate

    def get_next_eviction_candidate(self) -> ModelEntry | None:
        """Return the LRU unlocked entry, or *None* if all are locked."""
        unlocked = [e for e in self._models.values() if not e.locked]
        if not unlocked:
            return None
        return min(unlocked, key=lambda e: e.last_accessed)

    def lock(self, gpu_device: str) -> None:
        entry = self._models.get(gpu_device)
        if entry is not None:
            entry.locked = True

    def unlock(self, gpu_device: str) -> None:
        entry = self._models.get(gpu_device)
        if entry is not None:
            entry.locked = False

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def get(self, gpu_device: str) -> ModelEntry | None:
        return self._models.get(gpu_device)

    def list_resident(self) -> list[ModelEntry]:
        return list(self._models.values())

    def get_total_memory_used(self) -> int:
        return sum(e.memory_mb for e in self._models.values())

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _find_by_name(self, name: str) -> ModelEntry | None:
        for entry in self._models.values():
            if entry.model_name == name:
                return entry
        return None

    def _allocate_device(self) -> str:
        device = f"gpu:{self._next_device}"
        self._next_device += 1
        return device

    def _ensure_capacity(self, needed_mb: int) -> None:
        while len(self._models) >= self._max_models:
            self.evict_lru()
