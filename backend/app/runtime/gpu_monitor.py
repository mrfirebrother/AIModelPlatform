from __future__ import annotations

import dataclasses
import random


@dataclasses.dataclass(frozen=True, slots=True)
class GpuStats:
    """Snapshot of GPU telemetry for a single device."""

    device_index: int
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilization_percent: float
    temperature_celsius: float


_MOCK_TOTAL_MB = 8192


class GpuMonitor:
    """Queries GPU telemetry; returns simulated data in CPU mode."""

    def __init__(self, *, cpu_mode: bool = True) -> None:
        self._cpu_mode = cpu_mode
        self._mock_used: dict[int, int] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get_device_count(self) -> int:
        if self._cpu_mode:
            return 2
        raise NotImplementedError("Real GPU query not implemented")

    def query_gpu(self, device_index: int) -> GpuStats:
        if self._cpu_mode:
            return self._mock_query(device_index)
        raise NotImplementedError("Real GPU query not implemented")

    def query_all_gpus(self) -> list[GpuStats]:
        return [self.query_gpu(i) for i in range(self.get_device_count())]

    def get_available_memory(self, device_index: int) -> int:
        stats = self.query_gpu(device_index)
        return stats.free_memory_mb

    def has_enough_memory(self, device_index: int, required_mb: int) -> bool:
        return self.get_available_memory(device_index) >= required_mb

    def reserve_memory(self, device_index: int, mb: int) -> None:
        current = self._mock_used.get(device_index, 0)
        self._mock_used[device_index] = current + mb

    def release_memory(self, device_index: int, mb: int) -> None:
        current = self._mock_used.get(device_index, 0)
        self._mock_used[device_index] = max(0, current - mb)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _mock_query(self, device_index: int) -> GpuStats:
        used = self._mock_used.get(device_index, 0)
        total = _MOCK_TOTAL_MB
        free = max(0, total - used)
        util = round(used / total * 100, 1) if total else 0.0
        temp = round(random.uniform(30.0, 65.0), 1)
        return GpuStats(
            device_index=device_index,
            total_memory_mb=total,
            used_memory_mb=used,
            free_memory_mb=free,
            utilization_percent=util,
            temperature_celsius=temp,
        )
