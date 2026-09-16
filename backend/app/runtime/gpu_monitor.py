from __future__ import annotations

import dataclasses
import logging
import random
import shutil
import subprocess

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class GpuStats:
    """Snapshot of GPU telemetry for a single device."""

    device_index: int
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilization_percent: float
    temperature_celsius: float
    name: str = ""
    source: str = "mock"


_MOCK_TOTAL_MB = 8192
_MOCK_DEVICE_COUNT = 2
_SMI_FIELDS = "index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"


class GpuMonitor:
    """GPU telemetry.

    ``cpu_mode=True`` returns simulated numbers, which is what the unit tests and any
    machine without a usable driver get. With ``cpu_mode=False`` real telemetry is read
    through **NVML** (``pynvml``/``nvidia-ml-py``) and, if that is unavailable, by parsing
    ``nvidia-smi``; if neither works the monitor degrades to the simulated values and logs
    a warning rather than raising.

    Note that ``reserve_memory``/``release_memory`` only maintain bookkeeping counters -
    real VRAM is owned by the driver, so those never change what ``query_gpu`` reports on a
    real device.
    """

    def __init__(self, *, cpu_mode: bool = True) -> None:
        self._cpu_mode = cpu_mode
        self._mock_used: dict[int, int] = {}
        self._nvml: object | None = None
        self._backend = "mock"
        if not cpu_mode:
            self._backend = self._init_backend()
            if self._backend == "mock":
                logger.warning(
                    "Real GPU telemetry unavailable (no NVML and no nvidia-smi); "
                    "falling back to simulated values"
                )

    # ------------------------------------------------------------------
    # backend selection
    # ------------------------------------------------------------------

    def _init_backend(self) -> str:
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            if int(pynvml.nvmlDeviceGetCount()) > 0:
                self._nvml = pynvml
                return "nvml"
            pynvml.nvmlShutdown()
        except Exception as exc:  # noqa: BLE001 - any failure means "no NVML here"
            logger.debug("NVML unavailable: %s", exc)
        return "nvidia-smi" if shutil.which("nvidia-smi") else "mock"

    @property
    def backend(self) -> str:
        """``"nvml"``, ``"nvidia-smi"`` or ``"mock"``."""
        return self._backend

    def is_real(self) -> bool:
        return self._backend != "mock"

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get_device_count(self) -> int:
        if self._cpu_mode or self._backend == "mock":
            return _MOCK_DEVICE_COUNT
        if self._backend == "nvml":
            return int(self._nvml.nvmlDeviceGetCount())  # type: ignore[union-attr]
        return len(self._smi_query())

    def query_gpu(self, device_index: int) -> GpuStats:
        if self._cpu_mode or self._backend == "mock":
            return self._mock_query(device_index)
        try:
            if self._backend == "nvml":
                return self._nvml_query(device_index)
            stats = self._smi_query()
            return stats[device_index]
        except Exception as exc:  # noqa: BLE001 - never break the caller over telemetry
            logger.warning("GPU query for device %s failed: %s", device_index, exc)
            return self._mock_query(device_index)

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
    # real backends
    # ------------------------------------------------------------------

    def _nvml_query(self, device_index: int) -> GpuStats:
        nvml = self._nvml
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)  # type: ignore[union-attr]
        name = nvml.nvmlDeviceGetName(handle)  # type: ignore[union-attr]
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        mem = nvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[union-attr]
        total = int(mem.total // (1024 * 1024))
        used = int(mem.used // (1024 * 1024))
        try:
            util = float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            util = 0.0
        temperature = 0.0
        for sensor in (0, 1):  # 0 is the GPU core on most drivers
            try:
                value = float(nvml.nvmlDeviceGetTemperature(handle, sensor))  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001 - not every board exposes a sensor
                continue
            if value > 0:
                temperature = value
                break
        return GpuStats(
            device_index=device_index,
            total_memory_mb=total,
            used_memory_mb=used,
            free_memory_mb=max(0, total - used),
            utilization_percent=util,
            temperature_celsius=temperature,
            name=str(name),
            source="nvml",
        )

    def _smi_query(self) -> list[GpuStats]:
        smi = shutil.which("nvidia-smi") or "nvidia-smi"
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [smi, f"--query-gpu={_SMI_FIELDS}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        stats: list[GpuStats] = []
        for line in completed.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 7:
                continue
            total = _as_int(parts[2])
            used = _as_int(parts[3])
            stats.append(
                GpuStats(
                    device_index=_as_int(parts[0]),
                    total_memory_mb=total,
                    used_memory_mb=used,
                    free_memory_mb=max(0, total - used),
                    utilization_percent=_as_float(parts[5]),
                    temperature_celsius=_as_float(parts[6]),
                    name=parts[1],
                    source="nvidia-smi",
                )
            )
        if not stats:
            raise RuntimeError("nvidia-smi returned no GPU rows")
        return stats

    # ------------------------------------------------------------------
    # simulated backend
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
            name=f"simulated-gpu-{device_index}",
            source="mock",
        )


def _as_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_gpu_monitor() -> GpuMonitor:
    """Monitor matching the runtime configuration (``GPU_DEVICE=cpu`` keeps it simulated)."""
    from backend.app.config import get_settings

    return GpuMonitor(cpu_mode=str(get_settings().gpu_device).lower() == "cpu")
