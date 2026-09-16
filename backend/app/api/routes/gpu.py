from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.app.api.dependencies import verify_api_key
from backend.app.runtime.gpu_monitor import build_gpu_monitor
from backend.app.runtime.model_manager import ModelManager

router = APIRouter(prefix="/api/gpu", tags=["gpu"])

# Real telemetry whenever GPU_DEVICE points at a device; simulated otherwise, and also
# whenever the driver cannot be queried (the monitor then logs and degrades instead of
# raising - see GpuMonitor._init_backend).
_monitor = build_gpu_monitor()
_manager = ModelManager(gpu_monitor=_monitor)


class GpuDeviceEntry(BaseModel):
    index: int
    name: str
    total_memory_mb: int
    used_memory_mb: int
    utilization_percent: float


class GpuStatusResponse(BaseModel):
    device_count: int
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    # Added so callers can tell real telemetry apart from simulated values.
    utilization_percent: float = 0.0
    source: str = "mock"
    devices: list[GpuDeviceEntry] = []


class GpuModelEntry(BaseModel):
    model_name: str
    gpu_device: str
    memory_mb: int


class GpuModelListResponse(BaseModel):
    models: list[GpuModelEntry]


class LoadModelRequest(BaseModel):
    model_name: str
    memory_mb: int


class LoadModelResponse(BaseModel):
    success: bool
    gpu_device: str | None = None


class UnloadModelRequest(BaseModel):
    model_name: str


class UnloadModelResponse(BaseModel):
    success: bool


@router.get("/status", response_model=GpuStatusResponse)
def get_gpu_status(
    _key: str = Depends(verify_api_key),
) -> Any:
    all_stats = _monitor.query_all_gpus()
    total = sum(s.total_memory_mb for s in all_stats)
    # Real VRAM in use across all processes, not just the memory this API handed out.
    used = sum(s.used_memory_mb for s in all_stats)
    free = max(0, total - used)
    average_util = (
        round(sum(s.utilization_percent for s in all_stats) / len(all_stats), 1)
        if all_stats
        else 0.0
    )
    return GpuStatusResponse(
        device_count=len(all_stats),
        total_memory_mb=total,
        used_memory_mb=used,
        free_memory_mb=free,
        utilization_percent=average_util,
        source=_monitor.backend,
        devices=[
            GpuDeviceEntry(
                index=s.device_index,
                name=s.name,
                total_memory_mb=s.total_memory_mb,
                used_memory_mb=s.used_memory_mb,
                utilization_percent=s.utilization_percent,
            )
            for s in all_stats
        ],
    )


@router.get("/models", response_model=GpuModelListResponse)
def get_loaded_models(
    _key: str = Depends(verify_api_key),
) -> Any:
    entries = _manager.list_resident()
    return GpuModelListResponse(
        models=[
            GpuModelEntry(
                model_name=e.model_name,
                gpu_device=e.gpu_device,
                memory_mb=e.memory_mb,
            )
            for e in entries
        ]
    )


@router.post("/load", response_model=LoadModelResponse)
def load_model(
    payload: LoadModelRequest,
    _key: str = Depends(verify_api_key),
) -> Any:
    try:
        device = _manager.load(payload.model_name, memory_mb=payload.memory_mb)
        return LoadModelResponse(success=True, gpu_device=device)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load model",
        )


@router.post("/unload", response_model=UnloadModelResponse)
def unload_model(
    payload: UnloadModelRequest,
    _key: str = Depends(verify_api_key),
) -> Any:
    entries = _manager.list_resident()
    target = next((e for e in entries if e.model_name == payload.model_name), None)
    if target is None:
        return UnloadModelResponse(success=False)
    ok = _manager.unload(target.gpu_device)
    return UnloadModelResponse(success=ok)
