from __future__ import annotations

import time

import pytest

from backend.app.runtime.gpu_monitor import GpuMonitor
from backend.app.runtime.model_manager import (
    EvictionError,
    ModelEntry,
    ModelManager,
)


class TestModelManagerLoad:
    def test_load_returns_device_string(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        device = mgr.load("yolo_v8", memory_mb=256)
        assert device.startswith("gpu:")

    def test_load_registers_model_entry(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        device = mgr.load("yolo_v8", memory_mb=256)
        entry = mgr.get(device)
        assert entry is not None
        assert entry.model_name == "yolo_v8"
        assert entry.memory_mb == 256
        assert entry.gpu_device == device

    def test_load_multiple_models(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        d1 = mgr.load("model_a", memory_mb=128)
        d2 = mgr.load("model_b", memory_mb=256)
        assert d1 != d2
        assert len(mgr.list_resident()) == 2

    def test_load_same_model_twice_returns_existing(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        d1 = mgr.load("yolo_v8", memory_mb=256)
        d2 = mgr.load("yolo_v8", memory_mb=256)
        assert d1 == d2
        assert len(mgr.list_resident()) == 1

    def test_load_updates_access_time(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        device = mgr.load("yolo_v8", memory_mb=256)
        entry = mgr.get(device)
        t1 = entry.last_accessed
        time.sleep(0.01)
        mgr.load("yolo_v8", memory_mb=256)
        entry2 = mgr.get(device)
        assert entry2.last_accessed >= t1


class TestModelManagerUnload:
    def test_unload_removes_model(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        device = mgr.load("yolo_v8", memory_mb=256)
        assert mgr.unload(device) is True
        assert mgr.get(device) is None
        assert len(mgr.list_resident()) == 0

    def test_unload_nonexistent_returns_false(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        assert mgr.unload("gpu:999") is False

    def test_unload_frees_memory(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        device = mgr.load("yolo_v8", memory_mb=256)
        stats_before = monitor.query_gpu(device_index=0)
        mgr.unload(device)
        stats_after = monitor.query_gpu(device_index=0)
        assert stats_after.used_memory_mb <= stats_before.used_memory_mb


class TestModelManagerEviction:
    def test_evict_lru_when_overloaded(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor, max_models=2)
        mgr.load("model_a", memory_mb=100)
        time.sleep(0.01)
        mgr.load("model_b", memory_mb=100)
        time.sleep(0.01)
        # Third load should evict model_a (LRU)
        mgr.load("model_c", memory_mb=100)
        resident = mgr.list_resident()
        names = [e.model_name for e in resident]
        assert "model_a" not in names
        assert "model_c" in names

    def test_evict_lru_recent_not_evicted(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor, max_models=2)
        d_a = mgr.load("model_a", memory_mb=100)
        d_b = mgr.load("model_b", memory_mb=100)
        # Make model_a more recently accessed than model_b
        entry_a = mgr.get(d_a)
        entry_a.last_accessed = time.time() + 10
        mgr.load("model_c", memory_mb=100)
        resident = mgr.list_resident()
        names = [e.model_name for e in resident]
        assert "model_b" not in names
        assert "model_a" in names

    def test_evict_raises_when_nothing_to_evict(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor, max_models=1)
        mgr.load("model_a", memory_mb=100)
        # Force the model to be locked (simulate)
        entry = mgr.get_next_eviction_candidate()
        if entry is not None:
            mgr.lock(entry.gpu_device)
        with pytest.raises(EvictionError):
            mgr.evict_lru()


class TestModelManagerResidency:
    def test_list_resident_empty(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        assert mgr.list_resident() == []

    def test_list_resident_after_load(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        mgr.load("model_a", memory_mb=128)
        mgr.load("model_b", memory_mb=256)
        resident = mgr.list_resident()
        assert len(resident) == 2
        assert all(isinstance(e, ModelEntry) for e in resident)

    def test_get_nonexistent_returns_none(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        assert mgr.get("gpu:999") is None


class TestModelManagerMemoryTracking:
    def test_total_memory_used(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        mgr.load("model_a", memory_mb=128)
        mgr.load("model_b", memory_mb=256)
        assert mgr.get_total_memory_used() == 384

    def test_total_memory_used_empty(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        assert mgr.get_total_memory_used() == 0

    def test_memory_after_unload(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor)
        device = mgr.load("model_a", memory_mb=128)
        assert mgr.get_total_memory_used() == 128
        mgr.unload(device)
        assert mgr.get_total_memory_used() == 0
