from __future__ import annotations

import pytest

from backend.app.runtime.gpu_monitor import GpuMonitor, GpuStats


class TestGpuMonitorStats:
    def test_cpu_mode_returns_mock_stats(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        stats = monitor.query_gpu(device_index=0)
        assert isinstance(stats, GpuStats)
        assert stats.device_index == 0
        assert stats.total_memory_mb > 0
        assert stats.used_memory_mb >= 0
        assert stats.free_memory_mb >= 0
        assert 0 <= stats.utilization_percent <= 100
        assert 0 <= stats.temperature_celsius <= 150

    def test_cpu_mode_default_all_devices(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        all_stats = monitor.query_all_gpus()
        assert len(all_stats) >= 1
        assert all(isinstance(s, GpuStats) for s in all_stats)

    def test_cpu_mode_available_memory(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        stats = monitor.query_gpu(device_index=0)
        available = monitor.get_available_memory(device_index=0)
        assert available >= 0
        assert available <= stats.total_memory_mb

    def test_get_device_count_cpu_mode(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        count = monitor.get_device_count()
        assert count >= 1


class TestGpuMonitorDirectQuery:
    def test_query_multiple_devices(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        stats_0 = monitor.query_gpu(device_index=0)
        stats_1 = monitor.query_gpu(device_index=1)
        assert stats_0.device_index == 0
        assert stats_1.device_index == 1

    def test_stats_consistency(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        stats = monitor.query_gpu(device_index=0)
        assert stats.free_memory_mb == stats.total_memory_mb - stats.used_memory_mb


class TestGpuMonitorMemoryCheck:
    def test_has_enough_memory_true(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        assert monitor.has_enough_memory(device_index=0, required_mb=1) is True

    def test_has_enough_memory_false_when_exceeding(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        stats = monitor.query_gpu(device_index=0)
        assert (
            monitor.has_enough_memory(
                device_index=0, required_mb=stats.total_memory_mb + 1
            )
            is False
        )
