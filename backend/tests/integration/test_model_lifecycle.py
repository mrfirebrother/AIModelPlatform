from __future__ import annotations

import time
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from backend.app.models import Base, ModelNode
from backend.app.runtime.gpu_monitor import GpuMonitor
from backend.app.runtime.model_manager import ModelManager


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    event.listen(
        engine,
        "connect",
        lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
    )
    Base.metadata.create_all(engine)
    with Session(engine) as db_session:
        yield db_session
    engine.dispose()


def _make_model(session: Session, name: str = "test_model") -> ModelNode:
    model = ModelNode(
        artifact_path=f"models/{uuid4()}.pt",
        artifact_hash=f"sha256:{uuid4().hex}",
        task_type="object_detection",
        model_family="yolo",
        status="approved",
        metadata_json={"model_name": name},
    )
    session.add(model)
    session.flush()
    return model


class TestModelLoadUnloadCycle:
    def test_full_load_evict_cycle(self, session: Session) -> None:
        _make_model(session, "yolo_v8")
        session.commit()

        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor, max_models=2)

        d1 = mgr.load("yolo_v8", memory_mb=512)
        assert mgr.get(d1) is not None
        assert mgr.get_total_memory_used() == 512

        mgr.unload(d1)
        assert mgr.get(d1) is None
        assert mgr.get_total_memory_used() == 0

    def test_lru_eviction_under_capacity_pressure(self, session: Session) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor, max_models=3)

        d_a = mgr.load("model_a", memory_mb=100)
        time.sleep(0.01)
        d_b = mgr.load("model_b", memory_mb=100)
        time.sleep(0.01)
        d_c = mgr.load("model_c", memory_mb=100)
        time.sleep(0.01)

        # model_a is LRU, should be evicted
        d_d = mgr.load("model_d", memory_mb=100)

        resident = mgr.list_resident()
        names = [e.model_name for e in resident]
        assert "model_a" not in names
        assert "model_d" in names
        assert len(resident) == 3

    def test_concurrent_model_loading(self, session: Session) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor, max_models=5)

        devices = []
        for i in range(4):
            d = mgr.load(f"model_{i}", memory_mb=64)
            devices.append(d)

        assert len(set(devices)) == 4
        assert mgr.get_total_memory_used() == 256

    def test_model_reloading_updates_access(self, session: Session) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor, max_models=2)

        d1 = mgr.load("model_a", memory_mb=100)
        entry_a = mgr.get(d1)
        t1 = entry_a.last_accessed

        time.sleep(0.01)
        d2 = mgr.load("model_b", memory_mb=100)
        time.sleep(0.01)

        # Reload model_a to make it recently used
        mgr.load("model_a", memory_mb=100)
        time.sleep(0.01)

        # Load model_c; model_b should be evicted (LRU)
        mgr.load("model_c", memory_mb=100)

        resident = mgr.list_resident()
        names = [e.model_name for e in resident]
        assert "model_b" not in names
        assert "model_a" in names
        assert "model_c" in names


class TestGpuMonitorIntegration:
    def test_monitor_tracks_memory_across_loads(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        mgr = ModelManager(gpu_monitor=monitor, max_models=3)

        stats_before = monitor.query_gpu(device_index=0)
        initial_used = stats_before.used_memory_mb

        d1 = mgr.load("model_a", memory_mb=256)
        stats_mid = monitor.query_gpu(device_index=0)
        assert stats_mid.used_memory_mb == initial_used + 256

        mgr.unload(d1)
        stats_after = monitor.query_gpu(device_index=0)
        assert stats_after.used_memory_mb == initial_used

    def test_multiple_device_queries(self) -> None:
        monitor = GpuMonitor(cpu_mode=True)
        count = monitor.get_device_count()
        assert count >= 1

        for idx in range(count):
            stats = monitor.query_gpu(device_index=idx)
            assert stats.total_memory_mb > 0
            assert stats.device_index == idx
