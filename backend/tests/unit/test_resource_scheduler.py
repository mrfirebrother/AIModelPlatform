from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models import Base, GPUResource, GPUResourceLease
from backend.app.runtime.gpu_monitor import GpuMonitor
from backend.app.training.resource_scheduler import (
    ConcurrencyViolationError,
    InsufficientGPUError,
    ResourceScheduler,
    ResourceSchedulerError,
)


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


@pytest.fixture()
def cpu_monitor() -> GpuMonitor:
    return GpuMonitor(cpu_mode=True)


@pytest.fixture()
def scheduler(session: Session, cpu_monitor: GpuMonitor) -> ResourceScheduler:
    return ResourceScheduler(session=session, gpu_monitor=cpu_monitor, cpu_mode=True)


def _create_gpu_resource(
    session: Session, device: str = "0", capacity: int = 8192
) -> GPUResource:
    resource = GPUResource(
        gpu_device=device,
        capacity_memory_mb=capacity,
        reserved_memory_mb=0,
    )
    session.add(resource)
    session.flush()
    return resource


class TestResourceSchedulerReserveGPU:
    def test_reserve_gpu_creates_active_lease(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 8192)
        attempt_id = uuid4()

        lease = scheduler.reserve_gpu(attempt_id, required_memory_mb=4096)

        assert lease.id is not None
        assert lease.status == "active"
        assert lease.reserved_memory_mb == 4096
        assert lease.owner_id == attempt_id
        assert lease.owner_type == "training_attempt"
        assert lease.lease_token is not None
        assert lease.fencing_token == 1

    def test_reserve_gpu_prevents_duplicate_lease(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 8192)
        attempt_id = uuid4()

        scheduler.reserve_gpu(attempt_id, required_memory_mb=4096)

        with pytest.raises(ConcurrencyViolationError, match="already has active lease"):
            scheduler.reserve_gpu(attempt_id, required_memory_mb=2048)

    def test_reserve_gpu_selects_device_with_enough_memory(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 4096)
        _create_gpu_resource(session, "1", 8192)
        attempt_id = uuid4()

        lease = scheduler.reserve_gpu(attempt_id, required_memory_mb=6000)

        assert lease.gpu_device == "1"

    def test_reserve_gpu_raises_when_insufficient_memory(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 4096)
        attempt_id = uuid4()

        with pytest.raises(InsufficientGPUError, match="No GPU has"):
            scheduler.reserve_gpu(attempt_id, required_memory_mb=8192)

    def test_reserve_gpu_specific_device(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 8192)
        _create_gpu_resource(session, "1", 8192)
        attempt_id = uuid4()

        lease = scheduler.reserve_gpu(
            attempt_id, required_memory_mb=4096, gpu_device="1"
        )

        assert lease.gpu_device == "1"

    def test_reserve_gpu_specific_device_not_found(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 8192)
        attempt_id = uuid4()

        with pytest.raises(InsufficientGPUError, match="not found"):
            scheduler.reserve_gpu(
                attempt_id, required_memory_mb=4096, gpu_device="99"
            )

    def test_reserve_gpu_specific_device_insufficient_memory(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 4096)
        attempt_id = uuid4()

        with pytest.raises(InsufficientGPUError, match="has.*available"):
            scheduler.reserve_gpu(
                attempt_id, required_memory_mb=8192, gpu_device="0"
            )

    def test_reserve_gpu_accounts_for_existing_leases(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 4096)
        scheduler.reserve_gpu(uuid4(), required_memory_mb=3000)

        with pytest.raises(InsufficientGPUError):
            scheduler.reserve_gpu(uuid4(), required_memory_mb=2000)

    def test_reserve_gpu_increments_fencing_token(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 8192)
        lease1 = scheduler.reserve_gpu(uuid4(), required_memory_mb=1024)
        lease2 = scheduler.reserve_gpu(uuid4(), required_memory_mb=1024)

        assert lease2.fencing_token == lease1.fencing_token + 1


class TestResourceSchedulerReleaseGPU:
    def test_release_gpu_sets_released_status(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 8192)
        attempt_id = uuid4()
        scheduler.reserve_gpu(attempt_id, required_memory_mb=4096)

        released = scheduler.release_gpu(attempt_id)

        assert released is not None
        assert released.status == "released"

    def test_release_gpu_returns_none_if_no_lease(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 8192)
        result = scheduler.release_gpu(uuid4())
        assert result is None

    def test_release_gpu_allows_new_lease_for_same_attempt(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 8192)
        attempt_id = uuid4()

        scheduler.reserve_gpu(attempt_id, required_memory_mb=4096)
        scheduler.release_gpu(attempt_id)
        lease2 = scheduler.reserve_gpu(attempt_id, required_memory_mb=4096)

        assert lease2.id is not None
        assert lease2.status == "active"


class TestResourceSchedulerHeartbeat:
    def test_heartbeat_extends_lease_expiration(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 8192)
        attempt_id = uuid4()
        lease = scheduler.reserve_gpu(attempt_id, required_memory_mb=4096)
        old_expiration = lease.lease_expires_at

        updated = scheduler.heartbeat(attempt_id, lease.lease_token)

        assert updated.heartbeat_at is not None
        assert updated.lease_expires_at > old_expiration

    def test_heartbeat_rejects_invalid_token(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 8192)
        attempt_id = uuid4()
        scheduler.reserve_gpu(attempt_id, required_memory_mb=4096)

        with pytest.raises(ResourceSchedulerError, match="No active lease"):
            scheduler.heartbeat(attempt_id, uuid4())

    def test_heartbeat_rejects_released_lease(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 8192)
        attempt_id = uuid4()
        lease = scheduler.reserve_gpu(attempt_id, required_memory_mb=4096)
        scheduler.release_gpu(attempt_id)

        with pytest.raises(ResourceSchedulerError, match="No active lease"):
            scheduler.heartbeat(attempt_id, lease.lease_token)


class TestResourceSchedulerGetAvailableGPU:
    def test_get_available_gpu_returns_none_when_no_resources(
        self, scheduler: ResourceScheduler
    ) -> None:
        result = scheduler.get_available_gpu(required_memory_mb=4096)
        assert result is None

    def test_get_available_gpu_returns_device_with_enough_memory(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 4096)
        _create_gpu_resource(session, "1", 8192)

        result = scheduler.get_available_gpu(required_memory_mb=6000)

        assert result is not None
        assert result.gpu_device == "1"

    def test_get_available_gpu_excludes_lease(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        resource = _create_gpu_resource(session, "0", 4096)
        attempt_id = uuid4()
        lease = scheduler.reserve_gpu(attempt_id, required_memory_mb=3000)

        result = scheduler.get_available_gpu(
            required_memory_mb=2000, exclude_lease_id=lease.id
        )

        assert result is not None
        assert result.id == resource.id

    def test_get_available_gpu_accounts_for_existing_leases(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        _create_gpu_resource(session, "0", 4096)
        scheduler.reserve_gpu(uuid4(), required_memory_mb=3000)

        result = scheduler.get_available_gpu(required_memory_mb=2000)

        assert result is None


class TestResourceSchedulerEnsureGPUResource:
    def test_ensure_gpu_resource_creates_new(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        resource = scheduler.ensure_gpu_resource("0", 8192)

        assert resource.id is not None
        assert resource.gpu_device == "0"
        assert resource.capacity_memory_mb == 8192

    def test_ensure_gpu_resource_returns_existing(
        self, session: Session, scheduler: ResourceScheduler
    ) -> None:
        existing = _create_gpu_resource(session, "0", 8192)

        result = scheduler.ensure_gpu_resource("0", 8192)

        assert result.id == existing.id
        count = session.execute(
            select(GPUResource).where(GPUResource.gpu_device == "0")
        ).scalars().all().__len__()
        assert count == 1
