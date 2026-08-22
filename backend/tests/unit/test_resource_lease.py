from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.models import Base, GPUResource, GPUResourceLease


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


def _make_resource(
    session: Session,
    device: str = "0",
    capacity: int = 1024,
) -> GPUResource:
    resource = GPUResource(
        gpu_device=device,
        capacity_memory_mb=capacity,
        reserved_memory_mb=0,
    )
    session.add(resource)
    session.flush()
    return resource


def _make_active_lease(
    session: Session,
    resource: GPUResource,
    *,
    owner_id: uuid4 | None = None,
    memory_mb: int = 128,
    fencing_token: int = 1,
) -> GPUResourceLease:
    lease = GPUResourceLease(
        resource=resource,
        gpu_device=resource.gpu_device,
        owner_type="training_attempt",
        owner_id=owner_id or uuid4(),
        reserved_memory_mb=memory_mb,
        lease_token=uuid4(),
        fencing_token=fencing_token,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        status="active",
    )
    session.add(lease)
    session.flush()
    return lease


class TestGPUResourceLeaseCapacity:
    def test_active_lease_reduces_available_capacity(
        self, session: Session
    ) -> None:
        resource = _make_resource(session, capacity=1024)
        _make_active_lease(session, resource, memory_mb=512)
        session.commit()
        session.refresh(resource)
        assert resource.reserved_memory_mb == 512

    def test_multiple_active_leases_sum_reserved_memory(
        self, session: Session
    ) -> None:
        resource = _make_resource(session, capacity=1024)
        _make_active_lease(session, resource, memory_mb=256, fencing_token=1)
        _make_active_lease(session, resource, memory_mb=128, fencing_token=2)
        session.commit()
        session.refresh(resource)
        assert resource.reserved_memory_mb == 384

    def test_exceeding_capacity_is_rejected(self, session: Session) -> None:
        resource = _make_resource(session, capacity=100)
        _make_active_lease(session, resource, memory_mb=80, fencing_token=1)
        session.commit()
        second = GPUResourceLease(
            resource=resource,
            gpu_device="0",
            owner_type="training_attempt",
            owner_id=uuid4(),
            reserved_memory_mb=30,
            lease_token=uuid4(),
            fencing_token=2,
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            status="active",
        )
        session.add(second)
        with pytest.raises(ValueError, match="capacity"):
            session.commit()

    def test_released_lease_frees_capacity(self, session: Session) -> None:
        resource = _make_resource(session, capacity=100)
        lease = _make_active_lease(session, resource, memory_mb=80)
        session.commit()
        lease.status = "released"
        session.commit()
        session.refresh(resource)
        assert resource.reserved_memory_mb == 0

    def test_expired_lease_frees_capacity(self, session: Session) -> None:
        resource = _make_resource(session, capacity=100)
        lease = _make_active_lease(session, resource, memory_mb=80)
        session.commit()
        lease.status = "expired"
        session.commit()
        session.refresh(resource)
        assert resource.reserved_memory_mb == 0

    def test_failed_lease_frees_capacity(self, session: Session) -> None:
        resource = _make_resource(session, capacity=100)
        lease = _make_active_lease(session, resource, memory_mb=80)
        session.commit()
        lease.status = "failed"
        session.commit()
        session.refresh(resource)
        assert resource.reserved_memory_mb == 0

    def test_capacity_shrink_below_reserved_is_rejected(
        self, session: Session
    ) -> None:
        resource = _make_resource(session, capacity=100)
        _make_active_lease(session, resource, memory_mb=80)
        session.commit()
        resource.capacity_memory_mb = 50
        with pytest.raises(ValueError, match="capacity"):
            session.commit()


class TestGPULeaseOwnerUniqueness:
    def test_same_owner_cannot_have_two_active_leases(
        self, session: Session
    ) -> None:
        resource = _make_resource(session, capacity=1024)
        owner = uuid4()
        _make_active_lease(session, resource, owner_id=owner, fencing_token=1)
        session.commit()
        second = GPUResourceLease(
            resource=resource,
            gpu_device="0",
            owner_type="training_attempt",
            owner_id=owner,
            reserved_memory_mb=128,
            lease_token=uuid4(),
            fencing_token=2,
            lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            status="active",
        )
        session.add(second)
        with pytest.raises(IntegrityError):
            session.commit()

    def test_different_owners_can_have_active_leases(
        self, session: Session
    ) -> None:
        resource = _make_resource(session, capacity=1024)
        _make_active_lease(session, resource, owner_id=uuid4(), fencing_token=1)
        _make_active_lease(session, resource, owner_id=uuid4(), fencing_token=2)
        session.commit()
        count = session.scalar(
            select(GPUResourceLease).where(
                GPUResourceLease.gpu_resource_id == resource.id,
                GPUResourceLease.status == "active",
            )
        )
        # Multiple active leases from different owners are allowed
        assert session.execute(
            select(GPUResourceLease).where(
                GPUResourceLease.gpu_resource_id == resource.id,
                GPUResourceLease.status == "active",
            )
        ).scalars().all().__len__() == 2


class TestGPULeaseIdentityImmutability:
    def test_gpu_device_cannot_change(self, session: Session) -> None:
        resource = _make_resource(session, device="0", capacity=1024)
        lease = _make_active_lease(session, resource)
        session.commit()
        lease.gpu_device = "1"
        with pytest.raises(ValueError, match="immutable"):
            session.commit()

    def test_gpu_resource_id_cannot_change(self, session: Session) -> None:
        r1 = _make_resource(session, device="0", capacity=1024)
        r2 = _make_resource(session, device="1", capacity=1024)
        lease = _make_active_lease(session, r1)
        session.commit()
        lease.gpu_resource_id = r2.id
        with pytest.raises(ValueError, match="immutable"):
            session.commit()
