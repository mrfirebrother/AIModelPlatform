from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from backend.app.models import GPUResource, GPUResourceLease

LEASE_DURATION_MINUTES = 30


def acquire_lease(
    session: Session,
    *,
    gpu_resource_id: UUID,
    owner_type: str,
    owner_id: UUID,
    reserved_memory_mb: int,
    lease_duration_minutes: int = LEASE_DURATION_MINUTES,
) -> GPUResourceLease:
    resource = session.get(GPUResource, gpu_resource_id, with_for_update=True)
    if resource is None:
        raise ValueError(f"GPU resource {gpu_resource_id} not found")

    now = datetime.now(timezone.utc)
    fencing_token = session.scalar(
        select(func.coalesce(func.max(GPUResourceLease.fencing_token), 0)).where(
            GPUResourceLease.gpu_resource_id == gpu_resource_id
        )
    )
    lease = GPUResourceLease(
        resource=resource,
        gpu_device=resource.gpu_device,
        owner_type=owner_type,
        owner_id=owner_id,
        reserved_memory_mb=reserved_memory_mb,
        lease_token=uuid4(),
        fencing_token=fencing_token + 1,
        lease_expires_at=now + timedelta(minutes=lease_duration_minutes),
        heartbeat_at=now,
        status="active",
    )
    session.add(lease)
    session.flush()
    return lease


def release_lease(
    session: Session,
    lease_id: UUID,
    lease_token: UUID,
) -> GPUResourceLease:
    lease = session.get(GPUResourceLease, lease_id, with_for_update=True)
    if lease is None:
        raise ValueError(f"Lease {lease_id} not found")
    if lease.lease_token != lease_token:
        raise ValueError("Invalid lease token")
    if lease.status != "active":
        return lease
    lease.status = "released"
    session.flush()
    return lease


def heartbeat_lease(
    session: Session,
    lease_id: UUID,
    lease_token: UUID,
    *,
    lease_duration_minutes: int = LEASE_DURATION_MINUTES,
) -> GPUResourceLease:
    lease = session.get(GPUResourceLease, lease_id, with_for_update=True)
    if lease is None:
        raise ValueError(f"Lease {lease_id} not found")
    if lease.lease_token != lease_token:
        raise ValueError("Invalid lease token")
    if lease.status != "active":
        return lease
    now = datetime.now(timezone.utc)
    lease.heartbeat_at = now
    lease.lease_expires_at = now + timedelta(minutes=lease_duration_minutes)
    session.flush()
    return lease


def find_expired_leases(session: Session) -> list[GPUResourceLease]:
    now = datetime.now(timezone.utc)
    return list(
        session.execute(
            select(GPUResourceLease).where(
                GPUResourceLease.status == "active",
                GPUResourceLease.lease_expires_at < now,
            )
        ).scalars().all()
    )


def expire_lease(session: Session, lease_id: UUID) -> GPUResourceLease:
    lease = session.get(GPUResourceLease, lease_id, with_for_update=True)
    if lease is None:
        raise ValueError(f"Lease {lease_id} not found")
    if lease.status != "active":
        return lease
    lease.status = "expired"
    session.flush()
    return lease
