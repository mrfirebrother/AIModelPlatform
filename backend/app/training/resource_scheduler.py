from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import GPUResource, GPUResourceLease, TrainingAttempt
from backend.app.runtime.gpu_monitor import GpuMonitor

logger = logging.getLogger(__name__)

LEASE_DURATION_MINUTES = 30


class ResourceSchedulerError(Exception):
    pass


class InsufficientGPUError(ResourceSchedulerError):
    pass


class ConcurrencyViolationError(ResourceSchedulerError):
    pass


class ResourceScheduler:
    """Manages GPU resource leasing for training attempts.

    Responsibilities:
    - Check GPU memory availability
    - Reserve (lease) GPU memory for a training attempt
    - Release GPU memory when training completes/fails
    - Prevent concurrent training on the same attempt
    """

    def __init__(
        self,
        session: Session,
        gpu_monitor: GpuMonitor | None = None,
        cpu_mode: bool = True,
    ) -> None:
        self._session = session
        self._gpu_monitor = gpu_monitor or GpuMonitor(cpu_mode=cpu_mode)
        self._cpu_mode = cpu_mode

    def get_available_gpu(
        self, required_memory_mb: int, exclude_lease_id: UUID | None = None
    ) -> GPUResource | None:
        """Find a GPU with enough free memory for the required amount."""
        resources = self._session.execute(
            select(GPUResource).order_by(GPUResource.gpu_device)
        ).scalars().all()

        for resource in resources:
            available = self._calculate_available_memory(
                resource, exclude_lease_id=exclude_lease_id
            )
            if available >= required_memory_mb:
                return resource
        return None

    def _calculate_available_memory(
        self,
        resource: GPUResource,
        exclude_lease_id: UUID | None = None,
    ) -> int:
        """Calculate available memory for a GPU resource."""
        query = select(GPUResourceLease).where(
            GPUResourceLease.gpu_resource_id == resource.id,
            GPUResourceLease.status == "active",
        )
        if exclude_lease_id:
            query = query.where(GPUResourceLease.id != exclude_lease_id)

        active_leases = self._session.execute(query).scalars().all()
        reserved = sum(lease.reserved_memory_mb for lease in active_leases)
        return max(0, resource.capacity_memory_mb - reserved)

    def reserve_gpu(
        self,
        attempt_id: UUID,
        required_memory_mb: int,
        gpu_device: str | None = None,
    ) -> GPUResourceLease:
        """Reserve GPU memory for a training attempt.

        Raises:
            ConcurrencyViolationError: If attempt already has an active lease
            InsufficientGPUError: If no GPU has enough memory
        """
        existing_lease = self._find_active_lease(attempt_id)
        if existing_lease is not None:
            raise ConcurrencyViolationError(
                f"Attempt {attempt_id} already has active lease {existing_lease.id}"
            )

        if gpu_device:
            resource = self._session.execute(
                select(GPUResource).where(GPUResource.gpu_device == gpu_device)
            ).scalar_one_or_none()
            if resource is None:
                raise InsufficientGPUError(f"GPU device {gpu_device} not found")
            available = self._calculate_available_memory(resource)
            if available < required_memory_mb:
                raise InsufficientGPUError(
                    f"GPU {gpu_device} has {available}MB available, "
                    f"need {required_memory_mb}MB"
                )
        else:
            resource = self.get_available_gpu(required_memory_mb)
            if resource is None:
                raise InsufficientGPUError(
                    f"No GPU has {required_memory_mb}MB available"
                )

        lease_token = uuid4()
        fencing_token = self._next_fencing_token(resource)

        lease = GPUResourceLease(
            resource=resource,
            gpu_device=resource.gpu_device,
            owner_type="training_attempt",
            owner_id=attempt_id,
            reserved_memory_mb=required_memory_mb,
            lease_token=lease_token,
            fencing_token=fencing_token,
            lease_expires_at=datetime.now(timezone.utc) + timedelta(
                minutes=LEASE_DURATION_MINUTES
            ),
            status="active",
        )
        self._session.add(lease)
        self._session.flush()

        logger.info(
            "Reserved %dMB on GPU %s for attempt %s (lease %s)",
            required_memory_mb,
            resource.gpu_device,
            attempt_id,
            lease.id,
        )
        return lease

    def release_gpu(self, attempt_id: UUID) -> GPUResourceLease | None:
        """Release GPU lease for a training attempt."""
        lease = self._find_active_lease(attempt_id)
        if lease is None:
            return None

        lease.status = "released"
        self._session.flush()

        logger.info(
            "Released GPU lease %s for attempt %s",
            lease.id,
            attempt_id,
        )
        return lease

    def heartbeat(self, attempt_id: UUID, lease_token: UUID) -> GPUResourceLease:
        """Extend lease expiration for a running attempt."""
        lease = self._session.execute(
            select(GPUResourceLease).where(
                GPUResourceLease.owner_type == "training_attempt",
                GPUResourceLease.owner_id == attempt_id,
                GPUResourceLease.lease_token == lease_token,
                GPUResourceLease.status == "active",
            )
        ).scalar_one_or_none()

        if lease is None:
            raise ResourceSchedulerError(
                f"No active lease for attempt {attempt_id} with given token"
            )

        lease.heartbeat_at = datetime.now(timezone.utc)
        lease.lease_expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=LEASE_DURATION_MINUTES
        )
        self._session.flush()
        return lease

    def _find_active_lease(self, attempt_id: UUID) -> GPUResourceLease | None:
        """Find active lease for a training attempt."""
        return self._session.execute(
            select(GPUResourceLease).where(
                GPUResourceLease.owner_type == "training_attempt",
                GPUResourceLease.owner_id == attempt_id,
                GPUResourceLease.status == "active",
            )
        ).scalar_one_or_none()

    def _next_fencing_token(self, resource: GPUResource) -> int:
        """Generate next fencing token for a GPU resource."""
        last_lease = self._session.execute(
            select(GPUResourceLease)
            .where(GPUResourceLease.gpu_resource_id == resource.id)
            .order_by(GPUResourceLease.fencing_token.desc())
            .limit(1)
        ).scalar_one_or_none()

        if last_lease is None:
            return 1
        return last_lease.fencing_token + 1

    def ensure_gpu_resource(
        self, device: str, capacity_mb: int
    ) -> GPUResource:
        """Ensure a GPU resource record exists, create if not."""
        resource = self._session.execute(
            select(GPUResource).where(GPUResource.gpu_device == device)
        ).scalar_one_or_none()

        if resource is None:
            resource = GPUResource(
                gpu_device=device,
                capacity_memory_mb=capacity_mb,
                reserved_memory_mb=0,
            )
            self._session.add(resource)
            self._session.flush()
            logger.info("Created GPU resource record for device %s", device)

        return resource
