from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.dependencies import get_db, verify_api_key
from backend.app.models import GPUResource, GPUResourceLease, ModelResidencyPlan

router = APIRouter(prefix="/api/resources", tags=["resources"])


class GPUResourceResponse(BaseModel):
    id: UUID
    gpu_device: str
    capacity_memory_mb: int
    reserved_memory_mb: int
    model_config = {"from_attributes": True}


class GPUResourceListResponse(BaseModel):
    resources: list[GPUResourceResponse]
    total: int


class ResidencyPlanResponse(BaseModel):
    id: UUID
    binding_id: UUID
    release_id: UUID
    model_node_id: UUID
    gpu_device: str
    reserved_memory_mb: int
    desired_state: str
    priority: int
    model_config = {"from_attributes": True}


class ResidencyPlanListResponse(BaseModel):
    plans: list[ResidencyPlanResponse]
    total: int


@router.get("/gpu", response_model=GPUResourceListResponse)
def list_gpu_resources(
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    resources = list(db.execute(select(GPUResource)).scalars().all())
    return GPUResourceListResponse(
        resources=[GPUResourceResponse.model_validate(r) for r in resources],
        total=len(resources),
    )


@router.get("/gpu/{resource_id}", response_model=GPUResourceResponse)
def get_gpu_resource(
    resource_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    resource = db.get(GPUResource, resource_id)
    if resource is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="GPU resource not found"
        )
    return resource


@router.get("/gpu/{resource_id}/leases")
def get_gpu_leases(
    resource_id: UUID,
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    resource = db.get(GPUResource, resource_id)
    if resource is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="GPU resource not found"
        )
    leases = list(
        db.execute(
            select(GPUResourceLease).where(
                GPUResourceLease.gpu_resource_id == resource_id,
                GPUResourceLease.status == "active",
            )
        ).scalars().all()
    )
    return {
        "resource_id": str(resource_id),
        "leases": [
            {
                "id": str(l.id),
                "owner_type": l.owner_type,
                "owner_id": str(l.owner_id),
                "reserved_memory_mb": l.reserved_memory_mb,
                "status": l.status,
            }
            for l in leases
        ],
        "total_reserved": sum(l.reserved_memory_mb for l in leases),
    }


@router.get("/residency", response_model=ResidencyPlanListResponse)
def list_residency_plans(
    db: Session = Depends(get_db),
    _key: str = Depends(verify_api_key),
) -> Any:
    plans = list(db.execute(select(ModelResidencyPlan)).scalars().all())
    return ResidencyPlanListResponse(
        plans=[ResidencyPlanResponse.model_validate(p) for p in plans],
        total=len(plans),
    )
