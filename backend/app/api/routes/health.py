from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}
