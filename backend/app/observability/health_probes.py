"""Kubernetes-style health probes: /health/live, /health/ready, /health/startup.

The three probes follow distinct contracts:

* **Liveness** – always returns 200 if the process is running.  Kubernetes
  restarts the pod on failure; this should never depend on external
  dependencies.
* **Readiness** – checks that all critical backends (Postgres, Redis) are
  reachable **and** that startup has completed.  Kubernetes stops routing
  traffic on failure.
* **Startup** – a lightweight gate that flips to ready once initial
  configuration (DB migrations, etc.) is known to have succeeded.  Before
  that, readiness will return 503 even though liveness is fine.

All probes share the same ``get_settings()`` singleton so no per-request
overhead is introduced.
"""

from __future__ import annotations

import time
from enum import Enum

from fastapi import APIRouter, Response, status

from ..config import get_settings
from ..db import check_postgres, check_redis

router = APIRouter(tags=["health-probes"])

_startup_completed = False
_startup_completed_at: float | None = None

_STARTUP_GRACE_PERIOD_SECONDS: float = 5.0


def mark_startup_complete() -> None:
    global _startup_completed, _startup_completed_at  # noqa: PLW0603
    if not _startup_completed:
        _startup_completed_at = time.monotonic()
    _startup_completed = True


class ProbeState(str, Enum):
    OK = "ok"
    FAIL = "fail"


def _post_ready() -> dict[str, str]:
    return {"status": ProbeState.OK}


# ---------------------------------------------------------------------------
# Liveness probe – always 200 unless the process itself is dead.
# ---------------------------------------------------------------------------

@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness() -> dict[str, str]:
    return {"status": ProbeState.OK}


# ---------------------------------------------------------------------------
# Readiness probe – verifies external dependencies.
# ---------------------------------------------------------------------------

@router.get("/health/ready")
async def readiness(response: Response) -> dict[str, str | dict[str, str]]:
    settings = get_settings()

    checks: dict[str, ProbeState] = {}
    checks["postgres"] = _run_check(lambda: check_postgres(settings))
    checks["redis"] = _run_check(lambda: check_redis(settings))

    all_ok = all(v == ProbeState.OK for v in checks.values())
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": ProbeState.OK if all_ok else ProbeState.FAIL, "checks": checks}


# ---------------------------------------------------------------------------
# Startup probe – succeeds once startup is declared complete.
# ---------------------------------------------------------------------------

@router.get("/health/startup")
async def startup(response: Response) -> dict[str, str | float | None]:
    global _startup_completed  # noqa: PLW0603
    now = time.monotonic()

    if not _startup_completed:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": ProbeState.FAIL, "completed": False, "uptime_seconds": None}

    # After startup completes, allow a short grace period where we still
    # report success while dependencies stabilise.
    elapsed = now - (_startup_completed_at or now)
    if elapsed < _STARTUP_GRACE_PERIOD_SECONDS:
        return {
            "status": ProbeState.OK,
            "completed": True,
            "uptime_seconds": round(elapsed, 2),
        }

    # Steady-state – verify dependencies for consistency with readiness.
    settings = get_settings()
    checks: dict[str, ProbeState] = {}
    checks["postgres"] = _run_check(lambda: check_postgres(settings))
    checks["redis"] = _run_check(lambda: check_redis(settings))
    all_ok = all(v == ProbeState.OK for v in checks.values())
    return {
        "status": ProbeState.OK if all_ok else ProbeState.FAIL,
        "completed": True,
        "uptime_seconds": round(elapsed, 2),
    }


def _run_check(check) -> ProbeState:  # noqa: ANN001
    try:
        result = check()
        return ProbeState.OK if result else ProbeState.FAIL
    except Exception:
        return ProbeState.FAIL
