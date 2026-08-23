from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI

from .config import Settings, get_settings
from .db import default_health_checks

HealthCheck = Callable[[], bool | Awaitable[bool]]


async def _run_check(check: HealthCheck) -> bool:
    try:
        if inspect.iscoroutinefunction(check):
            result = check()
        else:
            result = await asyncio.to_thread(check)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
    except Exception:
        return False


def create_app(
    settings: Settings | None = None,
    checks: dict[str, HealthCheck] | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_checks = (
        default_health_checks(resolved_settings)
        if checks is None
        else checks
    )

    app = FastAPI(title="AI Model Platform API", version="0.1.0")

    from .api.routes import (
        alerts_router,
        backup_router,
        bindings_router,
        datasets_router,
        evaluations_router,
        gpu_router,
        health_router,
        infer_router,
        inference_router,
        models_router,
        operations_router,
        releases_router,
        resources_router,
        training_logs_router,
        training_router,
    )

    app.include_router(alerts_router)
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(bindings_router)
    app.include_router(datasets_router)
    app.include_router(training_router)
    app.include_router(training_logs_router)
    app.include_router(evaluations_router)
    app.include_router(releases_router)
    app.include_router(resources_router)
    app.include_router(infer_router)
    app.include_router(inference_router)
    app.include_router(gpu_router)
    app.include_router(operations_router)
    app.include_router(backup_router)

    @app.get("/health/live")
    async def liveness() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        results = {
            name: await _run_check(check)
            for name, check in resolved_checks.items()
        }
        if not results.get("postgres", False) or not results.get("redis", False):
            status = "unhealthy"
        elif not all(results.get(name, False) for name in ("gpu", "worker")):
            status = "degraded"
        else:
            status = "ready"
        return {"status": status, "checks": results}

    return app


app = create_app()
