from __future__ import annotations

from typing import Any

from sqlalchemy.engine import make_url

from .config import Settings


def check_postgres(settings: Settings) -> bool:
    try:
        import psycopg

        psycopg_dsn = make_url(settings.postgres_url).set(
            drivername="postgresql"
        ).render_as_string(hide_password=False)
        with psycopg.connect(
            psycopg_dsn,
            connect_timeout=1,
            options="-c statement_timeout=1000",
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                return cursor.fetchone() == (1,)
    except Exception:
        return False


def check_redis(settings: Settings) -> bool:
    try:
        import redis

        client = redis.Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        return bool(client.ping())
    except Exception:
        return False


def check_gpu(settings: Settings) -> bool:
    if settings.gpu_ready:
        return True
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        if settings.gpu_device.lower() == "cpu":
            return False
        torch.cuda.get_device_properties(int(settings.gpu_device))
        return True
    except (ImportError, RuntimeError, ValueError, TypeError, AttributeError):
        return False


def check_worker(settings: Settings) -> bool:
    """Use an explicit readiness flag until a real worker heartbeat is available."""
    return settings.worker_ready


def default_health_checks(settings: Settings) -> dict[str, Any]:
    return {
        "postgres": lambda: check_postgres(settings),
        "redis": lambda: check_redis(settings),
        "gpu": lambda: check_gpu(settings),
        "worker": lambda: check_worker(settings),
    }
