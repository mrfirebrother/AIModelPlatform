import sys
import threading
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.config import Settings
from backend.app.db import check_gpu, check_postgres
from backend.app.main import create_app


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("postgres", "redis", "gpu", "worker", "expected"),
    [
        (True, True, True, True, "ready"),
        (True, True, False, True, "degraded"),
        (True, True, True, False, "degraded"),
        (False, True, True, True, "unhealthy"),
        (True, False, True, True, "unhealthy"),
    ],
)
async def test_health_reports_dependency_readiness(
    postgres: bool,
    redis: bool,
    gpu: bool,
    worker: bool,
    expected: str,
) -> None:
    app = create_app(
        checks={
            "postgres": lambda: postgres,
            "redis": lambda: redis,
            "gpu": lambda: gpu,
            "worker": lambda: worker,
        }
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == expected
    assert body["checks"] == {
        "postgres": postgres,
        "redis": redis,
        "gpu": gpu,
        "worker": worker,
    }


@pytest.mark.anyio
async def test_empty_injected_checks_are_not_replaced_by_defaults() -> None:
    app = create_app(checks={})

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.json() == {"status": "unhealthy", "checks": {}}


@pytest.mark.anyio
async def test_liveness_is_independent_of_readiness() -> None:
    app = create_app(
        checks={
            "postgres": lambda: False,
            "redis": lambda: False,
            "gpu": lambda: False,
            "worker": lambda: False,
        }
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


@pytest.mark.anyio
async def test_sync_checks_run_without_blocking_async_endpoint() -> None:
    main_thread = threading.get_ident()
    check_threads: list[int] = []

    def sync_check() -> bool:
        check_threads.append(threading.get_ident())
        return True

    app = create_app(
        checks={
            "postgres": sync_check,
            "redis": sync_check,
            "gpu": sync_check,
            "worker": sync_check,
        }
    )

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert check_threads
    assert all(thread_id != main_thread for thread_id in check_threads)


def test_gpu_ready_override_skips_local_cuda_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    class TorchCuda:
        @staticmethod
        def is_available() -> bool:
            raise AssertionError("CUDA probe should be skipped by override")

    monkeypatched_torch = SimpleNamespace(cuda=TorchCuda())
    monkeypatch.setitem(sys.modules, "torch", monkeypatched_torch)
    assert check_gpu(Settings(GPU_READY=True)) is True


def test_gpu_check_falls_back_to_cuda_when_override_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(device: int) -> object:
            assert device == 0
            return object()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=TorchCuda()))
    assert check_gpu(Settings(GPU_READY=False)) is True


def test_scheduler_tasks_use_default_queue() -> None:
    from backend.app.workers.celery_app import celery_app

    assert celery_app.conf.task_routes["platform.scheduler.*"]["queue"] == "default"


def test_postgres_check_converts_sqlalchemy_url_for_psycopg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    class Cursor:
        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def execute(self, query: str) -> None:
            assert query == "SELECT 1"

        def fetchone(self) -> tuple[int]:
            return (1,)

    class Connection:
        def __enter__(self) -> "Connection":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def cursor(self) -> Cursor:
            return Cursor()

    def connect(dsn: str, connect_timeout: int, options: str) -> Connection:
        assert connect_timeout == 1
        assert options == "-c statement_timeout=1000"
        seen.append(dsn)
        if "+psycopg" in dsn:
            raise ValueError("psycopg does not accept SQLAlchemy driver names")
        return Connection()

    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=connect))

    assert check_postgres(
        Settings(
            POSTGRES_URL=(
                "postgresql+psycopg://ai_platform:secret@postgres:5432/ai_platform"
            )
        )
    ) is True
    assert seen == [
        "postgresql://ai_platform:secret@postgres:5432/ai_platform"
    ]


def test_postgres_check_returns_false_for_malformed_url() -> None:
    assert check_postgres(Settings(POSTGRES_URL="not a database URL")) is False


def test_settings_builds_encoded_postgres_url_from_components() -> None:
    settings = Settings(
        POSTGRES_HOST="db",
        POSTGRES_PORT=5433,
        POSTGRES_DB="custom",
        POSTGRES_USER="svc",
        POSTGRES_PASSWORD="p@ss:word",
    )

    assert settings.postgres_url == (
        "postgresql+psycopg://svc:p%40ss%3Aword@db:5433/custom"
    )
