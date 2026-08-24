from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.config import get_settings


_worker_engine = None
_worker_session_factory = None


def _get_worker_engine():
    global _worker_engine
    if _worker_engine is None:
        settings = get_settings()
        _worker_engine = create_engine(settings.postgres_url, pool_pre_ping=True)
    return _worker_engine


def _get_worker_session_factory():
    global _worker_session_factory
    if _worker_session_factory is None:
        _worker_session_factory = sessionmaker(
            bind=_get_worker_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _worker_session_factory


def set_worker_engine_override(engine) -> None:
    global _worker_engine, _worker_session_factory
    _worker_engine = engine
    _worker_session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )


@contextmanager
def worker_session() -> Generator[Session, None, None]:
    factory = _get_worker_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_worker_session_factory() -> None:
    global _worker_engine, _worker_session_factory
    if _worker_engine is not None:
        _worker_engine.dispose()
    _worker_engine = None
    _worker_session_factory = None
