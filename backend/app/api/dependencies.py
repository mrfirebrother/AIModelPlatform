from __future__ import annotations

from collections.abc import Generator
from typing import Any

from fastapi import Depends, HTTPException, Header, status
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import Settings, get_settings

_engine: Any = None
_session_factory: Any = None

_db_override: Any = None
_settings_override: Settings | None = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.postgres_url, pool_pre_ping=True)
    return _engine


def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=_get_engine(), autocommit=False, autoflush=False)
    return _session_factory


def set_db_override(override: Any) -> None:
    global _db_override
    _db_override = override


def get_db_override() -> Any:
    return _db_override


def set_settings_override(settings: Settings | None) -> None:
    global _settings_override
    _settings_override = settings


def get_effective_settings() -> Settings:
    if _settings_override is not None:
        return _settings_override
    return get_settings()


def get_db() -> Generator[Session, None, None]:
    override = get_db_override()
    if override is not None:
        yield from override()
        return
    factory = _get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def verify_api_key(
    api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_effective_settings),
) -> str:
    if api_key is None or api_key != settings.platform_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key
