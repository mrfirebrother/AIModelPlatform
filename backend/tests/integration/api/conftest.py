from __future__ import annotations

from typing import Any, Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.api import dependencies
from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.models import Base


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _make_verify_api_key(api_key_value: str):
    from fastapi import Header, HTTPException, status as http_status

    def _verify(api_key: str | None = Header(default=None, alias="X-API-Key")) -> str:
        if api_key is None or api_key != api_key_value:
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return api_key
    return _verify


@pytest.fixture()
def app(session):
    application = create_app(checks={})

    def _get_db_override():
        yield session

    from backend.app.api.dependencies import get_db, verify_api_key
    application.dependency_overrides[get_db] = _get_db_override
    application.dependency_overrides[verify_api_key] = _make_verify_api_key("test-api-key-123")
    yield application
    application.dependency_overrides.clear()
