from __future__ import annotations

from typing import Any, Generator

import pytest
from sqlalchemy import create_engine, event
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

    @event.listens_for(eng, "connect")
    def _enable_fk(dbapi_conn, _record):
        # 强制外键约束，与生产 Postgres 行为一致（否则 FK 顺序类 bug 测不出来）
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

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
    # ui_password=None: 仓库根 .env 现在配置了 UI_PASSWORD，测试要显式关掉
    # 界面密码门，否则所有不带 X-UI-Password 的请求都会被拦成 401。
    application = create_app(checks={}, settings=Settings(ui_password=None))

    def _get_db_override():
        yield session

    from backend.app.api.dependencies import get_db, verify_api_key
    application.dependency_overrides[get_db] = _get_db_override
    application.dependency_overrides[verify_api_key] = _make_verify_api_key("test-api-key-123")
    yield application
    application.dependency_overrides.clear()
