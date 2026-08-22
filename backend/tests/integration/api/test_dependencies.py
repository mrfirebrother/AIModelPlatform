from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.app.api.dependencies import verify_api_key, set_settings_override
from backend.app.config import Settings


def test_valid_api_key_passes():
    settings = Settings(PLATFORM_API_KEY="test-key-123")
    result = verify_api_key(api_key="test-key-123", settings=settings)
    assert result == "test-key-123"


def test_invalid_api_key_raises():
    settings = Settings(PLATFORM_API_KEY="test-key-123")
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(api_key="wrong-key", settings=settings)
    assert exc_info.value.status_code == 401


def test_missing_api_key_raises():
    settings = Settings(PLATFORM_API_KEY="test-key-123")
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(api_key=None, settings=settings)
    assert exc_info.value.status_code == 401


def test_api_key_only_from_header():
    set_settings_override(Settings(PLATFORM_API_KEY="test-key"))
    try:
        app = FastAPI()

        @app.get("/protected")
        async def protected(key: str = Depends(verify_api_key)):
            return {"key": key}

        client = TestClient(app)
        response = client.get("/protected", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        assert response.json() == {"key": "test-key"}
    finally:
        set_settings_override(None)
