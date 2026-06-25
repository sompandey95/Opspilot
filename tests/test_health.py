from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
def app():
    """App with mocked db/redis lifecycle so no real services needed."""
    with (
        patch("app.main.init_db", new_callable=AsyncMock),
        patch("app.main.init_redis", new_callable=AsyncMock),
        patch("app.main.close_db", new_callable=AsyncMock),
        patch("app.main.close_redis", new_callable=AsyncMock),
    ):
        yield create_app()


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_app_starts(client):
    """Lifespan completes without errors when db/redis are mocked."""
    resp = await client.get("/")
    assert resp.status_code == 200


async def test_root_returns_service_info(client):
    resp = await client.get("/")
    assert resp.json() == {"service": "opspilot", "version": "0.1.0"}
