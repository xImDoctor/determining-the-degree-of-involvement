from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.db.rooms_and_clients import RoomNotFoundError
from app.services.room import RoomService


def get_mock_room_service():
    return MagicMock(spec=RoomService)


@pytest.fixture
def mock_room_service():
    mock_service = MagicMock(spec=RoomService)
    mock_service.get_rooms = AsyncMock(return_value=[])
    mock_service.get_clients_in_room = AsyncMock(return_value=[])
    return mock_service


@pytest.mark.asyncio
async def test_health_check():
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == settings.app_version


@pytest.mark.asyncio
async def test_get_rooms_empty():
    from app.main import app

    mock_service = MagicMock(spec=RoomService)
    mock_service.get_rooms = AsyncMock(return_value=[])

    def override_get_room_service():
        return mock_service

    app.dependency_overrides[RoomService] = override_get_room_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rooms")
        assert response.status_code == 200
        assert response.json() == []

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_rooms_with_data():
    from app.db.rooms_and_clients import Room
    from app.main import app
    from app.services.room import RoomService

    mock_service = MagicMock(spec=RoomService)
    mock_room = Room(id_="test_room", clients={})
    mock_service.get_rooms = AsyncMock(return_value=[mock_room])

    def override_get_room_service():
        return mock_service

    app.dependency_overrides[RoomService] = override_get_room_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rooms")
        assert response.status_code == 200
        assert response.json() == ["test_room"]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_clients_in_room_success():
    from app.db.rooms_and_clients import Client as RoomClient
    from app.main import app
    from app.services.room import RoomService

    mock_service = MagicMock(spec=RoomService)
    test_client = RoomClient(id_=uuid4(), name="test_client", room_id="test_room")
    mock_service.get_clients_in_room = AsyncMock(return_value=[test_client])

    def override_get_room_service():
        return mock_service

    app.dependency_overrides[RoomService] = override_get_room_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rooms/test_room/clients")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "test_client"
        assert "id_" in data[0]

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_clients_in_room_not_found():
    from app.main import app
    from app.services.room import RoomService

    mock_service = MagicMock(spec=RoomService)
    mock_service.get_clients_in_room = AsyncMock(side_effect=RoomNotFoundError("Room not found"))

    def override_get_room_service():
        return mock_service

    app.dependency_overrides[RoomService] = override_get_room_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rooms/nonexistent/clients")
        assert response.status_code == 404

    app.dependency_overrides.clear()
