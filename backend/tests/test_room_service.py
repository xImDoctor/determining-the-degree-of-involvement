from uuid import uuid4

import pytest

from app.db.rooms_and_clients import Client, ClientNotFoundError, RoomNotFoundError
from app.services.room import RoomService


class TestRoomService:
    @pytest.mark.asyncio
    async def test_add_client_creates_room(self, room_service, client):
        room_id = "test_room"
        await room_service.add_client(room_id, client)

        rooms = await room_service.get_rooms()
        assert len(rooms) == 1
        assert rooms[0].id_ == room_id
        assert client.id_ in rooms[0].clients

    @pytest.mark.asyncio
    async def test_add_multiple_clients_to_same_room(self, room_service):
        room_id = "test_room"
        client1 = Client(id_=uuid4(), name="client1")
        client2 = Client(id_=uuid4(), name="client2")

        await room_service.add_client(room_id, client1)
        await room_service.add_client(room_id, client2)

        rooms = await room_service.get_rooms()
        assert len(rooms) == 1
        assert len(rooms[0].clients) == 2

    @pytest.mark.asyncio
    async def test_add_clients_to_different_rooms(self, room_service):
        room1 = "room1"
        room2 = "room2"
        client1 = Client(id_=uuid4(), name="client1")
        client2 = Client(id_=uuid4(), name="client2")

        await room_service.add_client(room1, client1)
        await room_service.add_client(room2, client2)

        rooms = await room_service.get_rooms()
        assert len(rooms) == 2

    @pytest.mark.asyncio
    async def test_get_clients_in_room(self, room_service):
        room_id = "test_room"
        client1 = Client(id_=uuid4(), name="client1")
        client2 = Client(id_=uuid4(), name="client2")

        await room_service.add_client(room_id, client1)
        await room_service.add_client(room_id, client2)

        clients = await room_service.get_clients_in_room(room_id)
        assert len(clients) == 2

    @pytest.mark.asyncio
    async def test_get_clients_in_nonexistent_room_raises_error(self, room_service):
        with pytest.raises(RoomNotFoundError):
            await room_service.get_clients_in_room("nonexistent")

    @pytest.mark.asyncio
    async def test_get_client_success(self, room_service, client):
        room_id = "test_room"
        await room_service.add_client(room_id, client)

        found_client = await room_service.get_client(room_id, client.id_)
        assert found_client.id_ == client.id_
        assert found_client.name == client.name

    @pytest.mark.asyncio
    async def test_get_client_nonexistent_room_raises_error(self, room_service, client):
        with pytest.raises(RoomNotFoundError):
            await room_service.get_client("nonexistent", client.id_)

    @pytest.mark.asyncio
    async def test_get_client_nonexistent_client_raises_error(self, room_service):
        room_id = "test_room"
        await room_service.add_client(room_id, Client(id_=uuid4(), name="existing"))

        with pytest.raises(ClientNotFoundError):
            await room_service.get_client(room_id, uuid4())

    @pytest.mark.asyncio
    async def test_remove_client(self, room_service, client):
        room_id = "test_room"
        await room_service.add_client(room_id, client)

        await room_service.remove_client(room_id, client)

        rooms = await room_service.get_rooms()
        assert len(rooms) == 0

    @pytest.mark.asyncio
    async def test_remove_client_from_nonexistent_room(self, room_service, client):
        await room_service.remove_client("nonexistent", client)

    @pytest.mark.asyncio
    async def test_remove_client_updates_room_state(self, room_service):
        room_id = "test_room"
        client1 = Client(id_=uuid4(), name="client1")
        client2 = Client(id_=uuid4(), name="client2")

        await room_service.add_client(room_id, client1)
        await room_service.add_client(room_id, client2)

        await room_service.remove_client(room_id, client1)

        rooms = await room_service.get_rooms()
        assert len(rooms) == 1
        assert client2.id_ in rooms[0].clients

    @pytest.mark.asyncio
    async def test_room_deleted_when_last_client_removed(self, room_service):
        room_id = "test_room"
        client = Client(id_=uuid4(), name="client")
        await room_service.add_client(room_id, client)

        await room_service.remove_client(room_id, client)

        rooms = await room_service.get_rooms()
        assert len(rooms) == 0

    @pytest.mark.asyncio
    async def test_get_rooms_empty_initially(self, room_service):
        rooms = await room_service.get_rooms()
        assert len(rooms) == 0
