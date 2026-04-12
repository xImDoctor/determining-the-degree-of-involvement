from uuid import UUID

from pydantic import BaseModel


class ClientInfo(BaseModel):
    """Информация о клиенте (для списка клиентов комнаты)."""

    name: str
    id_: UUID


class ClientSchema(BaseModel):
    """Полное представление клиента"""

    id_: UUID
    name: str
    room_id: str
    source_closed: bool = False


class RoomSchema(BaseModel):
    """Представление комнаты с клиентами"""

    id_: str
    clients: dict[UUID, ClientSchema]


class RoomNotFoundResponse(BaseModel):
    """Ответ при ненайденной комнате (HTTP 404)"""

    detail: str
