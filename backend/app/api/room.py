"""
Модуль REST эндпоинтов для управления комнатами.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.db.rooms_and_clients import RoomNotFoundError
from app.schemas.room import ClientInfo, RoomNotFoundResponse
from app.services.room import RoomService

logger = logging.getLogger(__name__)

room_router = APIRouter()


@room_router.get("/rooms", response_model=list[str])
async def get_rooms(room_service: Annotated[RoomService, Depends()]):
    """
    Получение списка всех активных комнат.

    Args:
        room_service: Сервис управления комнатами

    Returns:
        list: Список ID активных комнат
    """
    rooms = await room_service.get_rooms()
    logger.debug(f"Retrieved {len(rooms)} rooms")
    return [room.id_ for room in rooms]


@room_router.get(
    "/rooms/{room_id}/clients",
    response_model=list[ClientInfo],
    responses={404: {"model": RoomNotFoundResponse}},
)
async def get_clients(room_id: str, room_service: Annotated[RoomService, Depends()]):
    """
    Получение списка клиентов в указанной комнате.

    Args:
        room_id: ID комнаты
        room_service: Сервис управления комнатами

    Returns:
        list[ClientInfo]: Список клиентов с именем и ID

    Raises:
        HTTPException 404: Если комната не найдена
    """
    try:
        clients = await room_service.get_clients_in_room(room_id)
        logger.debug(f"Retrieved {len(clients)} clients from room {room_id}")
        return [ClientInfo(name=item.name, id_=item.id_) for item in clients]
    except RoomNotFoundError:
        logger.warning(f"Room {room_id} not found")
        raise HTTPException(status_code=404, detail=f"Room {room_id} not found")
