"""
Модуль сервиса управления комнатами и клиентами.
"""

import logging
from uuid import UUID

from app.db.rooms_and_clients import Client, ClientAndRoomStorage, ClientFrame, ClientFrameRaw, Room
from app.services.video_processing import OneFaceMetricsAnalyzeResult

logger = logging.getLogger(__name__)


class RoomService:
    """
    Сервис управления комнатами и клиентами.

    Обеспечивает создание, поиск и удаление комнат и клиентов,
    а также безопасный доступ к данным через asyncio.Lock.
    """

    storage = ClientAndRoomStorage()

    def __init__(self):
        """Инициализирует сервис управления комнатами."""

    async def get_rooms(self) -> list[Room]:
        """
        Получает список всех активных комнат.

        Returns:
            list[Room]: Список объектов Room
        """
        return await self.storage.get_rooms()  # type: ignore[no-any-return]

    async def add_client(self, room_id: str, client: Client):
        """
        Добавляет клиента в комнату.

        Если комната не существует, она будет создана.

        Args:
            room_id: ID комнаты
            client: Объект клиента для добавления
        """
        await self.storage.add_client(room_id, client)

    async def get_client(self, room_id: str, client_id: UUID) -> Client:
        """
        Получает клиента из комнаты.

        Args:
            room_id: ID комнаты
            client_id: ID клиента

        Returns:
            Client: Объект клиента

        Raises:
            RoomNotFoundError: Если комната не найдена
            ClientNotFoundError: Если клиент не найден в комнате
        """
        return await self.storage.get_client(room_id, client_id)  # type: ignore[no-any-return]

    async def remove_client(self, room_id: str, client: Client):
        """
        Удаляет клиента из комнаты.

        Если в комнате больше нет клиентов, комната также удаляется.

        Args:
            room_id: ID комнаты
            client: Объект клиента для удаления
        """
        await self.close_client(client)
        await self.storage.remove_client(room_id, client)

    async def get_clients_in_room(self, room_id: str) -> list[Client]:
        """
        Получает всех клиентов в указанной комнате.

        Args:
            room_id: ID комнаты

        Returns:
            list[Client]: Список клиентов в комнате

        Raises:
            RoomNotFoundError: Если комната не найдена
        """
        return await self.storage.get_clients_in_room(room_id)  # type: ignore[no-any-return]

    async def close_client(self, client: Client):
        await self.storage.close_client(client)

    async def client_is_closed(self, client: Client) -> bool:
        return await self.storage.client_is_closed(client)  # type: ignore[no-any-return]

    async def send_frame(self, client: Client, src_b64: str, prc_b64: str, results: list[OneFaceMetricsAnalyzeResult]):
        await self.storage.send_frame(client, src_b64, prc_b64, results)

    async def get_frame(self, client: Client, timeout: float = 0.0) -> ClientFrame | None:
        return await self.storage.get_frame(client, timeout)  # type: ignore[no-any-return]

    async def get_frame_raw(self, client: Client, timeout: float = 0.0) -> ClientFrameRaw | None:
        return await self.storage.get_frame_raw(client, timeout)
