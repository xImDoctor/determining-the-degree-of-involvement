"""
Модуль сервиса управления комнатами и клиентами.
"""

import logging
from uuid import UUID

from app.db.rooms_and_clients import Client, ClientAndRoomStorage, ClientFrameRaw, Room
from app.services.video_processing import OneFaceMetricsAnalyzeResult

logger = logging.getLogger(__name__)


class RoomService:
    """
    Сервис управления комнатами и клиентами.

    Обеспечивает создание, поиск и удаление комнат и клиентов.
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

    async def close_client(self, client: Client) -> None:
        """
        Закрывает клиентский поток (отмечает как завершённый).

        Args:
            client: Объект клиента для закрытия
        """
        await self.storage.close_client(client)

    async def client_is_closed(self, client: Client) -> bool:
        """
        Проверяет, закрыт ли клиентский поток.

        Args:
            client: Объект клиента для проверки

        Returns:
            bool: True если поток закрыт, иначе False
        """
        return await self.storage.client_is_closed(client)  # type: ignore[no-any-return]

    async def send_frame(
        self, client: Client, src_b64: str, prc_b64: str, results: list[OneFaceMetricsAnalyzeResult]
    ) -> None:
        """
        Отправляет кадр клиенту через Pub/Sub.

        Args:
            client: Объект клиента
            src_b64: Исходное изображение в base64
            prc_b64: Обработанное изображение в base64
            results: Результаты анализа для каждого лица
        """
        await self.storage.send_frame(client, src_b64, prc_b64, results)

    async def get_frame_raw(self, client: Client, timeout: float = 0.0) -> ClientFrameRaw | None:
        """
        Получает кадр для клиента из Pub/Sub.

        Args:
            client: Объект клиента
            timeout: Таймаут ожидания кадра в секундах

        Returns:
            ClientFrameRaw с данными кадра или None если кадр недоступен
        """
        return await self.storage.get_frame_raw(client, timeout)
