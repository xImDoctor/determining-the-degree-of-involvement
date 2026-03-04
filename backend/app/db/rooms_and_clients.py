import json
import logging
from dataclasses import asdict, dataclass
from uuid import UUID

import cv2
import redis.asyncio as redis
from dacite import from_dict

from app.core.config import settings
from app.services.video_processing import OneFaceMetricsAnalyzeResult

logger = logging.getLogger(__name__)


def _result_to_dict(result: OneFaceMetricsAnalyzeResult) -> dict:
    """Convert OneFaceMetricsAnalyzeResult to dict, excluding ear_history for smaller payload."""
    data = asdict(result)
    if data.get("ear") and "ear_history" in data["ear"]:
        del data["ear"]["ear_history"]
    return data


def _convert_tuples(data: dict) -> None:
    """Recursively convert lists to tuples for fields that require tuples (bbox, vectors)."""
    if "bbox" in data and isinstance(data["bbox"], list):
        data["bbox"] = tuple(data["bbox"])
    if "head_pose" in data and data["head_pose"]:
        hp = data["head_pose"]
        if "rotation_vec" in hp and isinstance(hp["rotation_vec"], list):
            hp["rotation_vec"] = tuple(hp["rotation_vec"])
        if "translation_vec" in hp and isinstance(hp["translation_vec"], list):
            hp["translation_vec"] = tuple(hp["translation_vec"])


class RoomNotFoundError(Exception):
    """Исключение, выбрасываемое при отсутствии комнаты."""

    pass


class ClientNotFoundError(Exception):
    """Исключение, выбрасываемое при отсутствии клиента."""

    pass


@dataclass
class Client:
    """
    Представляет клиента в комнате для видеопотока.

    Attributes:
        id_: Уникальный идентификатор клиента
        name: Имя клиента
    """

    id_: UUID
    name: str
    source_closed: bool = False


@dataclass
class Room:
    """
    Представляет комнату для группового видеопотока.

    Attributes:
        id_: Уникальный идентификатор комнаты
        clients: Словарь клиентов в комнате
    """

    id_: str
    clients: dict[UUID, Client]


@dataclass
class ClientFrame:
    src: cv2.typing.MatLike
    prc: cv2.typing.MatLike
    results: list[OneFaceMetricsAnalyzeResult]


@dataclass
class ClientFrameRaw:
    src_b64: str
    prc_b64: str
    results: list[OneFaceMetricsAnalyzeResult]


class ClientAndRoomStorage:
    def __init__(self):
        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password if settings.redis_password else None,
            socket_timeout=settings.redis_timeout,
        )
        self.pubsubs: dict[str, redis.client.PubSub] = {}  # client id : pubsub
        # rooms -> {roomId, ...}
        # room:roomId -> {clientId, ...}
        # client:clientId {name:, source_closed:}
        # PubSup
        # client_stream:clientId -> {metrics:JSON, src_frame:base64, prc_frame:base64}

    async def get_rooms(self) -> list[Room]:
        """
        Получает список всех активных комнат.

        Returns:
            list[Room]: Список объектов Room
        """
        logger.debug("Fetching all rooms")
        rooms: list[Room] = []
        room_ids = await self.redis.smembers("rooms")
        for room_id in room_ids:
            room_id_str = room_id.decode() if isinstance(room_id, bytes) else room_id
            client_ids = await self.redis.smembers(f"room:{room_id_str}")
            clients: dict[UUID, Client] = {}
            for client_id in client_ids:
                client_id_str = client_id.decode() if isinstance(client_id, bytes) else client_id
                client_uuid = UUID(client_id_str)
                client_data: dict[bytes, bytes] = await self.redis.hgetall(f"client:{client_id_str}")
                clients[client_uuid] = Client(
                    client_uuid, client_data[b"name"].decode(), client_data[b"source_closed"].decode() == "True"
                )
            rooms.append(Room(room_id_str, clients))
        logger.debug(f"Found {len(rooms)} rooms")
        return rooms

    async def add_client(self, room_id: str, client: Client):
        """
        Добавляет клиента в комнату.

        Если комната не существует, она будет создана.

        Args:
            room_id: ID комнаты
            client: Объект клиента для добавления
        """
        logger.info(f"Adding client {client.id_} to room {room_id}")
        await self.redis.sadd("rooms", room_id)
        await self.redis.sadd(f"room:{room_id}", str(client.id_))
        await self.redis.hset(
            f"client:{client.id_}", mapping={"name": client.name, "source_closed": str(client.source_closed)}
        )
        logger.info(f"Client {client.id_} added to room {room_id}")

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
        logger.debug(f"Getting client {client_id} from room {room_id}")
        room_members = await self.redis.smembers("rooms")
        room_id_bytes = room_id.encode() if isinstance(room_id, str) else room_id
        if room_id_bytes not in room_members:
            logger.warning(f"Room {room_id} not found")
            raise RoomNotFoundError
        client_id_str = str(client_id)
        room_key = f"room:{room_id}"
        room_client_members = await self.redis.smembers(room_key)
        client_id_bytes = client_id_str.encode() if isinstance(client_id_str, str) else client_id_str
        if client_id_bytes not in room_client_members:
            logger.warning(f"Client {client_id} not found in room {room_id}")
            raise ClientNotFoundError
        client_data: dict[bytes, bytes] = await self.redis.hgetall(f"client:{client_id_str}")
        return Client(client_id, client_data[b"name"].decode(), client_data[b"source_closed"].decode() == "True")

    async def remove_client(self, room_id: str, client: Client):
        """
        Удаляет клиента из комнаты.

        Если в комнате больше нет клиентов, комната также удаляется.

        Args:
            room_id: ID комнаты
            client: Объект клиента для удаления
        """
        logger.info(f"Removing client {client.id_} from room {room_id}")
        room_members = await self.redis.smembers("rooms")
        room_id_bytes = room_id.encode() if isinstance(room_id, str) else room_id
        if room_id_bytes not in room_members:
            logger.warning(f"Room {room_id} not found when removing client {client.id_}")
            return
        client_id_str = str(client.id_)
        room_client_members = await self.redis.smembers(f"room:{room_id}")
        client_id_bytes = client_id_str.encode() if isinstance(client_id_str, str) else client_id_str
        if client_id_bytes not in room_client_members:
            logger.warning(f"Client {client_id_str} not found in room {room_id}")
            return
        await self.redis.delete(f"client:{client_id_str}")
        await self.redis.srem(f"room:{room_id}", client_id_str)
        if client_id_str in self.pubsubs:
            del self.pubsubs[client_id_str]
        # Delete room if no more clients
        remaining_clients = await self.redis.smembers(f"room:{room_id}")
        if not remaining_clients:
            await self.redis.srem("rooms", room_id)
            await self.redis.delete(f"room:{room_id}")
            logger.info(f"Room {room_id} deleted (no more clients)")
        logger.info(f"Client {client.id_} removed from room {room_id}")

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
        logger.debug(f"Getting clients in room {room_id}")
        room_members = await self.redis.smembers("rooms")
        room_id_bytes = room_id.encode() if isinstance(room_id, str) else room_id
        if room_id_bytes not in room_members:
            logger.warning(f"Room {room_id} not found")
            raise RoomNotFoundError
        client_ids = await self.redis.smembers(f"room:{room_id}")
        clients: list[Client] = []
        for client_id in client_ids:
            client_id_str = client_id.decode() if isinstance(client_id, bytes) else client_id
            client_data: dict[bytes, bytes] = await self.redis.hgetall(f"client:{client_id_str}")
            clients.append(
                Client(
                    UUID(client_id_str), client_data[b"name"].decode(), client_data[b"source_closed"].decode() == "True"
                )
            )
        logger.debug(f"Found {len(clients)} clients in room {room_id}")
        return clients

    async def close_client(self, client: Client) -> None:
        """
        Закрывает клиентский поток, отмечая его как завершённый в Redis.

        Args:
            client: Объект клиента для закрытия
        """
        logger.debug(f"Closing client {client.id_}")
        await self.redis.hset(f"client:{client.id_}", "source_closed", "True")

    async def client_is_closed(self, client: Client) -> bool:
        """
        Проверяет, закрыт ли клиентский поток.

        Args:
            client: Объект клиента для проверки

        Returns:
            bool: True если поток закрыт или клиент не найден, иначе False
        """
        logger.debug(f"Checking if client {client.id_} is closed")
        res = await self.redis.hgetall(f"client:{client.id_}")
        if not res:
            logger.debug(f"Client {client.id_} not found, treating as closed")
            return True
        source_closed = res.get(b"source_closed")
        is_closed = source_closed == b"True" if source_closed else False
        logger.debug(f"Client {client.id_} is_closed={is_closed}")
        return is_closed

    async def send_frame(
        self, client: Client, src_b64: str, prc_b64: str, results: list[OneFaceMetricsAnalyzeResult]
    ) -> None:
        """
        Отправляет кадр клиенту через Redis Pub/Sub.

        Args:
            client: Объект клиента
            src_b64: Исходное изображение в base64
            prc_b64: Обработанное изображение в base64
            results: Результаты анализа для каждого лица
        """
        logger.debug(f"Sending frame to client {client.id_} (results count: {len(results)})")
        payload = {
            "src": src_b64,
            "prc": prc_b64,
            "result": [_result_to_dict(r) for r in results],
        }
        await self.redis.publish(f"client_stream:{client.id_}", json.dumps(payload))

    async def get_frame_raw(self, client: Client, timeout: float = 0.0) -> ClientFrameRaw | None:
        """
        Получает кадр для клиента из Redis Pub/Sub.

        Args:
            client: Объект клиента
            timeout: Таймаут ожидания сообщения в секундах

        Returns:
            ClientFrameRaw с данными кадра или None если сообщение не получено
        """
        logger.debug(f"Getting frame for client {client.id_} (timeout: {timeout:.2f})")
        if str(client.id_) not in self.pubsubs:
            self.pubsubs[str(client.id_)] = self.redis.pubsub()
            await self.pubsubs[str(client.id_)].subscribe(f"client_stream:{client.id_}")
        pubsub = self.pubsubs[str(client.id_)]
        message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=timeout)
        if not message or message["type"] != "message":
            return None
        data = json.loads(message["data"].decode())
        for item in data["result"]:
            _convert_tuples(item)
        logger.debug(f"Frame received for client {client.id_} (results count: {len(data['result'])})")
        return ClientFrameRaw(
            data["src"],
            data["prc"],
            [from_dict(OneFaceMetricsAnalyzeResult, item) for item in data["result"]],
        )

    async def flushdb(self) -> None:
        """
        Очищает все ключи из текущей базы данных Redis.
        """
        await self.redis.flushdb()
