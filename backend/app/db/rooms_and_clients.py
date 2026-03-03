import base64
import logging
from dataclasses import asdict, dataclass
from json import JSONDecoder, JSONEncoder
from uuid import UUID

import cv2
import numpy as np
import redis.asyncio as redis
from dacite import from_dict

from app.core.config import settings
from app.services.video_processing import OneFaceMetricsAnalyzeResult

logger = logging.getLogger(__name__)


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

    async def add_client(self, room_id: str, client: Client) -> None:
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

    async def remove_client(self, room_id: str, client: Client) -> None:
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

    async def close_client(self, client: Client):
        logger.debug(f"Closing client {client.id_}")
        await self.redis.hset(f"client:{client.id_}", "source_closed", "True")

    async def client_is_closed(self, client: Client) -> bool:
        logger.debug(f"Checking if client {client.id_} is closed")
        res = await self.redis.hgetall(f"client:{client.id_}")
        if not res:
            logger.debug(f"Client {client.id_} not found, treating as closed")
            return True
        is_closed = res[b"source_closed"] == b"True"
        logger.debug(f"Client {client.id_} is_closed={is_closed}")
        return is_closed

    async def send_frame(self, client: Client, src_b64: str, prc_b64: str, results: list[OneFaceMetricsAnalyzeResult]):
        logger.debug(f"Sending frame to client {client.id_} (results count: {len(results)})")
        json = {
            "src": src_b64,
            "prc": prc_b64,
            "result": list(map(asdict, results)),
        }
        await self.redis.publish(f"client_stream:{client.id_}", JSONEncoder().encode(json))

    async def get_frame(self, client: Client, timeout: float = 0.0) -> ClientFrame | None:
        logger.debug(f"Getting frame for client {client.id_} (timeout: {timeout:.2f})")
        if str(client.id_) not in self.pubsubs:
            self.pubsubs[str(client.id_)] = self.redis.pubsub()
            await self.pubsubs[str(client.id_)].subscribe(f"client_stream:{client.id_}")
        pubsub = self.pubsubs[str(client.id_)]
        message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=timeout)
        if not message or message["type"] != "message":
            return None
        json = JSONDecoder().decode(message["data"].decode())
        for item in json["result"]:
            item["bbox"] = tuple(item["bbox"])
            if item["head_pose"]:
                item["head_pose"]["rotation_vec"] = tuple(item["head_pose"]["rotation_vec"])
                item["head_pose"]["translation_vec"] = tuple(item["head_pose"]["translation_vec"])
        logger.debug(f"Frame received for client {client.id_} (results count: {len(json['result'])})")
        return ClientFrame(
            self._base64_to_img(json["src"]),
            self._base64_to_img(json["prc"]),
            [from_dict(OneFaceMetricsAnalyzeResult, item) for item in json["result"]],
        )

    @staticmethod
    def _img_to_base64(img: cv2.typing.MatLike) -> str:
        _, buffer = cv2.imencode(".jpg", img)
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def _base64_to_img(image_b64: str) -> cv2.typing.MatLike:
        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            msg = "Failed to decode image"
            raise ValueError(msg)
        return img
