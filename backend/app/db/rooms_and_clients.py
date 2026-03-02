import base64
from dataclasses import asdict, dataclass
from json import JSONDecoder, JSONEncoder
from uuid import UUID

import cv2
import numpy as np
import redis
from dacite import from_dict

from app.core.config import settings
from app.services.video_processing import OneFaceMetricsAnalyzeResult


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
        self.pubsubs: dict[str, redis.client.PubSub] = {} # client id : pubsub
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
        rooms: list[Room] = []
        for room_id in self.redis.smembers('rooms'):
            room_id: str
            client_ids: list[str] = self.redis.smembers(f'room:{room_id}')
            clients: dict[UUID, Client] = {}
            for client_id in client_ids:
                client_uuid = UUID(client_id)
                client_data: dict[str, str] = self.redis.hgetall(f'client:{client_id}')
                clients[client_uuid] = Client(client_uuid, client_data['name'], client_data['source_closed'] == 'True')
            rooms.append(Room(room_id, clients))
        return rooms


    async def add_client(self, room_id: str, client: Client) -> None:
        """
        Добавляет клиента в комнату.

        Если комната не существует, она будет создана.

        Args:
            room_id: ID комнаты
            client: Объект клиента для добавления
        """
        if room_id not in self.redis.smembers('rooms'):
            self.redis.sadd('rooms', room_id)
        self.redis.sadd(f'room:{room_id}', str(client.id_))
        self.redis.hset(f'client:{client.id_}',
                            mapping={'name': client.name, 'source_closed': str(client.source_closed)})
        self.pubsubs[str(client.id_)] = self.redis.pubsub()
        self.pubsubs[str(client.id_)].subscribe(f'client_stream:{client.id_}')


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
        if room_id.encode() not in self.redis.smembers('rooms'):
            raise RoomNotFoundError
        if str(client_id).encode() not in self.redis.smembers(f'room:{room_id}'):
            raise ClientNotFoundError
        client_data: dict[bytes, bytes] = self.redis.hgetall(f'client:{client_id}')
        return Client(client_id, client_data[b'name'].decode(), client_data[b'source_closed'].decode() == 'True')

    async def remove_client(self, room_id: str, client: Client) -> None:
        """
        Удаляет клиента из комнаты.

        Если в комнате больше нет клиентов, комната также удаляется.

        Args:
            room_id: ID комнаты
            client: Объект клиента для удаления
        """
        if room_id not in self.redis.smembers('rooms'):
            return
        if str(client.id_) not in self.redis.get(f'room:{room_id}'):
            return
        self.redis.delete(f'client:{client.id_}')
        self.redis.srem(f'room:{room_id}', str(client.id_))
        if str(client.id_) in self.pubsubs:
            del self.pubsubs[str(client.id_)]


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
        client_ids = self.redis.smembers(f'room:{room_id}')
        clients: list[Client] = []
        for client_id in client_ids:
            client_id: bytes
            client_data: dict[bytes, bytes] = self.redis.hgetall(f'client:{client_id.decode()}')
            clients.append(Client(UUID(client_id.decode()), client_data[b'name'].decode(),
                                  client_data[b'source_closed'].decode() == 'True'))
        return clients

    async def close_client(self, client: Client):
        self.redis.hset(f'client:{client.id_}', 'source_closed', 'True')

    async def client_is_close(self, client: Client):
        res = self.redis.hgetall(f'client:{client.id_}')
        if res is None:
            return True
        return res[b'source_closed'] == b'True'

    async def send_frame(self, client: Client, frame: ClientFrame):
        json = {
            'src': await self._img_to_base64(frame.src),
            'prc': await self._img_to_base64(frame.prc),
            'result': list(map(asdict, frame.results))
        }
        self.redis.publish(f'client_stream:{client.id_}', JSONEncoder().encode(json))

    async def get_frame(self, client: Client, timeout: float = 0.0) -> ClientFrame | None:
        if str(client.id_) not in self.pubsubs:
            self.pubsubs[str(client.id_)] = self.redis.pubsub()
            self.pubsubs[str(client.id_)].subscribe(f'client_stream:{client.id_}')
        pubsub = self.pubsubs[str(client.id_)]
        message = pubsub.get_message(ignore_subscribe_messages=True, timeout=timeout)
        if not message or message['type'] != 'message':
            return None
        json = JSONDecoder().decode(message['data'].decode())
        for item in json['result']:
            item['bbox'] = tuple(item['bbox'])
            if item['head_pose']:
                item['head_pose']['rotation_vec'] = tuple(item['head_pose']['rotation_vec'])
                item['head_pose']['translation_vec'] = tuple(item['head_pose']['translation_vec'])
        return ClientFrame(await self._base64_to_img(json['src']), await self._base64_to_img(json['prc']),
                           [from_dict(OneFaceMetricsAnalyzeResult, item) for item in json['result']])

    @staticmethod
    async def _img_to_base64(img: cv2.typing.MatLike) -> str:
        _, buffer = cv2.imencode(".jpg", img)
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    async def _base64_to_img(image_b64: str) -> cv2.typing.MatLike:
        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
