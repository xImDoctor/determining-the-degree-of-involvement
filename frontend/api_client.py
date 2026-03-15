"""
Модуль WebSocket-клиента для взаимодействия с FastAPI-бэкендом.

Синхронный API-клиент для отправки видеокадров
на обработку и получения результатов анализа.
"""

import base64
import json
import logging

import cv2
import numpy as np
import requests
import websocket

logger = logging.getLogger(__name__)


class EngagementAPIClient:
    """Синхронный WebSocket-клиент для бэкенда распознавания вовлечённости."""

    def __init__(self, backend_ws_url: str = "ws://localhost:8000", backend_http_url: str = "http://localhost:8000"):
        self._backend_ws_url = backend_ws_url.rstrip("/")
        self._backend_http_url = backend_http_url.rstrip("/")
        self._ws: websocket.WebSocket | None = None
        self._room_id: str | None = None
        self._client_name: str | None = None

    def connect(self, room_id: str = "streamlit", name: str = "streamlit-user") -> None:
        """
        Подключение к WebSocket эндпоинту /ws/rooms/{room_id}/stream.

        Args:
            room_id: Идентификатор комнаты
            name: Имя клиента
        """
        self._room_id = room_id
        self._client_name = name
        ws_url = f"{self._backend_ws_url}/ws/rooms/{room_id}/stream?name={name}"
        try:
            self._ws = websocket.create_connection(ws_url, timeout=10)
            logger.info(f"Connected to {ws_url}")
        except Exception as e:
            logger.error(f"Failed to connect to {ws_url}: {e}")
            self._ws = None
            raise ConnectionError(f"Не удалось подключиться к бэкенду: {e}") from e

    def disconnect(self) -> None:
        """Закрытие WebSocket соединения."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            logger.info("Disconnected from backend")

    @property
    def is_connected(self) -> bool:
        """Проверка активности WebSocket соединения."""
        return self._ws is not None and self._ws.connected

    def _reconnect(self) -> bool:
        """Попытка переподключения при потере соединения."""
        if self._room_id is None:
            return False
        try:
            self.disconnect()
            self.connect(self._room_id, self._client_name or "streamlit-user")
            return True
        except ConnectionError:
            return False

    def send_frame(self, frame: np.ndarray) -> tuple[np.ndarray | None, list[dict]]:
        """
        Отправка кадра на обработку и получение результатов.

        Args:
            frame: Кадр в формате BGR (numpy array от OpenCV)

        Returns:
            Кортеж (обработанный кадр BGR или None, список результатов анализа)
        """
        if not self.is_connected:
            if not self._reconnect():
                return None, []

        # Кодирование кадра в base64 JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        try:
            self._ws.send(json.dumps({"image": image_b64}))
            response_raw = self._ws.recv()
            response = json.loads(response_raw)
        except (websocket.WebSocketException, ConnectionError, OSError) as e:
            logger.warning(f"WebSocket send/recv error: {e}")
            if self._reconnect():
                try:
                    self._ws.send(json.dumps({"image": image_b64}))
                    response_raw = self._ws.recv()
                    response = json.loads(response_raw)
                except Exception:
                    return None, []
            else:
                return None, []

        # Обработка ошибки от сервера
        if "error" in response:
            logger.warning(f"Server error: {response['error']}")
            return None, []

        # Декодирование обработанного изображения
        processed_frame = None
        if "image" in response:
            try:
                img_bytes = base64.b64decode(response["image"])
                nparr = np.frombuffer(img_bytes, np.uint8)
                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.warning(f"Failed to decode processed image: {e}")

        results = response.get("results", [])
        return processed_frame, results

    def check_health(self) -> bool:
        """
        Проверка доступности бэкенда через GET /health.

        Returns:
            True если бэкенд доступен и здоров
        """
        try:
            resp = requests.get(f"{self._backend_http_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
