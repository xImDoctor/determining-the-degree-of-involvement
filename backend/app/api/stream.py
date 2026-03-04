"""
Модуль WebSocket эндпоинтов для видеопотока и стриминга.
"""

import asyncio
import base64
import logging
from dataclasses import asdict
from typing import Annotated
from uuid import UUID, uuid4

import cv2
import numpy as np
from cv2 import error
from fastapi import APIRouter, Depends, Path, Query, WebSocket, WebSocketDisconnect, status

from app.db.rooms_and_clients import Client, ClientNotFoundError, RoomNotFoundError
from app.services.room import RoomService
from app.services.video_processing import FaceAnalysisPipelineService

logger = logging.getLogger(__name__)

stream_router = APIRouter()


@stream_router.websocket("/ws/rooms/{room_id}/stream")
async def stream(
    websocket: WebSocket,
    room_service: Annotated[RoomService, Depends()],
    analyzer_service: Annotated[FaceAnalysisPipelineService, Depends()],
    room_id: Annotated[str, Path(max_length=40)],
    name: Annotated[str | None, Query(max_length=30)] = None,
):
    """
    WebSocket эндпоинт для получения видеопотока и анализа эмоций.

    Клиент отправляет кадры в формате base64, сервер обрабатывает их и возвращает
    результаты анализа (эмоции, bounding boxes, EAR, HeadPose).

    Args:
        websocket: WebSocket соединение
        room_service: Сервис управления комнатами
        analyzer_service: Сервис анализа лиц и эмоций
        room_id: ID комнаты
        name: Имя клиента (опционально)

    Returns:
        JSON с обработанным изображением и результатами анализа в формате base64

    Raises:
        WebSocketDisconnect: При закрытии соединения клиентом
    """
    await websocket.accept()
    client_id = uuid4()
    client_name = name if name else f"client_{client_id.hex[:8]}"
    client: Client = Client(id_=client_id, name=client_name)
    await room_service.add_client(room_id, client)
    logger.info(f"Client {client.id_} connected to room {room_id} (name: {client_name})")
    try:
        while True:
            data = await websocket.receive_json()
            image_b64 = data.get("image")
            try:
                image_bytes = base64.b64decode(image_b64)
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except (ValueError, TypeError, error) as e:
                logger.warning(f"Failed to decode image from client {client.id_}: {e}")
                await websocket.send_json({"error": f"Failed to decode image: {str(e)}"})
                continue
            if img is None:
                logger.warning("Could not decode img in /ws/rooms/{room_id}/stream")
                continue
            analyze_res = await analyzer_service.analyze(client.id_, img)
            new_img = analyze_res.image
            results = analyze_res.metrics

            _, prc_buffer = cv2.imencode(".jpg", new_img)
            prc_b64 = base64.b64encode(prc_buffer).decode("utf-8")

            await room_service.send_frame(client, image_b64, prc_b64, results)
            results_serializable = list(map(asdict, results))
            await websocket.send_json({"image": prc_b64, "results": results_serializable})
    except WebSocketDisconnect:
        logger.info(f"Client {client.id_} disconnected from room {room_id}")
    finally:
        await room_service.remove_client(room_id, client)
        await analyzer_service.remove(client.id_)


@stream_router.websocket("/ws/rooms/{room_id}/clients/{client_id}/output_stream")
async def client_stream(
    websocket: WebSocket,
    room_id: Annotated[str, Path(max_length=40)],
    client_id: UUID,
    room_service: Annotated[RoomService, Depends()],
):
    """
    WebSocket эндпоинт для получения обработанного видеопотока конкретного клиента.

    Позволяет получать исходный и обработанный кадры для отображения клиенту.

    Args:
        websocket: WebSocket соединение
        room_id: ID комнаты
        client_id: ID клиента-источника видеопотока
        room_service: Сервис управления комнатами

    Returns:
        JSON с исходным и обработанным изображением в формате base64

    Raises:
        WebSocketDisconnect: При закрытии соединения клиентом
    """
    await websocket.accept()
    logger.info(f"Output stream requested for client {client_id} in room {room_id}")
    try:
        client = await room_service.get_client(room_id, client_id)
    except (RoomNotFoundError, ClientNotFoundError) as e:
        logger.warning(f"Failed to get client {client_id} in room {room_id}: {e}")
        await websocket.send_json({"error": str(e)})
        await websocket.close(status.WS_1008_POLICY_VIOLATION)
        return
    try:
        while True:
            if await room_service.client_is_closed(client):
                return
            frame_data = await room_service.get_frame_raw(client)
            if frame_data is None:
                await asyncio.sleep(0.01)
                continue
            results_serializable = list(map(asdict, frame_data.results))
            await websocket.send_json(
                {"image_src": frame_data.src_b64, "image": frame_data.prc_b64, "results": results_serializable}
            )

    except WebSocketDisconnect:
        logger.info(f"Output stream closed for client {client_id} in room {room_id}")
