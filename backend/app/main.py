"""
Модуль основного приложения FastAPI.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.room import room_router
from app.api.stream import stream_router
from app.core.config import settings
from app.services.room import RoomService

logging.getLogger().handlers.clear()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекст управления жизненным циклом приложения.

    Выполняет инициализацию при запуске, и очистку при завершении.
    """
    logger.info("Starting engagement detection API")
    logger.info(f"API version: {settings.app_version}")
    yield
    logger.info("Shutting down engagement detection API")
    await RoomService.storage.flushdb()


app = FastAPI(
    title="API распознавания эмоций",
    description="REST API для детекции лиц и распознавания эмоций в реальном времени",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@stream_router.get("/health")
async def health_check():
    """
    Проверка работоспособности сервиса.

    Returns:
        dict: Статус сервиса и версия приложения
    """
    logger.debug("Health check requested")
    return {"status": "healthy", "version": settings.app_version}


app.include_router(stream_router)
app.include_router(room_router)
