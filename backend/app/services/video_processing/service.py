"""
Модуль сервиса пайплайна анализа лиц для FastAPI.
"""

import logging
from uuid import UUID

from cv2.typing import MatLike

from .face_analysis_pipeline import (
    FaceAnalysisPipeline,
    FaceAnalyzeResult,
    make_face_analysis_pipeline,
)

logger = logging.getLogger(__name__)


class FaceAnalysisPipelineService:
    """
    Сервис для управления анализом лиц и эмоций.

    Создает и хранит отдельный экземпляр FaceAnalysisPipeline для каждого клиента.
    """

    _analyzers: dict[UUID, FaceAnalysisPipeline] = {}

    def __init__(self):
        """Инициализирует сервис."""

    async def analyze(self, client_id: UUID, image: MatLike) -> FaceAnalyzeResult:
        """
        Анализирует изображение для конкретного клиента.

        Если анализатор для клиента еще не создан, создает новый.

        Args:
            client_id: ID клиента
            image: Изображение для анализа

        Returns:
            FaceAnalyzeResult: Результат анализа с обработанным изображением и метриками
        """
        if client_id not in self._analyzers:
            self._analyzers[client_id] = make_face_analysis_pipeline()
            logger.debug(f"Created new FaceAnalysisPipeline for client {client_id}")
        return self._analyzers[client_id].analyze(image)

    async def remove(self, client_id: UUID):
        removed = self._analyzers.pop(client_id, None)
        if removed is not None:
            logger.debug(f"Removed FaceAnalysisPipeline for client {client_id}")
