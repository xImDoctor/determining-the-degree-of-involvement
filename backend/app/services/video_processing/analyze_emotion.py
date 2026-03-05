import logging
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Literal, cast

import cv2
import torch
from emotiefflib.facial_analysis import EmotiEffLibRecognizer  # type: ignore[import-untyped]

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EmotionRecognizeResult:
    label: str
    confidence: float


class EmotionRecognizer:
    """Распознавание с temporal smoothing + confidence thresholding"""

    if settings.emotion_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif settings.emotion_device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            warnings.warn('emotion_device is set as "cuda", but cuda is unavailable')
    else:
        device = "cpu"

    recognizer = EmotiEffLibRecognizer(model_name=settings.emotion_model_name, device=device)

    def __init__(
        self,
        *,
        device: Literal["cpu", "cuda", "auto"] | None = None,
        model_name: str | None = None,
        window_size: int | None = None,
        confidence_threshold: float | None = None,
        ambiguity_threshold: float | None = None,
    ):
        """
        Args:
            window_size: Размер окна для сглаживания
            confidence_threshold: Минимальный порог уверенности
            ambiguity_threshold: Порог для амбивалентных эмоций
        """
        actual_window_size = window_size if window_size is not None else settings.emotion_window_size
        actual_confidence = (
            confidence_threshold if confidence_threshold is not None else settings.emotion_confidence_threshold
        )
        actual_ambiguity = (
            ambiguity_threshold if ambiguity_threshold is not None else settings.emotion_ambiguity_threshold
        )
        if device is not None or model_name is not None:
            if settings.emotion_device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif settings.emotion_device == "cuda":
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
                    warnings.warn('device in EmotionRecognizer() is set as "cuda", but cuda is unavailable')
            else:
                device = "cpu"
            self.recognizer = EmotiEffLibRecognizer(model_name=model_name, device=device)

        self._validate_window_size(actual_window_size)
        self._validate_confidence_threshold(actual_confidence)
        self._validate_ambiguity_threshold(actual_ambiguity)

        self.emotion_labels = [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Sad",
            "Surprise",
            "Neutral",
            "Contempt",
        ]

        # Параметры сглаживания
        self.history: deque[dict[str, float | str]] = deque(maxlen=actual_window_size)

        # Параметры фильтрации
        self.confidence_threshold = actual_confidence
        self.ambiguity_threshold = actual_ambiguity

    @staticmethod
    def _validate_window_size(window_size: int):
        if not isinstance(window_size, int):
            raise TypeError(f'Type of "window_size" should be int, got {type(window_size).__name__}')
        if window_size < 0:
            raise ValueError('"window_size" should be >= 0')

    def set_window_size(self, window_size: int) -> None:
        """
        Устанавливает размер окна для сглаживания.

        Args:
            window_size: Размер окна для temporal smoothing
        """
        self._validate_window_size(window_size)
        self.history = deque(maxlen=window_size)

    @staticmethod
    def _validate_confidence_threshold(confidence_threshold: float) -> None:
        """
        Валидация порога уверенности.

        Args:
            confidence_threshold: Значение порога для проверки

        Raises:
            TypeError: Если тип не float
            ValueError: Если значение вне диапазона [0;1]
        """
        if not isinstance(confidence_threshold, (float, int)):
            raise TypeError(
                f'Type of "confidence_threshold" should be float, got {type(confidence_threshold).__name__}'
            )
        if confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError('"confidence_threshold" should be in [0;1]')

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        """
        Устанавливает минимальный порог уверенности для распознавания эмоций.

        Args:
            confidence_threshold: Значение порога в диапазоне [0, 1]
        """
        self._validate_confidence_threshold(confidence_threshold)
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _validate_ambiguity_threshold(ambiguity_threshold: float) -> None:
        """
        Валидация порога амбивалентности.

        Args:
            ambiguity_threshold: Значение порога для проверки

        Raises:
            TypeError: Если тип не float
            ValueError: Если значение вне диапазона [0;1]
        """
        if not isinstance(ambiguity_threshold, (float, int)):
            raise TypeError(f'Type of "ambiguity_threshold" should be float, got {type(ambiguity_threshold).__name__}')
        if ambiguity_threshold < 0 or ambiguity_threshold > 1:
            raise ValueError('"ambiguity_threshold" should be in [0;1]')

    def set_ambiguity_threshold(self, ambiguity_threshold: float) -> None:
        """
        Устанавливает порог для определения амбивалентных эмоций.

        Args:
            ambiguity_threshold: Значение порога в диапазоне [0, 1]
        """
        self._validate_ambiguity_threshold(ambiguity_threshold)
        self.ambiguity_threshold = ambiguity_threshold

    def predict(self, face_crop: cv2.typing.MatLike) -> EmotionRecognizeResult:
        """Предсказывает эмоцию с продвинутой фильтрацией"""
        if face_crop.size == 0:
            return EmotionRecognizeResult("Neutral", 0.0)  # Fallback к нейтральному

        try:
            # Получаем предсказание
            emotion, scores = self.recognizer.predict_emotions(face_crop, logits=True)

            # Берём топ эмоцию и confidence
            top_emotion = emotion[0]

            if scores is not None and len(scores) > 0:
                confidence = float(max(scores[0])) if hasattr(scores[0], "__iter__") else float(scores[0])
            else:
                confidence = 1.0

            # Шаг 1: Проверка confidence threshold
            if confidence < self.confidence_threshold:
                # Слишком низкая уверенность -> нейтральное состояние
                top_emotion = "Neutral"
                confidence = self.confidence_threshold * 0.9

            # Добавляем в историю
            self.history.append({"emotion": top_emotion, "confidence": confidence})

            # Шаг 2: Temporal smoothing
            if len(self.history) >= 3:
                emotion_votes: dict[str, float] = {}
                total_weight: float = 0.0

                for i, hist_item in enumerate(self.history):
                    weight = (i + 1) / len(self.history)
                    emo = cast(str, hist_item["emotion"])
                    conf = cast(float, hist_item["confidence"])

                    if emo not in emotion_votes:
                        emotion_votes[emo] = 0.0
                    emotion_votes[emo] += weight * conf
                    total_weight += weight

                # Сортируем эмоции по весу
                sorted_emotions = sorted(emotion_votes.items(), key=lambda x: x[1], reverse=True)

                # Шаг 3: Проверка амбивалентности
                if len(sorted_emotions) >= 2:
                    top_emotion_result, top_score = sorted_emotions[0]
                    second_emotion, second_score = sorted_emotions[1]

                    # Если две топ-эмоции слишком близки -> нейтральное
                    if (top_score - second_score) / total_weight < self.ambiguity_threshold:
                        return EmotionRecognizeResult("Neutral", 0.5)

                    return EmotionRecognizeResult(top_emotion_result, top_score / total_weight)
                else:
                    top_emotion_result, top_score = sorted_emotions[0]
                    return EmotionRecognizeResult(top_emotion_result, top_score / total_weight)

            return EmotionRecognizeResult(top_emotion, confidence)

        except (torch.cuda.OutOfMemoryError, MemoryError):
            logger.error("Out of memory in EmotionRecognizer.predict()")
            raise

        except (ValueError, RuntimeError, AttributeError) as e:
            logger.warning(f"Emotion recognition warning: {e}")
            return EmotionRecognizeResult("Neutral", 0.0)

    def reset(self):
        """Сброс истории"""
        self.history.clear()
