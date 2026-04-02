"""
Модуль расчёта метрики вовлечённости (engagement) на основе мультимодального анализа.
Учитывает эмоции, состояние глаз (EAR), поза головы (HPE).

Академическое обоснование текущего выбора весов:
- Эмоции (42%): ρ = 0.37, максимальная корреляция с engagement [Buono et al., 2022]
- Состояние глаз (33%): ρ = -0.36, критично для drowsiness [Dewi et al., 2022]
- Поза головы (25%): ρ = 0.24, вспомогательный индикатор [Gupta et al., 2023]

Формула:
    Engagement = 0.42 × Emotion_Score + 0.33 × Eye_Score + 0.25 × HeadPose_Score

Temporal Smoothing:
    - Эмоции уже сглажены внутри EmotionRecognizer (15 кадров)
    - Engagement сглаживается после вычисления (45 кадров, ~1.5 сек при 30 fps)
    - Адаптивное окно: меньше при стабильном состоянии, больше при изменчивом
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Literal, Optional

import numpy as np

from .analyze_ear import EyeAspectRatioAnalyzeResult
from .analyze_head_pose import HeadPoseEstimateResult


@dataclass
class EngagementComponents:
    emotion_score: float
    eye_score: float
    head_pose_score: float


@dataclass
class EngagementCalculateResult:
    score: float
    score_raw: float
    level: Literal["High", "Medium", "Low", "Very Low"]
    trend: Literal["rising", "falling", "stable"]
    components: EngagementComponents
    frame_count: int


class EngagementCalculator:
    """
    Вычисление и сглаживание метрики вовлечённости
    """

    # Научно обоснованные веса компонентов
    WEIGHTS = {
        "emotion": 0.42,  # Лицевые эмоции / Action Units
        "eye": 0.33,  # Состояние глаз (EAR, моргания)
        "head_pose": 0.25,  # Ориентация головы (pitch, yaw)
    }

    # Маппинг состояния глаз → score (пороги определены в analyze_ear.classify_attention_by_ear)
    EAR_STATE_SCORES = {
        "Alert": 1.0,  # avg_ear >= 0.30
        "Normal": 0.7,  # avg_ear >= 0.25
        "Drowsy": 0.4,  # avg_ear >= 0.20
        "Very Drowsy": 0.1,  # avg_ear < 0.20
    }

    # Маппинг позы головы → score (пороги определены в analyze_head_pose.classify_attention_state)
    HEAD_POSE_STATE_SCORES = {
        "Highly Attentive": 1.0,  # |pitch| < 10, |yaw| < 15
        "Attentive": 0.8,  # |pitch| < 20, |yaw| < 25
        "Distracted": 0.5,  # |pitch| < 30, |yaw| < 40
        "Very Distracted": 0.2,  # иначе
    }

    # Пороговые значения для классификации вовлечённости
    THRESHOLDS = {
        "high": 0.75,  # >= 0.75 — High engagement
        "medium": 0.50,  # >= 0.50 — Medium engagement
        "low": 0.25,  # >= 0.25 — Low engagement
        # < 0.25 — Very Low engagement
    }

    # Веса эмоций для emotion_score (экспертная оценка)
    EMOTION_WEIGHTS = {
        "Happy": 1.0,  # Позитивная вовлечённость
        "Surprise": 0.8,  # Интерес, удивление (продуктивно)
        "Neutral": 0.6,  # Спокойное внимание
        "Contempt": 0.4,  # Скептицизм (частично вовлечён)
        "Fear": 0.3,  # Беспокойство (низкая вовлечённость)
        "Sad": 0.2,  # Грусть, усталость
        "Angry": 0.1,  # Фрустрация
        "Disgust": 0.1,  # Отвращение, отторжение
    }

    def __init__(self, *, window_size: int = 45, bypass_threshold: float = 0.08, trend_window: int = 30):
        """
        Args:
            window_size: Размер окна сглаживания (45 кадров или ~1.5 сек при 30 FPS)
            bypass_threshold: Порог вариации для адаптивного окна
            trend_window: Размер окна для определения тренда
        """
        self.window_size = window_size
        self.bypass_threshold = bypass_threshold
        self.trend_window = trend_window

        # История для сглаживания вовлечённости
        self.engagement_history: deque[float] = deque(maxlen=window_size)

        # История для определения тренда
        self.trend_history: deque[float] = deque(maxlen=trend_window)

        # Время начала сессии (для расчёта частоты моргания)
        self.session_start_time: datetime | None = None

        # Счётчики для статистики
        self.frame_count = 0
        self.total_frames_analyzed = 0

    def reset(self):
        """Сброс истории (для новой сессии)"""
        self.engagement_history.clear()
        self.trend_history.clear()
        self.session_start_time = None
        self.frame_count = 0
        self.total_frames_analyzed = 0

    def calculate_emotion_score(self, emotion: str, confidence: float) -> float:
        """
        Вычисление emotion_score на основе эмоции и confidence

        Args:
            emotion: Название эмоции ('Happy', 'Sad', ...)
            confidence: Уверенность модели (0.0-1.0)

        Returns:
            Emotion score (0.0-1.0)
        """
        # Базовый вес эмоции
        emotion_weight = self.EMOTION_WEIGHTS[emotion]  # default для unknown

        # Учёт уверенности (confidence): если низкая, то значение снижается
        # меньше 0.55 - штраф
        if confidence < 0.55:
            confidence_penalty = confidence / 0.55  # 0.5 conf -> 0.91 penalty
        else:
            confidence_penalty = 1.0

        return emotion_weight * confidence * confidence_penalty

    def calculate_eye_score(self, ear_data: EyeAspectRatioAnalyzeResult, elapsed_time: Optional[float] = None) -> float:
        """
        Вычисление eye_score на основе EAR и частоты моргания.

        Использует предвычисленный attention_state из FaceAnalysisPipeline (EAR_STATE_SCORES).

        Args:
            ear_data: Объект EyeAspectRatioAnalyzeResult
            elapsed_time: Время с начала сессии (секунды) для расчёта blink_rate

        Returns:
            Eye score (0.0-1.0)
        """
        blink_count = ear_data.blink_count

        # Базовый score по attention_state (вычислен в FaceAnalysisPipeline через classify_attention_by_ear)
        attention_state = ear_data.attention_state
        base_score = self.EAR_STATE_SCORES[attention_state]

        # Модификатор по частоте моргания
        blink_modifier = 1.0

        if elapsed_time and elapsed_time > 0:
            # Расчёт частоты моргания (морганий в минуту)
            blink_rate_per_min = (blink_count / elapsed_time) * 60

            # Нормальная частота: 10-25 морганий/минуту
            # Источник: Magliacano et al. (2020), Neuroscience Letters
            if 10 <= blink_rate_per_min <= 25:
                # Идеальная частота — бонус +10%
                blink_modifier = 1.1
            elif blink_rate_per_min < 5:
                # Слишком редко — гиперфокус или усталость, -5%
                blink_modifier = 0.95
            elif blink_rate_per_min > 30:
                # Слишком часто — стресс/раздражение, -10%
                blink_modifier = 0.90

        # Итоговый eye_score (не превышает 1.0)
        return min(1.0, base_score * blink_modifier)

    def calculate_head_pose_score(self, head_pose_data: HeadPoseEstimateResult) -> float:
        """
        Вычисление head_pose_score на основе позы головы.

        Использует предвычисленный attention_state из FaceAnalysisPipeline (HEAD_POSE_STATE_SCORES).

        Args:
            head_pose_data: Объект HeadPoseEstimateResult

        Returns:
            Head pose score (0.0-1.0)
        """
        # Базовый score по attention_state (вычислен в FaceAnalysisPipeline через classify_attention_state)
        attention_state = head_pose_data.attention_state
        return self.HEAD_POSE_STATE_SCORES[attention_state]

    def calculate(
        self,
        emotion: str,
        emotion_confidence: float,
        ear_data: Optional[EyeAspectRatioAnalyzeResult] = None,
        head_pose_data: Optional[HeadPoseEstimateResult] = None,
        timestamp: Optional[datetime] = None,
    ) -> EngagementCalculateResult:
        """
        Главная функция: вычисление engagement score

        Args:
            emotion: Распознанная эмоция
            emotion_confidence: Уверенность модели
            ear_data: Данные состояния глаз (может быть None если лицо не детектировано)
            head_pose_data: Данные позы головы (может быть None)
            timestamp: Временная метка текущего кадра

        Returns:
            EngagementCalculateResult
        """
        # Инициализация времени сессии
        if self.session_start_time is None and timestamp:
            self.session_start_time = timestamp

        # Вычисление elapsed time
        elapsed_time = None
        if self.session_start_time and timestamp:
            elapsed_time = (timestamp - self.session_start_time).total_seconds()

        # 1. Вычисление компонентных значений score
        emotion_score = self.calculate_emotion_score(emotion, emotion_confidence)

        # Если ear_data отсутствует (лицо не детектировано), используется нейтральное значение
        eye_score = self.calculate_eye_score(ear_data, elapsed_time) if ear_data else 0.5

        # аналогично для head_pose
        head_pose_score = self.calculate_head_pose_score(head_pose_data) if head_pose_data else 0.5

        # 2. Взвешенная сумма (raw engagement без сглаживания)
        engagement_raw = (
            self.WEIGHTS["emotion"] * emotion_score
            + self.WEIGHTS["eye"] * eye_score
            + self.WEIGHTS["head_pose"] * head_pose_score
        )

        # Ограничение диапазона до [0.0, 1.0]
        engagement_raw = max(0.0, min(1.0, engagement_raw))

        # 3. Добавление в историю для сглаживания
        self.engagement_history.append(engagement_raw)
        self.trend_history.append(engagement_raw)

        # 4. Temporal smoothing (адаптивное окно)
        if len(self.engagement_history) < 5:
            # Если пока недостаточно истории, используется raw
            engagement_smoothed = float(engagement_raw)
        else:
            # Вычисление вариации на последних 15 кадрах (~0.5 сек)
            recent_window = list(self.engagement_history)[-15:]
            variance = float(np.var(recent_window))

            if variance < self.bypass_threshold:
                # Стабильное состояние -> меньшее окно (быстрее реагирует на изменения)
                engagement_smoothed = float(np.mean(recent_window))
            else:
                # Изменчивое состояние -> полное окно (больше сглаживания)
                engagement_smoothed = float(np.mean(self.engagement_history))

        # 5. Определение тренда
        trend = self._calculate_trend()

        # 6. Классификация уровня engagement
        level = self._classify_level(engagement_smoothed)

        # 7. Обновление счётчиков
        self.frame_count += 1
        self.total_frames_analyzed += 1

        return EngagementCalculateResult(
            score=round(engagement_smoothed, 3),
            score_raw=round(engagement_raw, 3),
            level=level,
            trend=trend,
            components=EngagementComponents(
                emotion_score=round(emotion_score, 3),
                eye_score=round(eye_score, 3),
                head_pose_score=round(head_pose_score, 3),
            ),
            frame_count=self.frame_count,
        )

    def _classify_level(self, score: float) -> Literal["High", "Medium", "Low", "Very Low"]:
        """
        Классификация вовлечённости по категории

        Args:
            score: Engagement score (0.0-1.0)

        Returns:
            'High', 'Medium', 'Low', или 'Very Low'
        """
        if score >= self.THRESHOLDS["high"]:
            return "High"
        elif score >= self.THRESHOLDS["medium"]:
            return "Medium"
        elif score >= self.THRESHOLDS["low"]:
            return "Low"
        else:
            return "Very Low"

    def _calculate_trend(self) -> Literal["rising", "falling", "stable"]:
        """
        Определение тренда вовлечённости (растёт/падает/стабилен)

        Returns:
            'rising', 'falling', или 'stable'
        """
        if len(self.trend_history) < 10:
            return "stable"  # Пока недостаточно данных, заглушкой возвращается stable

        # Сравниваем первую и вторую половину окна
        half = len(self.trend_history) // 2
        first_half_mean = float(np.mean(list(self.trend_history)[:half]))
        second_half_mean = float(np.mean(list(self.trend_history)[half:]))

        diff = second_half_mean - first_half_mean

        # Порог для определения значимого изменения
        threshold = 0.05

        if diff > threshold:
            return "rising"
        elif diff < -threshold:
            return "falling"
        else:
            return "stable"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики за текущую сессию

        Returns:
            Словарь со статистикой
        """
        if not self.engagement_history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "total_frames": self.total_frames_analyzed}

        history_array = np.array(self.engagement_history)

        return {
            "mean": round(np.mean(history_array), 3),
            "std": round(np.std(history_array), 3),
            "min": round(np.min(history_array), 3),
            "max": round(np.max(history_array), 3),
            "total_frames": self.total_frames_analyzed,
            "current_window_size": len(self.engagement_history),
        }
