"""
Модуль расчёта Eye Aspect Ratio (EAR) для детекции состояния глаз
для расчёта метрики вовлечённости
"""

import numpy as np
from collections import deque

# Индексы landmarks для правого глаза (6 точек)
# Порядок: [P1, P2, P3, P4, P5, P6]
# P1, P4 = горизонтальные углы (внешний, внутренний)
# P2, P6 = верхнее/нижнее веко (первая вертикальная пара)
# P3, P5 = верхнее/нижнее веко (вторая вертикальная пара)
# Индексы из Face Mesh
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 145]

# Индексы landmarks для левого глаза (6 точек)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]


# TODO: донастройка параметров и порогов при практическом тесте механизма
class EyeAspectRatioAnalyzer:
    """Анализ состояния глаз с использованием Eye Aspect Ratio (EAR)"""

    def __init__(self, *, ear_threshold: float = 0.25, consec_frames: int = 3):
        """
        Args:
            ear_threshold: Порог EAR для детекции закрытых глаз (обычно 0.25)
            consec_frames: Количество кадров подряд для подтверждения моргания
        """
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames

        # История для детекции моргания (для каждого лица отдельно)
        self.blink_counters = {}  # {face_id: counter}
        self.blink_totals = {}    # {face_id: total_blinks}
        self.ear_history = {}     # {face_id: deque([ear_values])}

        print(f"EyeAspectRatioAnalyzer инициализирован: threshold={ear_threshold}, frames={consec_frames}")


    @staticmethod
    def _calculate_ear(eye_coords: list[tuple[float, float]]) -> float:
        """
        Расчёт Eye Aspect Ratio для одного глаза.

        Формула EAR:
        EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)

        Args:
            eye_coords: Список из 6 точек (x, y) в порядке [P1, P2, P3, P4, P5, P6]

        Returns:
            Значение EAR (~0.2-0.4 для открытых глаз, < 0.2 для закрытых)
        """
        # Вертикальные расстояния между верхним и нижним веком
        A = np.linalg.norm(np.array(eye_coords[1]) - np.array(eye_coords[5]))  # ||P2-P6||
        B = np.linalg.norm(np.array(eye_coords[2]) - np.array(eye_coords[4]))  # ||P3-P5||

        # Горизонтальное расстояние между углами глаза
        C = np.linalg.norm(np.array(eye_coords[0]) - np.array(eye_coords[3]))  # ||P1-P4||

        # EAR формула
        if C > 0:
            return (A + B) / (2.0 * C)
        else:
            return 0.0


    def _get_eye_coordinates(self, landmarks, eye_indices: list[int], w: int, h: int) -> list[tuple[float, float]]:
        """Извлечение координат глаза из landmarks"""

        return [(landmarks[idx].x * w, landmarks[idx].y * h) for idx in eye_indices]


    def analyze(self, face_landmarks, image_width: int, image_height: int, face_id: int = 0) -> dict[str, float | int | bool] | None:
        """
        Анализирует состояние глаз на основе landmarks (для одного лица).

        Args:
            face_landmarks: Объект landmarks из MediaPipe Face Mesh для одного лица
            image_width: Ширина изображения
            image_height: Высота изображения
            face_id: ID лица для отслеживания истории моргания (по умолчанию 0)

        Returns:
            Словарь с результатами анализа:
            {
                'left_ear': float,      # EAR левого глаза
                'right_ear': float,     # EAR правого глаза
                'avg_ear': float,       # Средний EAR
                'eyes_open': bool,      # Открыты ли глаза
                'blink_count': int,     # Общее количество морганий
                'is_blinking': bool     # Моргает ли сейчас
            }
            None, если лицо не обнаружено
        """
        landmarks = face_landmarks.landmark

        # Расчёт EAR для обоих глаз
        left_eye_coords = self._get_eye_coordinates(landmarks, LEFT_EYE_LANDMARKS, image_width, image_height)
        right_eye_coords = self._get_eye_coordinates(landmarks, RIGHT_EYE_LANDMARKS, image_width, image_height)

        left_ear = self._calculate_ear(left_eye_coords)
        right_ear = self._calculate_ear(right_eye_coords)
        avg_ear = (left_ear + right_ear) / 2.0

        # Инициализация счётчиков для нового лица
        if face_id not in self.blink_counters:
            self.blink_counters[face_id] = 0
            self.blink_totals[face_id] = 0
            self.ear_history[face_id] = deque(maxlen=30)  # История EAR за последние 30 кадров

        # Добавление в историю
        self.ear_history[face_id].append(avg_ear)

        # Детекция моргания
        is_blinking = False
        if avg_ear < self.ear_threshold:
            # Глаза закрыты
            self.blink_counters[face_id] += 1
            is_blinking = True
        else:
            # Глаза открыты
            if self.blink_counters[face_id] >= self.consec_frames:
                # Подтверждённое моргание
                self.blink_totals[face_id] += 1
            self.blink_counters[face_id] = 0

        return {
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'eyes_open': avg_ear >= self.ear_threshold,
            'blink_count': self.blink_totals[face_id],
            'is_blinking': is_blinking,
            'ear_history': list(self.ear_history[face_id])
        }

    def reset(self, face_id: int = None):
        """Сброс счётчиков моргания"""
        if face_id is None:
            # Сброс всех лиц
            self.blink_counters.clear()
            self.blink_totals.clear()
            self.ear_history.clear()
        else:
            # Сброс конкретного лица
            self.blink_counters.pop(face_id, None)
            self.blink_totals.pop(face_id, None)
            self.ear_history.pop(face_id, None)


def classify_attention_by_ear(avg_ear: float, blink_rate: float) -> str:
    """
    Классификация состояния внимания на основе EAR и частоты моргания.

    Args:
        avg_ear: Средний EAR
        blink_rate: Частота моргания (морганий/минуту)

    Returns:
        Строка с уровнем внимания: "Alert", "Normal", "Drowsy", "Very Drowsy"
    """
    # Нормальная частота моргания: 15-20 раз/минуту
    # Сниженная частота (< 5/мин) = сильная концентрация или усталость
    # Повышенная частота (> 30/мин) = стресс или раздражение

    if avg_ear >= 0.30:
        if 10 <= blink_rate <= 25:
            return "Alert"          # Нормальное состояние (внимательность, сосредоточненность)
        else:
            return "Normal"         # Немного отклонения (обычные, открытые глаза)
    elif 0.25 <= avg_ear < 0.30:
        return "Normal"             # Пограничное состояние (так же принимаем за нормальное)
    elif 0.20 <= avg_ear < 0.25:
        return "Drowsy"             # Начало усталости (читаемая усталость)
    else:
        return "Very Drowsy"        # Глаза почти закрыты (сильная усталость)
