"""
Модуль оценки позы головы (Head Pose Estimation) 
с использованием MediaPipe Face Mesh landmarks (лицевых точек)
для расчёта метрики вовлечённости
"""

import numpy as np
import cv2
import math

# Индексы landmarks для Head Pose (6-точечная модель)
HEAD_POSE_LANDMARKS = [1, 33, 61, 199, 263, 291]
# 1   = Nose tip                (кончик носа)
# 33  = Left eye outer corner   (внешний угол левого глаза)
# 61  = Left mouth corner       (левый угол рта)
# 199 = Chin                    (подбородок)
# 263 = Right eye outer corner  (внешний угол правого глаза)
# 291 = Right mouth corner      (правый угол рта)

# 3D-модель лица в мировых координатах      (в произвольных единицах)
# Эти координаты представляют усреднённую модель человеческого лица
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (-225.0, 170.0, -135.0),     # Left eye corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (0.0, -330.0, -65.0),        # Chin
    (225.0, 170.0, -135.0),      # Right eye corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)


class HeadPoseEstimator:
    """Оценка позы головы на основе Face Mesh landmarks"""

    def __init__(self):
        """Инициализация анализатора позы головы (класса)"""
        print("HeadPoseEstimator инициализирован")


    @staticmethod
    def _rotation_matrix_to_angles(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
        """
        Конвертация матрицы поворота в углы Эйлера (pitch, yaw, roll) в градусах.

        Args:
            rotation_matrix: 3x3 матрица поворота из cv2.Rodrigues (преобразования Родригеса из вектора в матрицу)

        Returns:
            (pitch, yaw, roll) в градусах
            - pitch: наклон вверх/вниз (положительный: вверх)
            - yaw: поворот влево/вправо (положительный: вправо)
            - roll: наклон головы к плечу (положительный: вправо)
        """

        # Формулы для преобразования матрицы поворота в углы Эйлера
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2))
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        # Конвертация радиан в градусы
        pitch = x * 180.0 / math.pi
        yaw = y * 180.0 / math.pi
        roll = z * 180.0 / math.pi

        return pitch, yaw, roll


    def estimate(self, face_landmarks, image_width: int, image_height: int) -> dict[str, float | tuple[float, float, float]] | None:
        """
        Оценивает позу головы на основе landmarks точек (для одного лица).

        Args:
            face_landmarks: Объект landmarks из MediaPipe Face Mesh (одного лица)
            image_width: Ширина изображения
            image_height: Высота изображения

        Returns:
            Словарь с результатами:
            {
                'pitch': float,  # Угол наклона вверх/вниз (-90 до +90)
                'yaw': float,    # Угол поворота влево/вправо (-90 до +90)
                'roll': float,   # Угол наклона к плечу (-180 до +180)
                'rotation_vec': tuple,  # Вектор поворота
                'translation_vec': tuple  # Вектор смещения
            }
            None, если не удалось вычислить позу
        """
        
        # Извлечение 2D координат ключевых точек для PnP
        image_points = np.array([
            (face_landmarks.landmark[idx].x * image_width,
             face_landmarks.landmark[idx].y * image_height)
            for idx in HEAD_POSE_LANDMARKS
        ], dtype=np.float64)

        # Матрица камеры (упрощ. (быстрая) модель pinhole camera)
        focal_length = image_width                      # Приближение: focal length ≈ image width
        center = (image_width / 2, image_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Коэффициенты дисторшена       (предполагаем отсутствие искажений)
        dist_coeffs = np.zeros((4, 1))


        # Решение Perspective-n-Point (PnP) задачи
        success, rotation_vec, translation_vec = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # если решение найдено - нужно его обработать
        if success:
            # Конвертация вектора поворота в матрицу
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)

            # Извлечение углов Эйлера
            pitch, yaw, roll = self._rotation_matrix_to_angles(rotation_mat)

            return {
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                'rotation_vec': tuple(rotation_vec.flatten()),
                'translation_vec': tuple(translation_vec.flatten())
            }

        return None


# TODO: донастройка параметров и порогов при практическом тесте механизма
def classify_attention_state(pitch: float, yaw: float, roll: float) -> str:
    """
    Классификация состояния внимания на основе углов головы.

    Args:
        pitch: Угол наклона вверх/вниз
        yaw: Угол поворота влево/вправо
        roll: Угол наклона к плечу

    Returns:
        Строка с уровнем внимания: "Highly Attentive", "Attentive", "Distracted", "Very Distracted"
    """
    abs_pitch = abs(pitch)
    abs_yaw = abs(yaw)

    # Критерии внимания (пороговые значения)
    if abs_pitch < 10 and abs_yaw < 15:
        return "Highly Attentive"                   # Прямой взгляд на экран
    elif abs_pitch < 20 and abs_yaw < 25:
        return "Attentive"                          # Небольшое отклонение
    elif abs_pitch < 30 and abs_yaw < 40:
        return "Distracted"                         # Заметное отклонение
    else:
        return "Very Distracted"                    # Взгляд в сторону/вниз/вверх

