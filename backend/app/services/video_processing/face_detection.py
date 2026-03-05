"""
Модуль детекции лиц
"""

import logging
from collections import deque
from dataclasses import dataclass
from time import time

import cv2
import mediapipe as mp  # type: ignore[import-untyped]

from app.core.config import settings

logger = logging.getLogger(__name__)

# Настройка MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


@dataclass
class FaceDetectResult:
    bbox: tuple[int, int, int, int]
    crop: cv2.typing.MatLike
    confidence: float
    keypoints: list[tuple[int, int]]


class FaceDetector:
    """Модуль детекции лиц с использованием MediaPipe"""

    detector = mp_face_detection.FaceDetection(
        model_selection=settings.face_detection_model_selection,
        min_detection_confidence=settings.face_detection_min_confidence,
    )

    def __init__(self, *, min_detection_confidence: float | None = None, margin: int | None = None):
        """

        :param margin: Добавочный отступ к bbox, предсказанный моделью
        """
        if min_detection_confidence is not None:
            self.detector = mp_face_detection.FaceDetection(
                model_selection=settings.face_detection_model_selection,
                min_detection_confidence=min_detection_confidence,
            )

        actual_margin = margin if margin is not None else settings.face_detection_margin
        self._validate_margin(actual_margin)

        self.margin = actual_margin

    @staticmethod
    def _validate_margin(margin: int) -> None:
        """Валидация margin"""
        if not isinstance(margin, int):
            raise TypeError(f'Type of "margin" should be int, got {type(margin).__name__}')
        if margin < 0:
            raise ValueError('"margin" should be >= 0')

    @staticmethod
    def _validate_confidence(min_detection_confidence: float) -> None:
        """Валидация confidence"""
        if not isinstance(min_detection_confidence, (float, int)):
            raise TypeError(
                f'Type of "min_detection_confidence" should be float, got {type(min_detection_confidence).__name__}'
            )
        if not 0 <= min_detection_confidence <= 1:
            raise ValueError('"min_detection_confidence" should be in [0, 1]')

    def set_margin(self, margin: int) -> None:
        """
        Устанавливает отступ для bounding box лица.

        Args:
            margin: Значение отступа в пикселях
        """
        self._validate_margin(margin)
        self.margin = margin

    def set_min_detection_confidence(self, min_detection_confidence: float) -> None:
        """
        Устанавливает минимальный порог уверенности для детекции лиц.

        Args:
            min_detection_confidence: Значение порога в диапазоне [0, 1]
        """
        self._validate_confidence(min_detection_confidence)
        self.detector = mp_face_detection.FaceDetection(
            model_selection=settings.face_detection_model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, image: cv2.typing.MatLike) -> list[FaceDetectResult]:
        """Детектирует лица на изображении"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)

        faces: list[FaceDetectResult] = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)

                x1 = max(0, x - self.margin)
                y1 = max(0, y - self.margin)
                x2 = min(w, x + w_box + self.margin)
                y2 = min(h, y + h_box + self.margin)

                face_crop = image[y1:y2, x1:x2]

                # ИЗВЛЕКАЕМ КЛЮЧЕВЫЕ ТОЧКИ (6 точек)
                keypoints: list[tuple[int, int]] = []
                if detection.location_data.relative_keypoints:
                    for keypoint in detection.location_data.relative_keypoints:
                        kp_x = int(keypoint.x * w)
                        kp_y = int(keypoint.y * h)
                        keypoints.append((kp_x, kp_y))

                faces.append(
                    FaceDetectResult(
                        bbox=(x1, y1, x2, y2),
                        crop=face_crop,
                        confidence=detection.score[0],
                        keypoints=keypoints,
                    )
                )

        return faces

    def close(self):
        self.detector.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from app.services.video_processing.video_stream import process_video_stream

    logger.info("Using camera 0")
    cap = cv2.VideoCapture(0)
    fps_history: deque[float] = deque()
    FPS_HISTORY_LEN = 3  # для более гладкого fps, будет выводится средние из последних FPS_HISTORY_LEN измерений

    for _ in range(FPS_HISTORY_LEN):
        fps_history.append(0.0)
    try:
        start_time = time()
        for img, emotions in process_video_stream(cap, flip_h=True):
            cv2.putText(
                img,
                f"FPS: {round(sum(fps_history) / FPS_HISTORY_LEN)}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )
            cv2.imshow("Test", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break  # esc to quit
            fps = 1 / (time() - start_time)
            fps_history.append(fps)
            fps_history.popleft()
            start_time = time()
    finally:
        logger.info("Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Done!")
