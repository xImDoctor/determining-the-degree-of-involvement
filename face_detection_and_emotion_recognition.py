"""
Модуль детекции лиц и распознавания эмоций
"""

import typing
from time import time
from collections import deque

import torch
import cv2
import mediapipe as mp
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from emotiefflib.facial_analysis import get_model_list

# Настройка MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class FaceDetector:
    """Детектор лиц MediaPipe Full-Range (для разных дистанций)"""

    def __init__(self, *, min_detection_confidence=0.5, margin=20):
        """

        :param min_detection_confidence: Минимальный уровень уверенности модели, чтобы считать, что лицо есть
        :param margin: Добавочный отступ к bbox, предсказанный моделью
        """
        self.detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = full-range model (до 5 метров)
            min_detection_confidence=min_detection_confidence
        )
        self.name = "MediaPipe Full-Range"
        self.margin = margin

    def detect(self, image: cv2.typing.MatLike) -> list[dict[str,
    tuple[int, int, int, int] | cv2.typing.MatLike | float | list[tuple[int, int]]]]:
        """Детектирует лица на изображении"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)

        faces = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)

                self.margin = 20
                x1 = max(0, x - self.margin)
                y1 = max(0, y - self.margin)
                x2 = min(w, x + w_box + self.margin)
                y2 = min(h, y + h_box + self.margin)

                face_crop = image[y1:y2, x1:x2]

                # ИЗВЛЕКАЕМ КЛЮЧЕВЫЕ ТОЧКИ (6 точек)
                keypoints = []
                if detection.location_data.relative_keypoints:
                    for keypoint in detection.location_data.relative_keypoints:
                        kp_x = int(keypoint.x * w)
                        kp_y = int(keypoint.y * h)
                        keypoints.append((kp_x, kp_y))

                faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'crop': face_crop,
                    'confidence': detection.score[0],
                    'keypoints': keypoints
                })

        return faces

    def close(self):
        self.detector.close()


class EmotionRecognizer:
    """Распознавание с temporal smoothing + confidence thresholding"""

    def __init__(self, *, device='cpu', window_size=15, alpha=0.3,
                 confidence_threshold=0.55, ambiguity_threshold=0.15,
                 model_name='enet_b2_8_best'):
        """
        Args:
            device: 'cpu' или 'cuda'
            window_size: Размер окна для сглаживания
            alpha: Коэффициент для EMA
            confidence_threshold: Минимальный порог уверенности
            ambiguity_threshold: Порог для амбивалентных эмоций
            model_name: Имя модели
        """
        self.recognizer = EmotiEffLibRecognizer(
            model_name=get_model_list()[2],
            device=device
        )
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy',
                               'Sad', 'Surprise', 'Neutral', 'Contempt']

        # Параметры сглаживания
        self.window_size = window_size
        self.alpha = alpha
        self.history = deque(maxlen=window_size)

        # Параметры фильтрации
        self.confidence_threshold = confidence_threshold
        self.ambiguity_threshold = ambiguity_threshold
        self.name = f"EmotiEffLib + Advanced ({model_name})"
        print(f"EmotiEffLib + Advanced загружен: модель={model_name}, устройство={device}")

    def predict(self, face_crop: cv2.typing.MatLike) -> tuple[str, float]:
        """Предсказывает эмоцию с продвинутой фильтрацией"""
        if face_crop.size == 0:
            return "Neutral", 0.0  # Fallback к нейтральному

        try:
            # Получаем предсказание
            emotion, scores = self.recognizer.predict_emotions(face_crop, logits=True)

            # Берём топ эмоцию и confidence
            top_emotion = emotion[0]

            if scores is not None and len(scores) > 0:
                confidence = float(max(scores[0])) if hasattr(scores[0], '__iter__') else float(scores[0])
            else:
                confidence = 1.0

            # Шаг 1: Проверка confidence threshold
            if confidence < self.confidence_threshold:
                # Слишком низкая уверенность -> нейтральное состояние
                top_emotion = "Neutral"
                confidence = self.confidence_threshold * 0.9

            # Добавляем в историю
            self.history.append({
                'emotion': top_emotion,
                'confidence': confidence
            })

            # Шаг 2: Temporal smoothing
            if len(self.history) >= 3:
                emotion_votes = {}
                total_weight = 0

                for i, hist_item in enumerate(self.history):
                    weight = (i + 1) / len(self.history)
                    emo = hist_item['emotion']
                    conf = hist_item['confidence']

                    if emo not in emotion_votes:
                        emotion_votes[emo] = 0
                    emotion_votes[emo] += weight * conf
                    total_weight += weight

                # Сортируем эмоции по весу
                sorted_emotions = sorted(
                    emotion_votes.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Шаг 3: Проверка амбивалентности
                if len(sorted_emotions) >= 2:
                    top_emotion_result, top_score = sorted_emotions[0]
                    second_emotion, second_score = sorted_emotions[1]

                    # Если две топ-эмоции слишком близки -> нейтральное
                    if (top_score - second_score) / total_weight < self.ambiguity_threshold:
                        return "Neutral", 0.5

                    return top_emotion_result, top_score / total_weight
                else:
                    top_emotion_result, top_score = sorted_emotions[0]
                    return top_emotion_result, top_score / total_weight

            return top_emotion, confidence

        except (torch.cuda.OutOfMemoryError, MemoryError):
            # Критично - пробрасываем выше для обработки
            print('Out of memory in EmotionRecognizer.predict()')
            raise

        except (ValueError, RuntimeError, AttributeError) as e:
            # Ожидаемые проблемы обработки - логируем и fallback
            print(f"Предупреждение при распознавании: {e}")
            return "Neutral", 0.0

    def reset(self):
        """Сброс истории"""
        self.history.clear()


class DetectFaceAndRecognizeEmotion:
    """Детектирует лица и распознаёт эмоции"""

    def __init__(self, face_detector: FaceDetector, emotion_recognizer: EmotionRecognizer):
        self.face_detector = face_detector
        self.emotion_recognizer = emotion_recognizer

    def detect_and_recognize(self, image: cv2.typing.MatLike) -> tuple[cv2.typing.MatLike, list[tuple[str, float]]]:
        """
        Детектирует лица и распознаёт эмоции
        :param image: Входное изображение с лицами для анализа
        :return: Возвращает изображение с bbox'ами, эмоции и уверенности в эмоциях.
        Формат: (image, [(emotion, confidence), ...])
        """
        vis_image = image.copy()  # создание копии для отрисовки на ней bbox'ов
        faces = self.face_detector.detect(image)
        emotions = []
        for face in faces:
            emotion, conf = self.emotion_recognizer.predict(face['crop'])
            emotions.append((emotion, conf))
            x1, y1, x2, y2 = face['bbox']

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            text = f"{emotion}: {conf:.2f}"
            cv2.putText(vis_image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        return vis_image, emotions


class CaptureReadError(Exception): pass


def process_video_stream(video_stream: cv2.VideoCapture,
                         face_detector_and_emotion_recognizer: typing.Optional[DetectFaceAndRecognizeEmotion] = None, *,
                         flip_h: bool = False):
    """
    Обрабатывает видеопоток, находя лица и распознавая эмоции
    :param video_stream: Видео поток
    :param face_detector_and_emotion_recognizer: То, с помощью чего обрабатывается видеопоток
    :param flip_h: Отзеркалить входящий видеопоток
    :return: Генератор, который возвращает обработанный кадр и эмоции. Формат: (image, [(emotion, confidence), ...])
    """
    use_inner_models = face_detector_and_emotion_recognizer is None
    if use_inner_models:
        face_detector = FaceDetector(min_detection_confidence=0.5)
        emotion_recognizer = EmotionRecognizer(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            window_size=15,
            alpha=0.3,
            confidence_threshold=0.55,
            ambiguity_threshold=0.15
        )
        face_detector_and_emotion_recognizer = DetectFaceAndRecognizeEmotion(face_detector, emotion_recognizer)

    if not video_stream.isOpened():
        raise CaptureReadError('"video_stream" is not opened')
    try:
        while True:
            ret_val, img = video_stream.read()
            if not ret_val:
                raise CaptureReadError('Failed to get image from "video_stream"')
            if flip_h:
                img = cv2.flip(img, 1)
            new_img, emotions = face_detector_and_emotion_recognizer.detect_and_recognize(img)
            yield new_img, emotions
    finally:
        if use_inner_models:
            face_detector.close()
            emotion_recognizer.reset()
            del face_detector_and_emotion_recognizer


if __name__ == '__main__':
    print('Using camera 0')
    cap = cv2.VideoCapture(0)
    fps_history = deque()
    FPS_HISTORY_LEN = 3  # для более гладкого fps, будет выводится средние из последних FPS_HISTORY_LEN измерений

    for _ in range(FPS_HISTORY_LEN):
        fps_history.append(0.0)
    try:
        start_time = time()
        for img, emotions in process_video_stream(cap, flip_h=True):
            cv2.putText(img, f'FPS: {round(sum(fps_history) / FPS_HISTORY_LEN)}', (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.imshow('Test', img)
            if cv2.waitKey(1) & 0xFF == 27:
                break  # esc to quit
            fps = 1 / (time() - start_time)
            fps_history.append(fps)
            fps_history.popleft()
            start_time = time()
    finally:
        print('Releasing resources...')
        cap.release()
        cv2.destroyAllWindows()
        print('Done!')
