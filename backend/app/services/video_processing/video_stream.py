import cv2

from .analyze_emotion import EmotionRecognizer
from .face_analysis_pipeline import FaceAnalysisPipeline
from .face_detection import FaceDetector


class CaptureReadError(Exception):
    """
    Исключение, выбрасываемое при ошибке чтения кадра из видеопотока.

    Attributes:
        message: Сообщение об ошибке
    """

    pass


def process_video_stream(
    video_stream: cv2.VideoCapture,
    face_analyze_pipeline: FaceAnalysisPipeline | None = None,
    *,
    flip_h: bool = False,
):
    """
    Обрабатывает видеопоток, находя лица и распознавая эмоции
    :param video_stream: Видео поток
    :param face_analyze_pipeline: То, с помощью чего обрабатывается видеопоток
    :param flip_h: Отзеркалить входящий видеопоток
    :return: Генератор, который возвращает обработанный кадр и эмоции. Формат: (image, [(emotion, confidence), ...])
    """
    use_inner_models = face_analyze_pipeline is None
    if use_inner_models:
        face_detector = FaceDetector()
        emotion_recognizer = EmotionRecognizer()
        face_analyze_pipeline = FaceAnalysisPipeline(face_detector, emotion_recognizer)

    if not video_stream.isOpened():
        raise CaptureReadError('"video_stream" is not opened')
    try:
        while True:
            ret_val, img = video_stream.read()
            if not ret_val:
                raise CaptureReadError('Failed to get image from "video_stream"')
            if flip_h:
                img = cv2.flip(img, 1)
            assert face_analyze_pipeline is not None
            analysis_result = face_analyze_pipeline.analyze(img)
            emotions = []
            for metric in analysis_result.metrics:
                emotion_dict: dict = {
                    "emotion": metric.emotion,
                    "confidence": metric.confidence,
                }
                if metric.ear:
                    emotion_dict["ear"] = {
                        "avg_ear": metric.ear.avg_ear,
                        "eyes_open": not metric.ear.is_blinking,
                        "blink_count": metric.ear.blink_count,
                    }
                if metric.head_pose:
                    emotion_dict["head_pose"] = {
                        "pitch": metric.head_pose.pitch,
                        "yaw": metric.head_pose.yaw,
                        "roll": metric.head_pose.roll,
                    }
                emotions.append(emotion_dict)
            yield analysis_result.image, emotions
    finally:
        if use_inner_models:
            face_detector.close()
            emotion_recognizer.reset()
            del face_analyze_pipeline
