import logging
from dataclasses import dataclass

import cv2

from app.core.config import settings

from .analyze_ear import EyeAspectRatioAnalyzer, EyeAspectRatioAnalyzeResult
from .analyze_emotion import EmotionRecognizer
from .analyze_head_pose import HeadPoseEstimateResult, HeadPoseEstimator
from .engagement_calculator import EngagementCalculateResult, EngagementCalculator
from .face_detection import FaceDetector, mp_face_mesh

logger = logging.getLogger(__name__)


@dataclass
class OneFaceMetricsAnalyzeResult:
    emotion: str
    confidence: float
    bbox: tuple[int, int, int, int]
    ear: EyeAspectRatioAnalyzeResult | None
    head_pose: HeadPoseEstimateResult | None
    engagement: EngagementCalculateResult | None


@dataclass
class FaceAnalyzeResult:
    image: cv2.typing.MatLike
    metrics: list[OneFaceMetricsAnalyzeResult]


class FaceAnalysisPipeline:
    """Пайплайн для комплексного анализа лица (детекция + эмоции + EAR + HeadPose)"""

    # Цвета аннотаций (BGR)
    _COLOR_BBOX = (255, 0, 255)       # Magenta
    _COLOR_EMOTION = (255, 0, 255)    # Magenta
    _COLOR_EAR = (0, 255, 200)        # Бирюзовый
    _COLOR_BLINK = (0, 0, 255)        # Красный
    _COLOR_HEAD_POSE = (255, 200, 0)  # Голубой
    _COLOR_ENGAGEMENT = (0, 255, 0)   # Зелёный

    def __init__(
        self,
        face_detector: FaceDetector,
        emotion_recognizer: EmotionRecognizer,
        ear_analyzer: EyeAspectRatioAnalyzer | None = None,
        head_pose_estimator: HeadPoseEstimator | None = None,
        engagement_calculator: EngagementCalculator | None = None,
        use_face_mesh: bool = True,
    ):
        """
        Args:
            face_detector: Детектор лиц
            emotion_recognizer: Распознаватель эмоций
            ear_analyzer: EyeAspectRatioAnalyzer (опционально)
            head_pose_estimator: HeadPoseEstimator (опционально)
            engagement_calculator: EngagementCalculator (опционально)
            use_face_mesh: Использовать Face Mesh для EAR/HeadPose (требует больше ресурсов)
        """
        self.face_detector = face_detector
        self.emotion_recognizer = emotion_recognizer
        self.ear_analyzer = ear_analyzer
        self.head_pose_estimator = head_pose_estimator
        self.engagement_calculator = engagement_calculator
        self.use_face_mesh = use_face_mesh

        self.face_mesh = None
        if use_face_mesh and (ear_analyzer or head_pose_estimator):
            self._init_face_mesh()
        logger.debug("FaceAnalysisPipeline initialized")

    def _init_face_mesh(self):
        """Инициализирует Face Mesh для EAR/HeadPose анализа"""
        if self.face_mesh is None:
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=settings.face_mesh_max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=settings.face_mesh_min_detection_confidence,
                min_tracking_confidence=settings.face_mesh_min_tracking_confidence,
            )

    def _close_face_mesh(self):
        """Закрывает Face Mesh"""
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None

    def set_ear_analyzer(self, ear_analyzer):
        """Устанавливает или сбрасывает EAR анализатор (hot-reload)"""
        self.ear_analyzer = ear_analyzer
        # Инициализируем Face Mesh, если нужен
        if ear_analyzer or self.head_pose_estimator:
            self._init_face_mesh()
        elif not self.head_pose_estimator:
            self._close_face_mesh()

    def set_head_pose_estimator(self, head_pose_estimator):
        """Устанавливает или сбрасывает HeadPose анализатор (hot-reload)"""
        self.head_pose_estimator = head_pose_estimator
        # Инициализируем Face Mesh, если нужен
        if head_pose_estimator or self.ear_analyzer:
            self._init_face_mesh()
        elif not self.ear_analyzer:
            self._close_face_mesh()

    def analyze(self, image: cv2.typing.MatLike) -> FaceAnalyzeResult:
        """
        Детектирует лица и распознаёт эмоции (опционально - EAR и HeadPose).

        Args:
            image: Входное изображение с лицами для анализа

        Returns:
            FaceAnalyzeResult: Результат анализа, содержащий:
                - image: Изображение с отрисованными bbox'ами и метриками
                - metrics: Список OneFaceMetricsAnalyzeResult для каждого обнаруженного лица
        """
        vis_image = image.copy()
        try:
            faces = self.face_detector.detect(image)
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            faces = []

        face_mesh_results = None
        if self.face_mesh:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                face_mesh_results = self.face_mesh.process(rgb_image)
            except Exception as e:
                logger.error(f"Face mesh processing failed: {e}")

        results = []
        h, w, _ = image.shape

        for face_idx, face in enumerate(faces):
            try:
                prediction = self.emotion_recognizer.predict(face.crop)
                emotion = prediction.label
                conf = prediction.confidence
            except Exception as e:
                logger.error(f"Emotion recognition failed for face {face_idx}: {e}")
                emotion = "unknown"
                conf = 0.0

            x1, y1, x2, y2 = face.bbox

            if self.ear_analyzer and face_mesh_results and face_mesh_results.multi_face_landmarks:
                if face_idx < len(face_mesh_results.multi_face_landmarks):
                    face_landmarks = face_mesh_results.multi_face_landmarks[face_idx]
                    try:
                        ear_result = self.ear_analyzer.analyze(face_landmarks, w, h, face_idx)
                        ear = ear_result
                    except Exception as e:
                        logger.error(f"EAR analysis failed for face {face_idx}: {e}")
                        ear = None
                else:
                    ear = None
            else:
                ear = None

            if self.head_pose_estimator and face_mesh_results and face_mesh_results.multi_face_landmarks:
                if face_idx < len(face_mesh_results.multi_face_landmarks):
                    face_landmarks = face_mesh_results.multi_face_landmarks[face_idx]
                    try:
                        head_pose_result = self.head_pose_estimator.estimate(face_landmarks, w, h)
                        head_pose = head_pose_result
                    except Exception as e:
                        logger.error(f"Head pose estimation failed for face {face_idx}: {e}")
                        head_pose = None
                else:
                    head_pose = None
            else:
                head_pose = None

            engagement = None
            if self.engagement_calculator:
                try:
                    engagement = self.engagement_calculator.calculate(
                        emotion=emotion,
                        emotion_confidence=conf,
                        ear_data=ear,
                        head_pose_data=head_pose,
                    )
                except Exception as e:
                    logger.error(f"Engagement calculation failed for face {face_idx}: {e}")

            result = OneFaceMetricsAnalyzeResult(
                emotion=emotion,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                ear=ear,
                head_pose=head_pose,
                engagement=engagement,
            )

            self._draw_face_info(vis_image, result)

            results.append(result)

        return FaceAnalyzeResult(vis_image, results)

    @classmethod
    def _draw_face_info(cls, image: cv2.typing.MatLike, result: OneFaceMetricsAnalyzeResult):
        """Отрисовывает информацию о лице на изображении"""
        x1, y1, x2, y2 = result.bbox

        # Рисуем bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), cls._COLOR_BBOX, 2)

        # Эмоция
        emotion_text = f"{result.emotion}: {result.confidence:.2f}"
        cv2.putText(image, emotion_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cls._COLOR_EMOTION, 2)

        # EAR (если доступен)
        y_offset = y1 - 30
        if result.ear:
            ear_text = f"EAR: {result.ear.avg_ear:.3f} [{result.ear.attention_state}]"
            cv2.putText(image, ear_text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls._COLOR_EAR, 1)
            y_offset -= 15

            if result.ear.is_blinking:
                blink_text = "BLINK"
                cv2.putText(image, blink_text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls._COLOR_BLINK, 1)
                y_offset -= 15

        # HeadPose (если доступен)
        if result.head_pose:
            pitch = result.head_pose.pitch
            yaw = result.head_pose.yaw
            roll = result.head_pose.roll
            hp_text = f"P:{pitch:.0f} Y:{yaw:.0f} R:{roll:.0f} [{result.head_pose.attention_state}]"
            cv2.putText(image, hp_text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls._COLOR_HEAD_POSE, 1)
            y_offset -= 15

        # Получение результирующего engagement, если метрика доступна
        if result.engagement:
            score = result.engagement.score
            level = result.engagement.level
            engagement_text = f"Eng: {score:.2f} ({level})"
            cv2.putText(image, engagement_text, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cls._COLOR_ENGAGEMENT, 1)


def make_face_analysis_pipeline() -> FaceAnalysisPipeline:
    return FaceAnalysisPipeline(
        face_detector=FaceDetector(),
        emotion_recognizer=EmotionRecognizer(),
        ear_analyzer=EyeAspectRatioAnalyzer(),
        head_pose_estimator=HeadPoseEstimator(),
        engagement_calculator=EngagementCalculator(),
    )
