from typing import Literal

from pydantic import BaseModel


class FrameRequest(BaseModel):
    """Входящие WS-сообщение - кадр от клиента"""

    image: str


class EARResult(BaseModel):
    """Результат анализа Eye Aspect Ratio"""

    left_ear: float
    right_ear: float
    avg_ear: float
    eyes_open: bool
    blink_count: int
    is_blinking: bool
    ear_history: list[float] | None = None
    attention_state: Literal["Alert", "Normal", "Drowsy", "Very Drowsy"] = "Normal"


class HeadPoseResult(BaseModel):
    """Результат анализа позиции головы (Head Pose)"""

    pitch: float
    yaw: float
    roll: float
    rotation_vec: tuple[float, float, float]
    translation_vec: tuple[float, float, float]
    attention_state: Literal["Highly Attentive", "Attentive", "Distracted", "Very Distracted"]


class EngagementComponents(BaseModel):
    """Критерии оценивания вовлеченности"""

    emotion_score: float
    eye_score: float
    head_pose_score: float


class EngagementResult(BaseModel):
    """Результат расчета вовлеченности"""

    score: float
    score_raw: float
    level: Literal["High", "Medium", "Low", "Very Low"]
    trend: Literal["rising", "falling", "stable"]
    components: EngagementComponents
    frame_count: int


class FaceAnalysisResult(BaseModel):
    """Результат анализа лица в целом"""

    emotion: str
    confidence: float
    bbox: tuple[int, int, int, int]
    ear: EARResult | None
    head_pose: HeadPoseResult | None
    engagement: EngagementResult | None


class FrameResponse(BaseModel):
    """Исходящее WS-сообщение - обработанный кадр (stream endpoint)"""

    image: str
    results: list[FaceAnalysisResult]


class OutputStreamFrameResponse(BaseModel):
    """Исходящее WS-сообщение - кадр для гипервизора (output_stream endpoint)"""

    image_src: str
    image: str
    results: list[FaceAnalysisResult]


class ErrorResponse(BaseModel):
    """Сообщение об ошибке"""

    error: str
