from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
class FrameRequest(BaseModel):
    image: str
    params: dict = Field(default\_factory=dict)
class EARResult(BaseModel):
    left_ear: float
    right_ear: float
    avg_ear: float
    eyes_open: bool
    blink_count: int
    is_blinking: bool
    attention_state: Literal
class HeadPoseResult(BaseModel):
    pitch: float
    yaw: float
    roll: float
    attention_state: Literal
class EngagementComponents(BaseModel):
    emotion_score: float
    eye_score: float
    head_pose_score: float
class EngagementResult(BaseModel):
    score: float
    score_raw: float
    level: Literal
    trend: Literal
    components: EngagementComponents
    frame_count: int
class FaceAnalysisResult(BaseModel):
    emotion: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    ear: Optional[EARResult]
    head_pose: Optional[HeadPoseResult]
    engagement: Optional[EngagementResult]
class FrameResponse(BaseModel):
    image: str
    results: List\[FaceAnalysisResult]
    metadata: dict = Field(default\_factory=dict)
class ErrorResponse(BaseModel):
    error: str
    code: str
