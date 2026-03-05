"""
Модуль конфигурации приложения.

Загружает настройки из переменных окружения и файла .env.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Настройки приложения для распознавания эмоций.

    Загружаются из переменных окружения или файла .env.
    Все параметры имеют значения по умолчанию для разработки.

    Attributes:
        app_version: Версия приложения
        cors_allowed_origins: Разрешенные источники для CORS
        face_detection_*: Параметры детекции лиц MediaPipe
        face_mesh_*: Параметры Face Mesh MediaPipe
        emotion_*: Параметры распознавания эмоций
        ear_*: Параметры анализа Eye Aspect Ratio
        head_pitch_*/head_yaw_*: Пороги для классификации позы головы
    """

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Settings
    # -------------------------------------------------------------------------
    app_version: str = "1.0.0"
    cors_allowed_origins: str = "http://localhost:8501,http://localhost:63342"

    # Face Detection Settings (MediaPipe)
    # -------------------------------------------------------------------------
    face_detection_min_confidence: float = 0.5
    face_detection_model_selection: int = 1
    face_detection_margin: int = 20

    # Face Mesh Settings (MediaPipe)
    # -------------------------------------------------------------------------
    face_mesh_max_num_faces: int = 5
    face_mesh_min_detection_confidence: float = 0.5
    face_mesh_min_tracking_confidence: float = 0.5

    # Emotion Recognition Settings (EmotiEffLib)
    # -------------------------------------------------------------------------
    emotion_model_name: str = "enet_b2_8"
    emotion_device: str = "auto"
    emotion_window_size: int = 15
    emotion_confidence_threshold: float = 0.55
    emotion_ambiguity_threshold: float = 0.15

    # Eye Aspect Ratio (EAR) Analysis Settings
    # -------------------------------------------------------------------------
    ear_threshold: float = 0.25
    ear_consec_frames: int = 3
    ear_history_maxlen: int = 30

    # Attention Classification Thresholds
    # -------------------------------------------------------------------------
    ear_alert_threshold: float = 0.30
    ear_drowsy_threshold: float = 0.20
    ear_very_drowsy_threshold: float = 0.15

    # Head pose thresholds (in degrees)
    head_pitch_highly_attentive: float = 10.0
    head_yaw_highly_attentive: float = 15.0
    head_pitch_attentive: float = 20.0
    head_yaw_attentive: float = 25.0
    head_pitch_distracted: float = 30.0
    head_yaw_distracted: float = 40.0

    # Redis Configuration
    # -------------------------------------------------------------------------
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_timeout: int = 5

    def get_cors_origins(self) -> list[str]:
        """
        Получает список разрешенных источников CORS.

        Returns:
            list[str]: Список разрешенных origins
        """
        return [origin.strip() for origin in self.cors_allowed_origins.split(",")]


settings = Settings()
