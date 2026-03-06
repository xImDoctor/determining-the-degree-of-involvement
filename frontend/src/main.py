import atexit
import os
import sys
import tempfile
from collections import deque
from pathlib import Path
from time import time as current_time

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.append('../backend')

try:
    from app.services.video_processing import (
        FaceDetector,
        EmotionRecognizer,
        FaceAnalysisPipeline
    )
    from app.services.video_processing import CaptureReadError, process_video_stream

    BACKEND_AVAILABLE = True
except ImportError as e:
    st.error(f"Не удалось импортировать модуль бэкенда: {e}")
    BACKEND_AVAILABLE = False

# Импорт модулей EAR и HeadPose (доп.)
EAR_HEADPOSE_AVAILABLE = False
try:
    from app.services.video_processing import EyeAspectRatioAnalyzer
    from app.services.video_processing import HeadPoseEstimator

    EAR_HEADPOSE_AVAILABLE = True
except ImportError:
    pass

APP_TITLE = "Распознавание эмоций в реальном времени"
APP_ICON = "🎭"
SUPPORTED_FORMATS = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv']
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# ============================================
# CSS СТИЛИ
# ============================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


# Загрузка внешних CSS стилей
def load_css():
    """Загружает внешний CSS файл"""
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning("Файл стилей styles.css не найден")


load_css()


# ============================================
# КЛАССЫ ДЛЯ ОБРАБОТКИ ВИДЕО
# ============================================

class EmotionDetectionProcessor:
    """Класс для обработки видео с использованием FaceAnalysisPipeline"""

    def __init__(self):
        self.detector = None
        self.is_initialized = False
        self.current_emotions = []

    def initialize_models(self, params):
        """
        Инициализирует модели для распознавания эмоций. 
        Дополнительно инициализирует EAR и HeadPose Estimation, если импортированы.
        """
        try:
            if BACKEND_AVAILABLE:
                # Детектор лиц
                face_detector = FaceDetector(min_detection_confidence=params.get('min_detection_confidence', 0.5))

                # Распознаватель эмоций
                emotion_recognizer = EmotionRecognizer(window_size=params.get('window_size', 15),
                                                       confidence_threshold=params.get('confidence_threshold', 0.55),
                                                       ambiguity_threshold=params.get('ambiguity_threshold', 0.15))

                # EAR анализатор (доп.)
                ear_analyzer = None
                if EAR_HEADPOSE_AVAILABLE and params.get('enable_ear', False):
                    ear_analyzer = EyeAspectRatioAnalyzer(
                        ear_threshold=params.get('ear_threshold', 0.25),
                        consec_frames=params.get('consec_frames', 3)
                    )

                # Head Pose анализатор (доп.)
                head_pose_estimator = None
                if EAR_HEADPOSE_AVAILABLE and params.get('enable_head_pose', False):
                    head_pose_estimator = HeadPoseEstimator()

                # Основной детектор
                self.detector = FaceAnalysisPipeline(
                    face_detector,
                    emotion_recognizer,
                    ear_analyzer=ear_analyzer,
                    head_pose_estimator=head_pose_estimator,
                    use_face_mesh=(ear_analyzer is not None or head_pose_estimator is not None)
                )

                self.is_initialized = True
                return True, "Модели успешно инициализированы"
            else:
                return False, "Модуль бэкенда не доступен"

        except Exception as e:
            return False, f"Ошибка инициализации моделей: {str(e)}"

    def process_frame(self, frame, flip_h=False):
        """Обрабатывает один кадр видео"""
        if not self.is_initialized or self.detector is None:
            return frame, []

        try:
            # Отзеркаливание если нужно
            if flip_h:
                frame = cv2.flip(frame, 1)

            # Обработка кадра с помощью детектора
            analysis_result = self.detector.analyze(frame)

            processed_frame = analysis_result.image
            results = []
            for metric in analysis_result.metrics:
                result_dict = {
                    'emotion': metric.emotion,
                    'confidence': metric.confidence,
                    'bbox': metric.bbox,
                }
                if metric.ear:
                    result_dict['ear'] = {
                        'avg_ear': metric.ear.avg_ear,
                        'eyes_open': not metric.ear.is_blinking,
                        'blink_count': metric.ear.blink_count
                    }
                if metric.head_pose:
                    result_dict['head_pose'] = {
                        'pitch': metric.head_pose.pitch,
                        'yaw': metric.head_pose.yaw,
                        'roll': metric.head_pose.roll,
                        'attention_state': getattr(metric.head_pose, 'attention_state', None)
                    }
                results.append(result_dict)

            self.current_emotions = results
            return processed_frame, results

        except Exception as e:
            st.error(f"Ошибка обработки кадра: {e}")
            return frame, []

    def get_emotion_statistics(self):
        """Получает статистику по распознанным эмоциям"""
        if not self.current_emotions:
            return {}

        stats = {}
        for emotion, confidence in self.current_emotions:
            if emotion in stats:
                stats[emotion] += 1
            else:
                stats[emotion] = 1

        return stats

    def reset(self):
        """Сбрасывает состояние процессора"""
        self.current_emotions = []
        if hasattr(self.detector, 'face_detector'):
            if hasattr(self.detector.face_detector, 'close'):
                self.detector.face_detector.close()
        if hasattr(self.detector, 'emotion_recognizer'):
            if hasattr(self.detector.emotion_recognizer, 'reset'):
                self.detector.emotion_recognizer.reset()


class VideoFileProcessor:
    """Класс для обработки видеофайлов"""

    def __init__(self):
        self.detection_processor = EmotionDetectionProcessor()

    def extract_video_info(self, video_path):
        """Извлекает информацию о видео"""
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return {"error": "Cannot open video file"}

            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": int(cap.get(cv2.CAP_PROP_FPS)),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS), 2) if cap.get(
                    cv2.CAP_PROP_FPS) > 0 else 0,
                "format": self._get_video_format(video_path)
            }

            # Извлекаем превью
            ret, frame = cap.read()
            if ret:
                preview_path = "temp_preview.jpg"
                cv2.imwrite(preview_path, frame)
                info["preview"] = preview_path

            cap.release()
            return info

        except Exception as e:
            return {"error": f"Cannot extract video info: {str(e)}"}

    def _get_video_format(self, video_path):
        """Определяет формат видео"""
        ext = os.path.splitext(video_path)[1].lower()
        formats = {
            '.mp4': 'MP4',
            '.avi': 'AVI',
            '.mov': 'MOV',
            '.mkv': 'MKV',
            '.webm': 'WebM',
            '.wmv': 'WMV'
        }
        return formats.get(ext, 'Unknown')

    def process_video_file(self, input_path, output_path, params, progress_callback=None):
        """
        Обрабатывает видеофайл с распознаванием эмоций

        Args:
            input_path: Путь к входному видео
            output_path: Путь для сохранения результата
            params: Параметры обработки
            progress_callback: Функция для обновления прогресса

        Returns:
            (success, message, output_path, statistics)
        """
        try:
            # Инициализируем модели
            success, message = self.detection_processor.initialize_models(params)
            if not success:
                return False, message, None, {}

            # Открываем видео
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return False, f"Cannot open video file: {input_path}", None, {}

            # Получаем параметры видео
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:
                fps = 30

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Создаем VideoWriter для выходного видео
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # Статистика
            all_emotions = []
            frame_count = 0

            # Обрабатываем  кадр
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Обрабатываем кадр
                processed_frame, emotions = self.detection_processor.process_frame(
                    frame,
                    flip_h=params.get('flip_h', False)
                )

                # Записываем обработанный кадр
                out.write(processed_frame)

                # Сохраняем эмоции для статистики
                for result in emotions:
                    all_emotions.append(result['emotion'])

                frame_count += 1

                # Обновляем прогресс
                if progress_callback and total_frames > 0:
                    progress = frame_count / total_frames
                    progress_callback(progress, frame_count, total_frames, emotions)

            # Закрываем видео
            cap.release()
            out.release()

            # Собираем статистику
            statistics = self._calculate_statistics(all_emotions, frame_count)

            return True, "Обработка завершена успешно", output_path, statistics

        except Exception as e:
            return False, f"Ошибка обработки видео: {str(e)}", None, {}

    def _calculate_statistics(self, all_emotions, total_frames):
        """Рассчитывает статистику по эмоциям"""
        if not all_emotions:
            return {}

        stats = {}
        for emotion in all_emotions:
            if emotion in stats:
                stats[emotion] += 1
            else:
                stats[emotion] = 1

        # Добавляем проценты
        total_detections = len(all_emotions)
        if total_detections > 0:
            for emotion in stats:
                stats[f"{emotion}_percent"] = (stats[emotion] / total_detections) * 100

        stats['total_frames'] = total_frames
        stats['total_detections'] = total_detections
        stats['detection_rate'] = (total_detections / total_frames) * 100 if total_frames > 0 else 0

        return stats

    def extract_sample_frames(self, video_path, num_frames=4):
        """Извлекает несколько кадров из видео для превью"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                return []

            # Выбираем равномерно распределенные кадры
            frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Конвертируем BGR в RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

            cap.release()
            return frames

        except Exception:
            return []


# ============================================
# ИНИЦИАЛИЗАЦИЯ СЕССИИ
# ============================================

if 'video_processor' not in st.session_state:
    st.session_state.video_processor = VideoFileProcessor()

if 'detection_processor' not in st.session_state:
    st.session_state.detection_processor = EmotionDetectionProcessor()

if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None

if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "idle"

if 'result_path' not in st.session_state:
    st.session_state.result_path = None

if 'video_info' not in st.session_state:
    st.session_state.video_info = {}

if 'processing_progress' not in st.session_state:
    st.session_state.processing_progress = 0

if 'current_emotions' not in st.session_state:
    st.session_state.current_emotions = []

if 'emotion_statistics' not in st.session_state:
    st.session_state.emotion_statistics = {}

if 'backend_params' not in st.session_state:
    # Параметры для обработки
    st.session_state.backend_params = {
        'min_detection_confidence': 0.5,
        'window_size': 15,
        'confidence_threshold': 0.55,
        'ambiguity_threshold': 0.15,
        'margin': 20,
        'flip_h': False,
        'show_preview': False,
        'enable_ear': False,  # Включить EAR анализ
        'enable_head_pose': False,  # Включить Head Pose анализ
        'ear_threshold': 0.25,  # Порог EAR для закрытых глаз
        'consec_frames': 1  # Количество последовательных кадров для моргания (1 = любое закрытие глаз)
    }

if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False

if 'webcam_detector' not in st.session_state:
    st.session_state.webcam_detector = None

if 'prev_webcam_params' not in st.session_state:
    st.session_state.prev_webcam_params = None

if 'vis_params' not in st.session_state:
    st.session_state.vis_params = {
        'show_fps': True,        # FPS-счётчик на кадре
        'show_emotions': True,   # блок текста с эмоциями под видео
        'show_ear_info': True,   # EAR-данные в блоке эмоций
        'show_hpe_info': True,   # HPE-данные в блоке эмоций
    }


# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

def display_header():
    """Отображает заголовок приложения"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f'<h1 class="main-header">{APP_TITLE}</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">Загрузите видео или используйте веб-камеру для детекции лиц и распознавания эмоций в реальном времени</p>',
            unsafe_allow_html=True)


def display_sidebar():
    """Отображает боковую панель с настройками"""
    with st.sidebar:
        st.markdown("### 🎭 Распознавание эмоций")

        # Информация о системе
        st.markdown("#### ℹ️ Состояние системы")

        if BACKEND_AVAILABLE:
            st.success("✅ Модуль бэкенда доступен")
        else:
            st.error("❌ Модуль бэкенда не найден")
            st.info("Убедитесь, что файл face_detection.py находится в текущей директории")

        st.markdown("---")

        # Настройки параметров
        st.markdown("#### ⚙️ Параметры обработки")

        # Face Detector Parameters
        st.markdown("##### Детекция лиц")
        st.session_state.backend_params['min_detection_confidence'] = st.slider(
            "Минимальная уверенность детекции (min_detection_confidence)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.backend_params['min_detection_confidence'],
            step=0.01,
            help="Минимальная уверенность для детекции лиц"
        )

        # Emotion Recognizer Parameters
        st.markdown("##### Распознавание эмоций")

        st.session_state.backend_params['window_size'] = st.slider(
            "Размер окна сглаживания (window_size)",
            min_value=3,
            max_value=30,
            value=st.session_state.backend_params['window_size'],
            step=1,
            help="Размер окна для temporal smoothing (взвешенное усреднение по истории кадров)"
        )

        st.session_state.backend_params['confidence_threshold'] = st.slider(
            "Порог уверенности (confidence_threshold)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.backend_params['confidence_threshold'],
            step=0.01,
            help="Минимальный порог уверенности для эмоции (ниже → fallback к Neutral)"
        )

        st.session_state.backend_params['ambiguity_threshold'] = st.slider(
            "Порог амбивалентности (ambiguity_threshold)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.backend_params['ambiguity_threshold'],
            step=0.01,
            help="Порог для амбивалентных эмоций (если две эмоции слишком близки → Neutral)"
        )

        # Настройка EAR и HeadPose
        if EAR_HEADPOSE_AVAILABLE:
            st.markdown("##### Дополнительные модули")

            st.session_state.backend_params['enable_ear'] = st.checkbox(
                "Включить Eye Aspect Ratio (enable_ear)",
                value=st.session_state.backend_params['enable_ear'],
                help="Анализ состояния глаз и моргания (требует больше ресурсов)"
            )

            if st.session_state.backend_params['enable_ear']:
                st.session_state.backend_params['ear_threshold'] = st.slider(
                    "Порог EAR (ear_threshold)",
                    min_value=0.10,
                    max_value=0.40,
                    value=st.session_state.backend_params['ear_threshold'],
                    step=0.01,
                    help="Порог для определения закрытых глаз (меньше → строже)"
                )

                st.session_state.backend_params['consec_frames'] = st.slider(
                    "Кадров для моргания (consec_frames)",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.backend_params['consec_frames'],
                    step=1,
                    help="Минимум кадров с закрытыми глазами для засчитывания моргания"
                )

            st.session_state.backend_params['enable_head_pose'] = st.checkbox(
                "Включить Head Pose Estimation (enable_head_pose)",
                value=st.session_state.backend_params['enable_head_pose'],
                help="Оценка направления взгляда и позы головы"
            )

        # Общие параметры
        st.markdown("##### Общие настройки")
        st.session_state.backend_params['flip_h'] = st.checkbox(
            "Отразить по горизонтали (flip_h)",
            value=st.session_state.backend_params['flip_h'],
            help="Отзеркаливание изображения по горизонтали (для веб-камеры)"
        )

        st.session_state.backend_params['show_preview'] = st.checkbox(
            "Показывать превью (show_preview)",
            value=st.session_state.backend_params['show_preview'],
            help="Показывать превью кадров во время обработки видео"
        )

        # Настройки визуализации
        st.markdown("##### Визуализация")

        st.session_state.vis_params['show_fps'] = st.checkbox(
            "Показывать FPS",
            value=st.session_state.vis_params['show_fps'],
            help="Счётчик FPS на кадре"
        )

        st.session_state.vis_params['show_emotions'] = st.checkbox(
            "Показывать эмоции",
            value=st.session_state.vis_params['show_emotions'],
            help="Блок с текстовыми данными об эмоциях под видео"
        )

        if EAR_HEADPOSE_AVAILABLE:
            st.session_state.vis_params['show_ear_info'] = st.checkbox(
                "Показывать данные EAR",
                value=st.session_state.vis_params['show_ear_info'],
                help="EAR-данные в блоке эмоций (только если EAR включён)"
            )

            st.session_state.vis_params['show_hpe_info'] = st.checkbox(
                "Показывать данные HPE",
                value=st.session_state.vis_params['show_hpe_info'],
                help="Данные Head Pose Estimation в блоке эмоций (только если HPE включён)"
            )

        st.markdown("---")

        # Кнопка сброса параметров
        if st.button("🔄 Сбросить по умолчанию", width='stretch'):
            st.session_state.backend_params = {
                'min_detection_confidence': 0.5,
                'window_size': 15,
                'confidence_threshold': 0.55,
                'ambiguity_threshold': 0.15,
                'flip_h': False,
                'show_preview': False
            }
            st.rerun()


def create_upload_section():
    """Создает секцию загрузки файла"""
    st.markdown("### 📤 Загрузка видео")

    # Зона загрузки
    uploaded_file = st.file_uploader(
        "Выберите видеофайл",
        type=SUPPORTED_FORMATS,
        help=f"Поддерживаемые форматы: {', '.join(SUPPORTED_FORMATS).upper()}",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Проверяем размер файла
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"Файл слишком большой! Максимальный размер: {MAX_FILE_SIZE // (1024 * 1024)}МБ")
            return

        # Сохраняем временный файл
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.session_state.uploaded_file_path = temp_path

        # Извлекаем информацию о видео
        video_info = st.session_state.video_processor.extract_video_info(temp_path)
        st.session_state.video_info = video_info

        if "error" not in video_info:
            # Показываем информацию о файле
            display_file_info(uploaded_file, video_info)

            # Показываем превью видео
            display_video_preview(temp_path, video_info)

            # Кнопка обработки
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Начать распознавание эмоций", type="primary", width='stretch'):
                    if not BACKEND_AVAILABLE:
                        st.error(
                            "Невозможно начать обработку: Backend модуль недоступен. Убедитесь, что файл face_detection.py находится в текущей директории со всеми необходимыми зависимостями.")
                    else:
                        st.session_state.processing_status = "starting"
                        st.rerun()
            with col2:
                if st.button("🗑️ Очистить файл", width='stretch'):
                    st.session_state.uploaded_file_path = None
                    st.rerun()
        else:
            st.error(f"Ошибка: {video_info['error']}")

    else:
        # Показываем подсказку
        st.info("Загрузите видеофайл, для этого выберите его через **Browse Files** или переместите в область выше.")
        st.markdown("""
        **Поддерживаемые форматы:** MP4, AVI, MOV, MKV, WebM, WMV
                    
        **Максимальный размер:** 100MB
        """)


def display_file_info(uploaded_file, video_info):
    """Отображает информацию о файле"""
    st.markdown("### 📊 Информация о видео")

    # Статистика в карточках
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Разрешение",
            value=f"{video_info['width']}×{video_info['height']}"
        )

    with col2:
        duration = video_info["duration"]
        if duration >= 60:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            duration_str = f"{minutes}:{seconds:02d}"
        else:
            duration_str = f"{duration:.1f}с"

        st.metric(
            label="Длительность",
            value=duration_str
        )

    with col3:
        st.metric(
            label="FPS",
            value=video_info["fps"]
        )

    with col4:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.metric(
            label="Размер (МБ)",
            value=f"{file_size_mb:.1f}"
        )


def display_video_preview(video_path, video_info):
    """Отображает превью видео"""
    st.markdown("### 👀 Превью видео")

    # Основное видео
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    video_bytes = open(video_path, "rb").read()
    st.video(video_bytes)
    st.markdown('</div>', unsafe_allow_html=True)

    # Примеры кадров
    st.markdown("#### 📸 Примеры кадров")

    frames = st.session_state.video_processor.extract_sample_frames(video_path, 4)
    if frames:
        cols = st.columns(4)
        for idx, (col, frame) in enumerate(zip(cols, frames)):
            with col:
                img = Image.fromarray(frame)
                st.image(img, caption=f"Кадр {idx + 1}", width='stretch')


def process_video():
    """Обрабатывает видео"""
    if st.session_state.processing_status == "starting" and st.session_state.uploaded_file_path:
        st.session_state.processing_status = "processing"

        # Показываем панель прогресса
        st.markdown('<div class="processing-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Обработка вашего видео")

        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_text = st.empty()

        # Функция обратного вызова для обновления прогресса
        def update_progress(progress, current_frame, total_frames, emotions):
            progress_bar.progress(progress)
            status_text.text(f"Обработка кадра {current_frame} из {total_frames} ({progress * 100:.1f}%)")

            # Обновление статистики
            if emotions:
                emotion_stats = {}
                for result in emotions:
                    emotion = result['emotion']
                    emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1

                stats_text.markdown("**Текущие эмоции:** " + ", ".join([f"{k}: {v}" for k, v in emotion_stats.items()]))

        # Создаем имя для выходного файла
        input_path = st.session_state.uploaded_file_path
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = int(current_time())
        output_filename = f"emotion_detected_{input_name}_{timestamp}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        # Запускаем обработку
        success, message, result_path, statistics = st.session_state.video_processor.process_video_file(
            input_path,
            output_path,
            st.session_state.backend_params,
            update_progress
        )

        if success:
            st.session_state.processing_status = "completed"
            st.session_state.result_path = result_path
            st.session_state.emotion_statistics = statistics
        else:
            st.session_state.processing_status = "failed"
            st.session_state.error_message = message

        st.markdown('</div>', unsafe_allow_html=True)

        # Показываем результат
        if st.session_state.processing_status == "completed":
            display_result()
        elif st.session_state.processing_status == "failed":
            display_error()


def display_result():
    """Отображает результат обработки"""
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### ✅ Обработка завершена!")

    result_path = st.session_state.result_path

    if result_path and os.path.exists(result_path):
        # Показываем обработанное видео
        st.markdown("#### 🎬 Обработанное видео")
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(result_path)
        st.markdown('</div>', unsafe_allow_html=True)

        # статистика эмоций
        if st.session_state.emotion_statistics:
            st.markdown("#### 📊 Статистика эмоций")

            # Фильтруем только эмоции (исключаем технические поля)
            emotion_stats = {k: v for k, v in st.session_state.emotion_statistics.items()
                             if not k.endswith('_percent') and k not in ['total_frames', 'total_detections',
                                                                         'detection_rate']}

            if emotion_stats:
                cols = st.columns(len(emotion_stats))
                for idx, (emotion, count) in enumerate(emotion_stats.items()):
                    with cols[idx % len(cols)]:
                        percent_key = f"{emotion}_percent"
                        percent = st.session_state.emotion_statistics.get(percent_key, 0)
                        st.metric(emotion.capitalize(), f"{count} ({percent:.1f}%)")

            # Общая статистика
            st.markdown("#### 📈 Общая статистика")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Всего кадров", st.session_state.emotion_statistics.get('total_frames', 0))
            with col2:
                st.metric("Детекций лиц", st.session_state.emotion_statistics.get('total_detections', 0))
            with col3:
                st.metric("Процент детекции", f"{st.session_state.emotion_statistics.get('detection_rate', 0):.1f}%")

        # Кнопка скачивания
        with open(result_path, "rb") as f:
            st.download_button(
                label="📥 Скачать обработанное видео",
                data=f,
                file_name=os.path.basename(result_path),
                mime="video/mp4",
                type="primary",
                width='stretch'
            )

    else:
        st.warning("Файл обработанного видео не найден")

    # Кнопка для новой обработки
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Обработать другое видео", width='stretch'):
            st.session_state.uploaded_file_path = None
            st.session_state.processing_status = "idle"
            st.session_state.result_path = None
            st.session_state.emotion_statistics = {}
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def display_error():
    """Отображает ошибку обработки"""
    st.markdown('<div class="error-card">', unsafe_allow_html=True)
    st.markdown("### ❌ Обработка не удалась")

    error_msg = getattr(st.session_state, 'error_message', 'Неизвестная ошибка')
    st.error(f"Ошибка: {error_msg}")

    # Советы по устранению неполадок
    st.markdown("#### 🔧 Советы по устранению неполадок:")
    st.markdown("""
    1. ✅ Убедитесь, что файл `face_detection.py` находится в той же директории
    2. ✅ Проверьте, что все зависимости установлены
    3. ✅ Попробуйте использовать более короткое видео (менее 1 минуты)
    4. ✅ Убедитесь, что формат видео поддерживается
    5. ✅ Проверьте наличие свободного места на диске
    """)

    if st.button("🔄 Попробовать снова", width='stretch'):
        st.session_state.processing_status = "idle"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def create_webcam_section():
    """Создает секцию работы с веб-камерой"""
    st.markdown("### 📷 Распознавание эмоций через веб-камеру")

    if not BACKEND_AVAILABLE:
        st.warning(
            "Распознавание эмоций через веб-камеру требует модуль бэкенда. Убедитесь, что файл face_detection.py доступен.")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        # Кнопки управления
        control_col1, control_col2 = st.columns(2)

        with control_col1:
            start_webcam = st.button("🎬 Запустить веб-камеру", type="primary", width='stretch')

        with control_col2:
            stop_webcam = st.button("⏹️ Остановить веб-камеру", type="secondary", width='stretch')

        # Место для отображения видео
        webcam_placeholder = st.empty()
        emotions_placeholder = st.empty()

        # Статистика
        stats_placeholder = st.empty()
        fps_placeholder = st.empty()

        # Состояние веб-камеры
        if start_webcam:
            st.session_state.webcam_running = True

        if stop_webcam:
            st.session_state.webcam_running = False

        # Запускаем веб-камеру
        if st.session_state.get('webcam_running', False):
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("Не удалось открыть веб-камеру")
                st.session_state.webcam_running = False
            else:
                # Настройки камеры
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                # Инициализируем детектор, если его нет
                if st.session_state.webcam_detector is None:
                    params = st.session_state.backend_params

                    face_detector = FaceDetector(min_detection_confidence=params.get('min_detection_confidence', 0.5))
                    emotion_recognizer = EmotionRecognizer(window_size=params['window_size'],
                                                           confidence_threshold=params['confidence_threshold'],
                                                           ambiguity_threshold=params['ambiguity_threshold'])

                    # EAR анализатор (доп.)
                    ear_analyzer = None
                    if EAR_HEADPOSE_AVAILABLE and params.get('enable_ear', False):
                        ear_analyzer = EyeAspectRatioAnalyzer(
                            ear_threshold=params.get('ear_threshold', 0.25),
                            consec_frames=params.get('consec_frames', 3)
                        )

                    # Head Pose анализатор (доп.)
                    head_pose_estimator = None
                    if EAR_HEADPOSE_AVAILABLE and params.get('enable_head_pose', False):
                        head_pose_estimator = HeadPoseEstimator()

                    st.session_state.webcam_detector = FaceAnalysisPipeline(
                        face_detector,
                        emotion_recognizer,
                        ear_analyzer=ear_analyzer,
                        head_pose_estimator=head_pose_estimator,
                        use_face_mesh=(ear_analyzer is not None or head_pose_estimator is not None)
                    )
                    st.session_state.prev_webcam_params = params.copy()

                # Hot-reload: применяем изменения параметров ДО цикла (при каждом rerune скрипта)
                elif st.session_state.prev_webcam_params != st.session_state.backend_params:
                    detector = st.session_state.webcam_detector
                    params = st.session_state.backend_params
                    prev_params = st.session_state.prev_webcam_params or {}

                    # Обновление FaceDetector
                    if params['min_detection_confidence'] != prev_params.get('min_detection_confidence'):
                        detector.face_detector.set_min_detection_confidence(params['min_detection_confidence'])

                    # Обновление EmotionRecognizer
                    if params['window_size'] != prev_params.get('window_size'):
                        detector.emotion_recognizer.set_window_size(params['window_size'])

                    if params['confidence_threshold'] != prev_params.get('confidence_threshold'):
                        detector.emotion_recognizer.set_confidence_threshold(params['confidence_threshold'])

                    if params['ambiguity_threshold'] != prev_params.get('ambiguity_threshold'):
                        detector.emotion_recognizer.set_ambiguity_threshold(params['ambiguity_threshold'])

                    # Hot-reload EAR анализатора
                    if EAR_HEADPOSE_AVAILABLE:
                        # Включение/выключение EAR
                        if params.get('enable_ear') != prev_params.get('enable_ear'):
                            if params.get('enable_ear'):
                                new_ear = EyeAspectRatioAnalyzer(
                                    ear_threshold=params.get('ear_threshold', 0.25),
                                    consec_frames=params.get('consec_frames', 1)
                                )
                                detector.set_ear_analyzer(new_ear)
                            else:
                                detector.set_ear_analyzer(None)
                        # Изменение параметров EAR (без сброса счётчиков)
                        elif params.get('enable_ear') and detector.ear_analyzer:
                            if params.get('ear_threshold') != prev_params.get('ear_threshold'):
                                detector.ear_analyzer.set_ear_threshold(params.get('ear_threshold', 0.25))
                            if params.get('consec_frames') != prev_params.get('consec_frames'):
                                detector.ear_analyzer.set_consec_frames(params.get('consec_frames', 1))

                        # Hot-reload HeadPose анализатора
                        if params.get('enable_head_pose') != prev_params.get('enable_head_pose'):
                            if params.get('enable_head_pose'):
                                detector.set_head_pose_estimator(HeadPoseEstimator())
                            else:
                                detector.set_head_pose_estimator(None)

                    st.session_state.prev_webcam_params = params.copy()

                # Используем функцию process_video_stream из бэкенда
                try:
                    fps_history = deque(maxlen=3)
                    for _ in range(3):
                        fps_history.append(0.0)

                    emotion_history = []
                    start_time = current_time()
                    vis = st.session_state.vis_params

                    for img, emotions in process_video_stream(cap, st.session_state.webcam_detector,
                                                              flip_h=st.session_state.backend_params['flip_h']):
                        if not st.session_state.get('webcam_running', False):
                            break

                        # Сохранение эмоции для статистики
                        if emotions:
                            for result in emotions:
                                emotion_history.append(result['emotion'])

                        # Ограничение истории
                        if len(emotion_history) > 100:
                            emotion_history = emotion_history[-100:]

                        # Расчет FPS
                        fps = 1 / (current_time() - start_time)
                        fps_history.append(fps)
                        avg_fps = round(sum(fps_history) / len(fps_history))

                        # FPS на изображение (только если включён)
                        if vis.get('show_fps', True):
                            cv2.putText(img, f'FPS: {avg_fps}', (5, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                        # Конвертация для отображения в Streamlit
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Отображение кадра
                        webcam_placeholder.image(img_rgb, channels="RGB", width='stretch')

                        # Отображение текущих эмоций и доп. данных
                        if emotions and vis.get('show_emotions', True):
                            emotion_text = "**Обнаруженные лица:**\n\n"
                            for i, result in enumerate(emotions):
                                emotion_text += f"**Лицо {i + 1}:**\n"
                                emotion_text += f"- Эмоция: {result['emotion']} ({result['confidence']:.2f})\n"

                                # EAR данные (только если включено отображение)
                                if result.get('ear') and vis.get('show_ear_info', True):
                                    ear_data = result['ear']
                                    emotion_text += f"- EAR: {ear_data['avg_ear']:.3f} "
                                    emotion_text += f"({'Открыты' if ear_data['eyes_open'] else 'Закрыты'}) "
                                    emotion_text += f"[Моргания: {ear_data['blink_count']}]\n"

                                # HeadPose данные (только если включено отображение)
                                if result.get('head_pose') and vis.get('show_hpe_info', True):
                                    hp = result['head_pose']
                                    emotion_text += f"- Поза: Pitch={hp['pitch']:.0f}° Yaw={hp['yaw']:.0f}° Roll={hp['roll']:.0f}°\n"
                                    if 'attention_state' in hp:
                                        emotion_text += f"- Состояние: {hp['attention_state']}\n"

                                emotion_text += "\n"

                            emotions_placeholder.markdown(emotion_text)
                        else:
                            emotions_placeholder.empty()

                        # Обновление статистики
                        if emotion_history:
                            stats = {}
                            for emotion in emotion_history:
                                stats[emotion] = stats.get(emotion, 0) + 1

                            stats_text = "**Статистика эмоций:**\n"
                            for emotion, count in stats.items():
                                percent = (count / len(emotion_history)) * 100
                                stats_text += f"{emotion}: {percent:.1f}%\n"

                            stats_placeholder.markdown(stats_text)

                        fps_placeholder.metric("Текущий FPS", avg_fps)
                        start_time = current_time()

                except CaptureReadError as e:
                    st.error(f"Ошибка веб-камеры: {e}")
                except Exception as e:
                    st.error(f"Ошибка обработки веб-камеры: {e}")
                finally:
                    cap.release()

                    # Cleanup только при явной остановке (webcam_running = False)
                    # При прерывании Streamlit из-за смены параметров webcam_running остаётся True
                    # детектор сохраняется для переиспользования на следующем перезапуске
                    if not st.session_state.get('webcam_running', True):
                        if st.session_state.webcam_detector:
                            if hasattr(st.session_state.webcam_detector.face_detector, 'close'):
                                st.session_state.webcam_detector.face_detector.close()
                            if hasattr(st.session_state.webcam_detector.emotion_recognizer, 'reset'):
                                st.session_state.webcam_detector.emotion_recognizer.reset()
                        st.session_state.webcam_detector = None
                        st.session_state.prev_webcam_params = None

                        # сообщение вместо empty() чтобы избежать ошибки MediaFileStorageError
                        webcam_placeholder.info("📷 Веб-камера остановлена")
                        emotions_placeholder.empty()
                        stats_placeholder.empty()
                        fps_placeholder.empty()

    with col2:
        # Статус и информация
        st.markdown("#### Статус")
        if st.session_state.get('webcam_running', False):
            st.success("✅ Веб-камера активна")
            st.info("Детекция лиц и эмоций в реальном времени")
        else:
            st.info("📷 Веб-камера готова")

        st.markdown("---")

        # Текущие параметры
        st.markdown("#### ⚙️ Текущие параметры")
        for key, value in st.session_state.backend_params.items():
            if key not in ['flip_h', 'show_preview']:
                st.metric(key.replace('_', ' ').title(), f"{value:.2f}" if isinstance(value, float) else value)

        st.markdown("---")

        # Инструкции
        st.markdown("#### 📝 Инструкции")
        st.markdown("""
        1. Запустите веб-камеру
        2. Посмотрите в камеру
        3. Эмоции будут распознаваться в реальном времени
        4. Настройте параметры на боковой панели
        5. Остановите когда закончите
        """)


# ============================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# ============================================

def main():
    """Основная функция приложения"""
    display_header()
    display_sidebar()

    # Проверяем доступность бэкенда

    # Основной контент в вкладках
    tab1, tab2, tab3 = st.tabs(["🎬 Загрузка и обработка", "📷 Веб-камера", "❓ Справка"])

    with tab1:
        # Проверяем статус обработки
        if st.session_state.processing_status in ["idle", "starting"]:
            create_upload_section()

        if st.session_state.processing_status == "processing":
            process_video()
        elif st.session_state.processing_status == "completed":
            display_result()
        elif st.session_state.processing_status == "failed":
            display_error()

    with tab2:
        create_webcam_section()

    with tab3:
        st.markdown("### ❓ Часто задаваемые вопросы")

        faqs = [
            {
                "question": "Как работает распознавание эмоций в реальном времени?",
                "answer": "Приложение использует класс FaceAnalysisPipeline, который объединяет детекцию лиц и распознавание эмоций. Оно обрабатывает каждый кадр видео в реальном времени, рисуя ограничивающие рамки и метки эмоций."
            },
            {
                "question": "Какие эмоции можно распознать?",
                "answer": "Система распознает базовые эмоции: радость, грусть, злость, удивление, страх, отвращение, нейтральность и возможно другие в зависимости от модели."
            },
            {
                "question": "Сохраняете ли вы мои видео или изображения?",
                "answer": "Нет. Вся обработка выполняется локально. Видео временно сохраняются только во время обработки и удаляются после."
            },
            {
                "question": "Могу ли я настроить параметры детекции?",
                "answer": "Да! Используйте боковую панель для настройки таких параметров, как уверенность детекции, размер окна для сглаживания и пороги уверенности."
            },
            {
                "question": "Что делать, если лица не обнаруживаются?",
                "answer": "Попробуйте настроить параметр 'Минимальная уверенность детекции' на боковой панели. Также убедитесь, что лица хорошо видны и освещены."
            },
            {
                "question": "Для чего отображается FPS?",
                "answer": "FPS (кадры в секунду) показывает скорость обработки. Более низкий FPS означает более медленную обработку, но может быть точнее."
            }
        ]

        for faq in faqs:
            with st.expander(f"**В:** {faq['question']}"):
                st.markdown(f"**О:** {faq['answer']}")

        st.markdown("---")

        st.markdown("### 🐛 Устранение неполадок")

        issues = [
            ("Веб-камера не работает",
             "Проверьте разрешения браузера для доступа к камере. Попробуйте обновить страницу."),
            ("Лица не обнаруживаются", "Настройте параметр уверенности детекции. Убедитесь в хорошем освещении."),
            ("Медленная работа", "Попробуйте уменьшить разрешение видео или частоту кадров."),
            ("Ошибки импорта",
             "Убедитесь, что файл face_detection.py находится в текущей директории."),
            ("Низкий FPS", "Модель может быть вычислительно затратной. Попробуйте использовать компьютер с GPU."),
        ]

        for issue, solution in issues:
            st.markdown(f"**{issue}:** {solution}")


# ============================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================

if __name__ == "__main__":
    try:
        main()

        # Очистка при завершении
        atexit.register(lambda: st.session_state.get('detection_processor', EmotionDetectionProcessor()).reset())

    except Exception as e:
        st.error(f"Ошибка приложения: {str(e)}")
        st.info("Пожалуйста, перезапустите приложение и попробуйте снова.")

        if st.button("🔄 Перезапустить приложение"):
            st.rerun()
