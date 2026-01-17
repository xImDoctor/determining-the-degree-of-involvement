import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import time
import threading
import queue
import atexit
from pathlib import Path
import sys
import subprocess
from collections import deque
from time import time as current_time

sys.path.append('face_detection_and_emotion_recognition.py')


try:
    from face_detection_and_emotion_recognition import (
        FaceDetector,
        EmotionRecognizer,
        DetectFaceAndRecognizeEmotion,
        process_video_stream,
        CaptureReadError
    )

    BACKEND_AVAILABLE = True
except ImportError as e:
    st.error(f"Не удалось импортировать модуль бэкенда: {e}")
    BACKEND_AVAILABLE = False

APP_TITLE = "Real-time Emotion Detection"
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

st.markdown("""
<style>
    /* Основные стили */
    .main-header {
        font-size: 2.8rem;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        padding: 0.5rem;
    }

    .webcam-container {
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea0a 0%, #764ba20a 100%);
        margin: 2rem 0;
    }

    .processing-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }

    .result-card {
        background: linear-gradient(135deg, #56ab2f, #a8e063);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(86, 171, 47, 0.3);
    }

    .error-card {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(255, 65, 108, 0.3);
    }

    .video-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        background: #000;
    }

    .emotion-display {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 15px;
    }

    .emotion-item {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .emotion-confidence {
        font-size: 0.8rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# КЛАССЫ ДЛЯ ОБРАБОТКИ ВИДЕО
# ============================================

class EmotionDetectionProcessor:
    """Класс для обработки видео с использованием DetectFaceAndRecognizeEmotion"""

    def __init__(self):
        self.detector = None
        self.is_initialized = False
        self.current_emotions = []

    def initialize_models(self, params):
        """Инициализирует модели для распознавания эмоций"""
        try:
            if BACKEND_AVAILABLE:
                #детектор лиц
                face_detector = FaceDetector(
                    min_detection_confidence=params.get('min_detection_confidence', 0.5)
                )

                #=распознаватель эмоций
                emotion_recognizer = EmotionRecognizer(
                    window_size=params.get('window_size', 15),
                    confidence_threshold=params.get('confidence_threshold', 0.55),
                    ambiguity_threshold=params.get('ambiguity_threshold', 0.15)
                )

                #=основной детектор
                self.detector = DetectFaceAndRecognizeEmotion(face_detector, emotion_recognizer)

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

            # Обрабатка кадра с помощью детектора
            processed_frame, emotions = self.detector.detect_and_recognize(frame)

            self.current_emotions = emotions
            return processed_frame, emotions

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
                for emotion, confidence in emotions:
                    all_emotions.append(emotion)

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

        except Exception as e:
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
        'show_preview': False
    }

if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False


# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

def display_header():
    """Отображает заголовок приложения"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f'<h1 class="main-header">{APP_TITLE}</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">Upload a video or use webcam to detect faces and recognize emotions in real-time</p>',
            unsafe_allow_html=True)


def display_sidebar():
    """Отображает боковую панель с настройками"""
    with st.sidebar:
        st.markdown("### 🎭 Emotion Detection")

        # Информация о системе
        st.markdown("#### ℹ️ System Status")

        if BACKEND_AVAILABLE:
            st.success("✅ Backend module available")
        else:
            st.error("❌ Backend module not found")
            st.info("Please ensure face_detection_and_emotion_recognition.py is in the current directory")

        st.markdown("---")

        # Настройки параметров
        st.markdown("#### ⚙️ Processing Parameters")

        # Face Detector Parameters
        st.markdown("##### Face Detection")
        st.session_state.backend_params['min_detection_confidence'] = st.slider(
            "Min Detection Confidence",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.backend_params['min_detection_confidence'],
            step=0.01,
            help="Минимальная уверенность для детекции лиц"
        )

        # Emotion Recognizer Parameters
        st.markdown("##### Emotion Recognition")

        st.session_state.backend_params['window_size'] = st.slider(
            "Window Size",
            min_value=3,
            max_value=30,
            value=st.session_state.backend_params['window_size'],
            step=1,
            help="Размер окна для сглаживания"
        )

        st.session_state.backend_params['confidence_threshold'] = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.backend_params['confidence_threshold'],
            step=0.01,
            help="Минимальный порог уверенности для эмоции"
        )

        st.session_state.backend_params['ambiguity_threshold'] = st.slider(
            "Ambiguity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.backend_params['ambiguity_threshold'],
            step=0.01,
            help="Порог для амбивалентных эмоций"
        )

        # Общие параметры
        st.markdown("##### General Settings")
        st.session_state.backend_params['flip_h'] = st.checkbox(
            "Flip Horizontal",
            value=st.session_state.backend_params['flip_h'],
            help="Отзеркаливание по горизонтали"
        )

        st.session_state.backend_params['show_preview'] = st.checkbox(
            "Show Preview",
            value=st.session_state.backend_params['show_preview'],
            help="Показывать превью во время обработки"
        )

        st.markdown("---")

        # Кнопка сброса параметров
        if st.button("🔄 Reset to Default", use_container_width=True):
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
    st.markdown("### 📤 Upload Video")

    # Зона загрузки
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=SUPPORTED_FORMATS,
        help=f"Supported formats: {', '.join(SUPPORTED_FORMATS).upper()}",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        # Проверяем размер файла
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"File too large! Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB")
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
                if st.button("🚀 Start Emotion Detection", type="primary", use_container_width=True):
                    if not BACKEND_AVAILABLE:
                        st.error("Невозможно начать обработку: Backend модуль недоступен. Убедитесь, что файл face_detection_and_emotion_recognition.py находится в текущей директории со всеми необходимыми зависимостями.")
                    else:
                        st.session_state.processing_status = "starting"
                        st.rerun()
            with col2:
                if st.button("🗑️ Clear File", use_container_width=True):
                    st.session_state.uploaded_file_path = None
                    st.rerun()
        else:
            st.error(f"Error: {video_info['error']}")

    else:
        # Показываем подсказку
        st.info("Загрузите видеофайл, используя загрузчик выше.")
        st.markdown("""
        **Поддерживаемые форматы:** MP4, AVI, MOV, MKV, WebM, WMV
        **Максимальный размер:** 100MB
        """)


def display_file_info(uploaded_file, video_info):
    """Отображает информацию о файле"""
    st.markdown("### 📊 Video Information")

    # Нативные Streamlit metrics для информации о видеофайле
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Resolution",
            value=f"{video_info['width']}×{video_info['height']}"
        )

    with col2:
        duration = video_info["duration"]
        if duration >= 60:
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            duration_str = f"{minutes}:{seconds:02d}"
        else:
            duration_str = f"{duration:.1f}s"

        st.metric(
            label="Duration",
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
            label="Size (MB)",
            value=f"{file_size_mb:.1f}"
        )


def display_video_preview(video_path, video_info):
    """Отображает превью видео"""
    st.markdown("### 👀 Video Preview")

    # Основное видео
    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    video_bytes = open(video_path, "rb").read()
    st.video(video_bytes)
    st.markdown('</div>', unsafe_allow_html=True)

    # Примеры кадров
    st.markdown("#### 📸 Sample Frames")

    frames = st.session_state.video_processor.extract_sample_frames(video_path, 4)
    if frames:
        cols = st.columns(4)
        for idx, (col, frame) in enumerate(zip(cols, frames)):
            with col:
                img = Image.fromarray(frame)
                st.image(img, caption=f"Frame {idx + 1}", use_container_width=True)


def process_video():
    """Обрабатывает видео"""
    if st.session_state.processing_status == "starting" and st.session_state.uploaded_file_path:
        st.session_state.processing_status = "processing"

        # Показываем панель прогресса
        st.markdown('<div class="processing-card">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Processing Your Video")

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
                for emotion, confidence in emotions:
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
    st.markdown("### ✅ Processing Completed!")

    result_path = st.session_state.result_path

    if result_path and os.path.exists(result_path):
        # Показываем обработанное видео
        st.markdown("#### 🎬 Processed Video")
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(result_path)
        st.markdown('</div>', unsafe_allow_html=True)

        # статистика эмоций
        if st.session_state.emotion_statistics:
            st.markdown("#### 📊 Emotion Statistics")

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
            st.markdown("#### 📈 Overall Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Frames", st.session_state.emotion_statistics.get('total_frames', 0))
            with col2:
                st.metric("Face Detections", st.session_state.emotion_statistics.get('total_detections', 0))
            with col3:
                st.metric("Detection Rate", f"{st.session_state.emotion_statistics.get('detection_rate', 0):.1f}%")

        # Кнопка скачивания
        with open(result_path, "rb") as f:
            st.download_button(
                label="📥 Download Processed Video",
                data=f,
                file_name=os.path.basename(result_path),
                mime="video/mp4",
                type="primary",
                use_container_width=True
            )

    else:
        st.warning("Processed video file not found")

    # Кнопка для новой обработки
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Process Another Video", use_container_width=True):
            st.session_state.uploaded_file_path = None
            st.session_state.processing_status = "idle"
            st.session_state.result_path = None
            st.session_state.emotion_statistics = {}
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def display_error():
    """Отображает ошибку обработки"""
    st.markdown('<div class="error-card">', unsafe_allow_html=True)
    st.markdown("### ❌ Processing Failed")

    error_msg = getattr(st.session_state, 'error_message', 'Unknown error')
    st.error(f"Error: {error_msg}")

    # Советы по устранению неполадок
    st.markdown("#### 🔧 Troubleshooting Tips:")
    st.markdown("""
    1. ✅ Ensure `face_detection_and_emotion_recognition.py` is in the same directory
    2. ✅ Check if all dependencies are installed
    3. ✅ Try a shorter video (under 1 minute)
    4. ✅ Ensure the video format is supported
    5. ✅ Check available disk space
    """)

    if st.button("🔄 Try Again", use_container_width=True):
        st.session_state.processing_status = "idle"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def create_webcam_section():
    """Создает секцию работы с веб-камерой"""
    st.markdown("### 📷 Webcam Live Emotion Detection")

    if not BACKEND_AVAILABLE:
        st.warning(
            "Webcam emotion detection requires backend module. Please ensure face_detection_and_emotion_recognition.py is available.")
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        # Кнопки управления
        control_col1, control_col2 = st.columns(2)

        with control_col1:
            start_webcam = st.button("🎬 Start Webcam", type="primary", use_container_width=True)

        with control_col2:
            stop_webcam = st.button("⏹️ Stop Webcam", type="secondary", use_container_width=True)

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
                st.error("Cannot open webcam")
                st.session_state.webcam_running = False
            else:
                # Настройки камеры
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                # Используем функцию process_video_stream из бэкенда
                try:
                    fps_history = deque(maxlen=3)
                    for _ in range(3):
                        fps_history.append(0.0)

                    emotion_history = []
                    start_time = current_time()

                    for img, emotions in process_video_stream(cap, flip_h=st.session_state.backend_params['flip_h']):
                        if not st.session_state.get('webcam_running', False):
                            break

                        # Сохраняем эмоции для статистики
                        if emotions:
                            for emotion, confidence in emotions:
                                emotion_history.append(emotion)

                        # Ограничиваем историю
                        if len(emotion_history) > 100:
                            emotion_history = emotion_history[-100:]

                        # Расчет FPS
                        fps = 1 / (current_time() - start_time)
                        fps_history.append(fps)
                        avg_fps = round(sum(fps_history) / len(fps_history))

                        # Добавляем FPS на изображение
                        cv2.putText(img, f'FPS: {avg_fps}', (5, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                        # Конвертируем для отображения в Streamlit
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Отображаем кадр
                        webcam_placeholder.image(img_rgb, channels="RGB", use_container_width=True)

                        # Отображаем текущие эмоции
                        if emotions:
                            emotion_text = "**Detected Emotions:**\n"
                            for i, (emotion, confidence) in enumerate(emotions):
                                emotion_text += f"Face {i + 1}: {emotion} ({confidence:.2f})\n"
                            emotions_placeholder.markdown(emotion_text)

                        # Обновляем статистику
                        if emotion_history:
                            # Подсчет статистики
                            stats = {}
                            for emotion in emotion_history:
                                stats[emotion] = stats.get(emotion, 0) + 1

                            # Отображение статистики
                            stats_text = "**Recent Emotion Statistics:**\n"
                            for emotion, count in stats.items():
                                percent = (count / len(emotion_history)) * 100
                                stats_text += f"{emotion}: {percent:.1f}%\n"

                            stats_placeholder.markdown(stats_text)

                        fps_placeholder.metric("Current FPS", avg_fps)
                        start_time = current_time()

                except CaptureReadError as e:
                    st.error(f"Webcam error: {e}")
                except Exception as e:
                    st.error(f"Error in webcam processing: {e}")
                finally:
                    cap.release()
                    webcam_placeholder.empty()
                    emotions_placeholder.empty()
                    stats_placeholder.empty()
                    fps_placeholder.empty()

    with col2:
        # Статус и информация
        st.markdown("#### 🔴 Live Status")
        if st.session_state.get('webcam_running', False):
            st.success("✅ Webcam Active")
            st.info("Detecting faces and emotions in real-time")
        else:
            st.info("📷 Webcam Ready")

        st.markdown("---")

        # Текущие параметры
        st.markdown("#### ⚙️ Current Parameters")
        for key, value in st.session_state.backend_params.items():
            if key not in ['flip_h', 'show_preview']:
                st.metric(key.replace('_', ' ').title(), f"{value:.2f}" if isinstance(value, float) else value)

        st.markdown("---")

        # Инструкции
        st.markdown("#### 📝 Instructions")
        st.markdown("""
        1. Start webcam
        2. Look at the camera
        3. Emotions will be detected in real-time
        4. Adjust parameters in sidebar
        5. Stop when done
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
    tab1, tab2, tab3 = st.tabs(["🎬 Upload Video", "📷 Webcam Live", "❓ Help & Support"])

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
        st.markdown("### ❓ Frequently Asked Questions")

        faqs = [
            {
                "question": "How does real-time emotion detection work?",
                "answer": "The app uses DetectFaceAndRecognizeEmotion class which combines face detection and emotion recognition. It processes each video frame in real-time, drawing bounding boxes and emotion labels."
            },
            {
                "question": "What emotions can be detected?",
                "answer": "The system detects basic emotions: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral, and possibly others depending on the model."
            },
            {
                "question": "Do you store my videos or images?",
                "answer": "No. All processing is done locally. Videos are temporarily stored only during processing and deleted afterward."
            },
            {
                "question": "Can I adjust detection parameters?",
                "answer": "Yes! Use the sidebar to adjust parameters like detection confidence, window size for smoothing, and confidence thresholds."
            },
            {
                "question": "What if no faces are detected?",
                "answer": "Try adjusting the 'Min Detection Confidence' parameter in the sidebar. Also ensure faces are clearly visible and well-lit."
            },
            {
                "question": "Why is FPS displayed?",
                "answer": "FPS (Frames Per Second) shows the processing speed. Lower FPS means slower processing but might be more accurate."
            }
        ]

        for faq in faqs:
            with st.expander(f"**Q:** {faq['question']}"):
                st.markdown(f"**A:** {faq['answer']}")

        st.markdown("---")

        st.markdown("### 🐛 Troubleshooting")

        issues = [
            ("Webcam not working", "Check browser permissions for camera access. Try refreshing the page."),
            ("No faces detected", "Adjust detection confidence parameter. Ensure good lighting."),
            ("Slow performance", "Try reducing video resolution or frame rate."),
            ("Import errors", "Ensure face_detection_and_emotion_recognition.py is in the current directory."),
            ("Low FPS", "The model might be computationally intensive. Try on a machine with GPU."),
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
        st.error(f"Application error: {str(e)}")
        st.info("Please restart the application and try again.")

        if st.button("🔄 Restart Application"):
            st.rerun()
