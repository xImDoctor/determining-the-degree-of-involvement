import atexit
import os
import sys
from collections import deque
from pathlib import Path
from time import time as current_time
import random

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.append('../backend')

try:
    from app.services.video_processing import (
        FaceDetector,
        EmotionRecognizer,
        FaceAnalysisPipeline
    )
    from app.services.video_processing import CaptureReadError

    BACKEND_AVAILABLE = True
except ImportError as e:
    st.error(f"Не удалось импортировать модуль бэкенда: {e}")
    BACKEND_AVAILABLE = False

# Импорт модулей EAR и HeadPose
EAR_HEADPOSE_AVAILABLE = False
try:
    from app.services.video_processing import EyeAspectRatioAnalyzer
    from app.services.video_processing import HeadPoseEstimator

    EAR_HEADPOSE_AVAILABLE = True
except ImportError:
    pass

APP_TITLE = "Распознавание эмоций в реальном времени"
APP_ICON = "🎭"

# ============================================
# CSS СТИЛИ
# ============================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)


def load_css():
    """Загружает внешний CSS файл"""
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Встроенные стили для компактного отображения
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            color: #4CAF50;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        .camera-container {
            border: 3px solid #4CAF50;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            margin-bottom: 15px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 15px;
            color: white;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .emotion-badge {
            display: inline-block;
            background: #4CAF50;
            color: white;
            border-radius: 20px;
            padding: 5px 15px;
            margin: 3px;
            font-size: 14px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)


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
        # Фиксированные параметры
        self.params = {
            'min_detection_confidence': 0.5,
            'window_size': 15,
            'confidence_threshold': 0.55,
            'ambiguity_threshold': 0.15,
            'enable_ear': True,
            'enable_head_pose': True,
            'ear_threshold': 0.25,
            'consec_frames': 1,
            'flip_h': False
        }

    def initialize_models(self):
        """Инициализирует модели для распознавания эмоций"""
        try:
            if BACKEND_AVAILABLE:
                face_detector = FaceDetector(min_detection_confidence=self.params['min_detection_confidence'])
                emotion_recognizer = EmotionRecognizer(
                    window_size=self.params['window_size'],
                    confidence_threshold=self.params['confidence_threshold'],
                    ambiguity_threshold=self.params['ambiguity_threshold']
                )

                ear_analyzer = None
                if EAR_HEADPOSE_AVAILABLE and self.params.get('enable_ear', False):
                    ear_analyzer = EyeAspectRatioAnalyzer(
                        ear_threshold=self.params.get('ear_threshold', 0.25),
                        consec_frames=self.params.get('consec_frames', 3)
                    )

                head_pose_estimator = None
                if EAR_HEADPOSE_AVAILABLE and self.params.get('enable_head_pose', False):
                    head_pose_estimator = HeadPoseEstimator()

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

    def process_frame(self, frame):
        """Обрабатывает один кадр видео"""
        if not self.is_initialized or self.detector is None:
            return frame, []

        try:
            if self.params['flip_h']:
                frame = cv2.flip(frame, 1)

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

    def reset(self):
        """Сбрасывает состояние процессора"""
        self.current_emotions = []
        if hasattr(self.detector, 'face_detector'):
            if hasattr(self.detector.face_detector, 'close'):
                self.detector.face_detector.close()


# ============================================
# ИНИЦИАЛИЗАЦИЯ СЕССИИ
# ============================================

if 'detection_processor' not in st.session_state:
    st.session_state.detection_processor = EmotionDetectionProcessor()
    success, message = st.session_state.detection_processor.initialize_models()
    if not success:
        st.error(message)

if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False

# История для графиков
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=100)

if 'head_pose_history' not in st.session_state:
    st.session_state.head_pose_history = {
        'pitch': deque(maxlen=100),
        'yaw': deque(maxlen=100),
        'roll': deque(maxlen=100)
    }

if 'ear_history' not in st.session_state:
    st.session_state.ear_history = deque(maxlen=100)

if 'timestamps' not in st.session_state:
    st.session_state.timestamps = deque(maxlen=100)

if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0


# ============================================
# ФУНКЦИИ ДЛЯ ГРАФИКОВ
# ============================================

def create_emotion_pie_chart(emotion_history):
    """Создает круговую диаграмму распределения эмоций"""
    if not emotion_history:
        emotions = ['happy', 'sad', 'angry', 'surprise', 'neutral']
        weights = [0.3, 0.2, 0.1, 0.15, 0.25]
        counts = {e: int(100 * w) for e, w in zip(emotions, weights)}
    else:
        counts = {}
        for emotion in emotion_history:
            counts[emotion] = counts.get(emotion, 0) + 1

    fig = go.Figure(data=[go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    fig.update_layout(
        title="Распределение эмоций",
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    return fig


def create_head_pose_chart(timestamps, pitch_history, yaw_history, roll_history):
    """Создает график положения головы"""
    if not timestamps or not pitch_history:
        t = list(range(30))
        pitch = [10 * np.sin(i * 0.2) + random.uniform(-2, 2) for i in t]
        yaw = [15 * np.cos(i * 0.2) + random.uniform(-2, 2) for i in t]
        roll = [5 * np.sin(i * 0.3) + random.uniform(-1, 1) for i in t]
        t = [i * 0.1 for i in t]
    else:
        t = list(timestamps)[-30:]
        pitch = list(pitch_history)[-30:]
        yaw = list(yaw_history)[-30:]
        roll = list(roll_history)[-30:]

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t, y=pitch,
        mode='lines',
        name='Pitch',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=t, y=yaw,
        mode='lines',
        name='Yaw',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=t, y=roll,
        mode='lines',
        name='Roll',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title="Положение головы",
        xaxis_title="Время (с)",
        yaxis_title="Угол (градусы)",
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def create_ear_chart(timestamps, ear_history):
    """Создает график EAR (Eye Aspect Ratio)"""
    if not timestamps or not ear_history:
        t = list(range(30))
        ear = [0.3 + 0.05 * np.sin(i * 0.3) + random.uniform(-0.02, 0.02) for i in t]
        t = [i * 0.1 for i in t]
    else:
        t = list(timestamps)[-30:]
        ear = list(ear_history)[-30:]

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t, y=ear,
        mode='lines',
        name='EAR',
        line=dict(color='purple', width=2)
    ))

    fig.update_layout(
        title="Eye Aspect Ratio (EAR)",
        xaxis_title="Время (с)",
        yaxis_title="EAR",
        height=150,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False
    )
    
    fig.add_hline(y=0.25, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_annotation(x=0.5, y=0.27, text="Порог закрытия", showarrow=False, 
                      xref="paper", yref="y", font=dict(size=10))
    
    return fig


# ============================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# ============================================

def display_header():
    """Отображает заголовок приложения"""
    st.markdown(f'<h1 class="main-header">{APP_ICON} {APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown("---")


def create_webcam_section():
    """Создает секцию работы с веб-камерой"""
    
    if not BACKEND_AVAILABLE:
        st.warning("Модуль бэкенда не доступен")
        return

    # Создаем две колонки: левая для камеры (30%), правая для графиков (70%)
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("#### 📹 Видеопоток")
        
        # Контейнер для камеры с фиксированным размером
        camera_container = st.container()
        with camera_container:
            video_placeholder = st.empty()
        
        # Кнопки управления
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.webcam_running:
                if st.button("▶️ Запустить", use_container_width=True):
                    st.session_state.webcam_running = True
                    st.rerun()
        
        with col2:
            if st.session_state.webcam_running:
                if st.button("⏹️ Стоп", use_container_width=True):
                    st.session_state.webcam_running = False
                    st.rerun()
        
        # Текущая эмоция и метрики положения головы
        st.markdown("---")
        st.markdown("#### 📊 Текущие показатели")
        
        emotion_metric = st.empty()
        
        # Метрики положения головы в реальном времени
        metrics_container = st.container()
        with metrics_container:
            pitch_metric = st.empty()
            yaw_metric = st.empty()
            roll_metric = st.empty()

    with right_col:
        st.markdown("#### 📈 Аналитика в реальном времени")
        
        # Три графика друг под другом
        chart_container = st.container()
        with chart_container:
            pie_placeholder = st.empty()
            pose_placeholder = st.empty()
            ear_placeholder = st.empty()

    # Запуск веб-камеры
    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Не удалось открыть веб-камеру")
            st.session_state.webcam_running = False
        else:
            # Устанавливаем компактное разрешение для камеры
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_FPS, 15)

            processor = st.session_state.detection_processor

            try:
                start_time = current_time()
                
                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    st.session_state.frame_count += 1
                    current_timestamp = current_time() - start_time
                    st.session_state.timestamps.append(current_timestamp)

                    # Обработка кадра
                    processed_frame, emotions = processor.process_frame(frame)

                    # Отображение видео в компактном размере
                    img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Изменяем размер для отображения
                    height, width = img_rgb.shape[:2]
                    new_width = 320
                    new_height = int(height * new_width / width)
                    img_resized = cv2.resize(img_rgb, (new_width, new_height))
                    
                    video_placeholder.image(img_resized, channels="RGB", use_container_width=True)

                    # Обновление метрик
                    if emotions:
                        result = emotions[0]  # Берем первое лицо
                        
                        # Эмоция
                        emotion_metric.info(f"**{result['emotion']}** (уверенность: {result['confidence']:.2f})")
                        st.session_state.emotion_history.append(result['emotion'])
                        
                        # Положение головы
                        if result.get('head_pose'):
                            hp = result['head_pose']
                            pitch_metric.metric("Pitch", f"{hp['pitch']:.1f}°", delta=None)
                            yaw_metric.metric("Yaw", f"{hp['yaw']:.1f}°", delta=None)
                            roll_metric.metric("Roll", f"{hp['roll']:.1f}°", delta=None)
                            
                            st.session_state.head_pose_history['pitch'].append(hp['pitch'])
                            st.session_state.head_pose_history['yaw'].append(hp['yaw'])
                            st.session_state.head_pose_history['roll'].append(hp['roll'])
                        else:
                            pitch_metric.metric("Pitch", "—")
                            yaw_metric.metric("Yaw", "—")
                            roll_metric.metric("Roll", "—")
                        
                        # EAR
                        if result.get('ear'):
                            st.session_state.ear_history.append(result['ear']['avg_ear'])
                        else:
                            # Эмуляция EAR для демонстрации
                            st.session_state.ear_history.append(0.25 + random.uniform(-0.05, 0.05))
                    else:
                        emotion_metric.warning("Лицо не обнаружено")
                        pitch_metric.metric("Pitch", "—")
                        yaw_metric.metric("Yaw", "—")
                        roll_metric.metric("Roll", "—")

                    # Обновление графиков (каждый 5-й кадр)
                    if st.session_state.frame_count % 5 == 0:
                        # Круговая диаграмма эмоций
                        pie_fig = create_emotion_pie_chart(st.session_state.emotion_history)
                        pie_placeholder.plotly_chart(
                            pie_fig,
                            use_container_width=True,
                            key=f"pie_{st.session_state.frame_count}"
                        )
                        
                        # График положения головы
                        pose_fig = create_head_pose_chart(
                            st.session_state.timestamps,
                            st.session_state.head_pose_history['pitch'],
                            st.session_state.head_pose_history['yaw'],
                            st.session_state.head_pose_history['roll']
                        )
                        pose_placeholder.plotly_chart(
                            pose_fig,
                            use_container_width=True,
                            key=f"pose_{st.session_state.frame_count}"
                        )
                        
                        # График EAR
                        ear_fig = create_ear_chart(
                            st.session_state.timestamps,
                            st.session_state.ear_history
                        )
                        ear_placeholder.plotly_chart(
                            ear_fig,
                            use_container_width=True,
                            key=f"ear_{st.session_state.frame_count}"
                        )

            except Exception as e:
                st.error(f"Ошибка: {e}")
            finally:
                cap.release()
                video_placeholder.empty()


def main():
    """Основная функция"""
    display_header()
    create_webcam_section()


if __name__ == "__main__":
    try:
        main()
        atexit.register(lambda: st.session_state.get('detection_processor', EmotionDetectionProcessor()).reset())
    except Exception as e:
        st.error(f"Ошибка приложения: {str(e)}")
