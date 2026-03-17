import os
from collections import deque
from time import time as current_time

import cv2
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from api_client import EngagementAPIClient

APP_TITLE = "Распознавание эмоций в реальном времени"
APP_ICON = "🎭"

# ============================================
# CSS СТИЛИ
# ============================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_css():
    """Загрузка внешнего CSS файла"""
    from pathlib import Path

    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Встроенные стили для компактного отображения
        st.markdown(
            """
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
        """,
            unsafe_allow_html=True,
        )


load_css()


# ============================================
# ИНИЦИАЛИЗАЦИЯ СЕССИИ
# ============================================

BACKEND_WS_URL = os.getenv("BACKEND_WS_URL", "ws://localhost:8000")
BACKEND_HTTP_URL = os.getenv("BACKEND_HTTP_URL", "http://localhost:8000")

if "api_client" not in st.session_state:
    st.session_state.api_client = EngagementAPIClient(
        backend_ws_url=BACKEND_WS_URL,
        backend_http_url=BACKEND_HTTP_URL,
    )

if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

if "backend_healthy" not in st.session_state:
    st.session_state.backend_healthy = False

if "last_health_check" not in st.session_state:
    st.session_state.last_health_check = 0.0

# История для графиков
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=100)

if "head_pose_history" not in st.session_state:
    st.session_state.head_pose_history = {
        "pitch": deque(maxlen=100),
        "yaw": deque(maxlen=100),
        "roll": deque(maxlen=100),
    }

if "ear_history" not in st.session_state:
    st.session_state.ear_history = deque(maxlen=100)

if "timestamps" not in st.session_state:
    st.session_state.timestamps = deque(maxlen=100)

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

HEALTH_CHECK_INTERVAL = 10.0  # Интервал проверки доступности бэкенда (секунды)


def check_backend_health() -> bool:
    """Проверка доступности бэкенда с кэшированием результата"""
    now = current_time()
    if now - st.session_state.last_health_check < HEALTH_CHECK_INTERVAL:
        return st.session_state.backend_healthy

    api_client: EngagementAPIClient = st.session_state.api_client
    st.session_state.backend_healthy = api_client.check_health()
    st.session_state.last_health_check = now
    return st.session_state.backend_healthy


# ============================================
# ФУНКЦИИ ДЛЯ ГРАФИКОВ
# ============================================


def create_emotion_pie_chart(emotion_history):
    """Создание круговой диаграммы распределения эмоций"""
    if not emotion_history:
        return None

    counts = {}
    for emotion in emotion_history:
        counts[emotion] = counts.get(emotion, 0) + 1

    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(counts.keys()),
                values=list(counts.values()),
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3),
            )
        ]
    )
    fig.update_layout(
        title="Распределение эмоций",
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def create_head_pose_chart(timestamps, pitch_history, yaw_history, roll_history):
    """Создание графика положения головы"""
    if not timestamps or not pitch_history:
        return None

    t = list(timestamps)[-30:]
    pitch = list(pitch_history)[-30:]
    yaw = list(yaw_history)[-30:]
    roll = list(roll_history)[-30:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=pitch, mode="lines", name="Pitch", line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=t, y=yaw, mode="lines", name="Yaw", line=dict(color="green", width=2)))
    fig.add_trace(go.Scatter(x=t, y=roll, mode="lines", name="Roll", line=dict(color="blue", width=2)))

    fig.update_layout(
        title="Положение головы",
        xaxis_title="Время (с)",
        yaxis_title="Угол (градусы)",
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    return fig


def create_ear_chart(timestamps, ear_history):
    """Создание графика EAR (Eye Aspect Ratio)"""
    if not timestamps or not ear_history:
        return None

    t = list(timestamps)[-30:]
    ear = list(ear_history)[-30:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=t, y=ear, mode="lines", name="EAR", line=dict(color="purple", width=2)))

    fig.update_layout(
        title="Eye Aspect Ratio (EAR)",
        xaxis_title="Время (с)",
        yaxis_title="EAR",
        height=150,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )

    fig.add_hline(y=0.25, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_annotation(
        x=0.5, y=0.27, text="Порог закрытия", showarrow=False, xref="paper", yref="y", font=dict(size=10)
    )

    return fig


# ============================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# ============================================


def display_header():
    """Отображение заголовка приложения"""
    st.markdown(f'<h1 class="main-header">{APP_ICON} {APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown("---")


def create_webcam_section():
    """Создание секции работы с веб-камерой"""

    # Проверка доступности бэкенда (с кэшированием)
    backend_available = check_backend_health()

    if not backend_available:
        st.warning("Бэкенд недоступен. Убедитесь, что сервер запущен и доступен.")
        st.info(f"Адрес бэкенда: {BACKEND_HTTP_URL}")
        return

    api_client: EngagementAPIClient = st.session_state.api_client

    # Две колонки: левая для камеры (30%), правая для графиков (70%)
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
                if st.button("▶️ Запустить", width='stretch'):
                    st.session_state.webcam_running = True
                    st.rerun()

        with col2:
            if st.session_state.webcam_running:
                if st.button("⏹️ Стоп", width='stretch'):
                    st.session_state.webcam_running = False
                    st.rerun()

        # Текущая эмоция и метрики
        st.markdown("---")
        st.markdown("#### 📊 Текущие показатели")

        emotion_metric = st.empty()
        engagement_metric = st.empty()

        # Метрики положения головы в реальном времени
        metrics_container = st.container()
        with metrics_container:
            pitch_metric = st.empty()
            yaw_metric = st.empty()
            roll_metric = st.empty()

    with right_col:
        st.markdown("#### 📈 Аналитика в реальном времени")

        # Графики
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
            # Установка компактного разрешения для камеры
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_FPS, 15)

            # Подключение к бэкенду через WebSocket
            if not api_client.is_connected:
                try:
                    api_client.connect(room_id="engagement-app", name="engagement-user")
                except ConnectionError as e:
                    st.error(f"Не удалось подключиться к бэкенду: {e}")
                    cap.release()
                    st.session_state.webcam_running = False
                    return

            try:
                start_time = current_time()

                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    st.session_state.frame_count += 1
                    current_timestamp = current_time() - start_time
                    st.session_state.timestamps.append(current_timestamp)

                    # Отправка кадра на бэкенд и получение результатов
                    processed_frame, results = api_client.send_frame(frame)

                    # Отображение видео
                    if processed_frame is not None:
                        img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    else:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Изменение размера для отображения
                    height, width = img_rgb.shape[:2]
                    new_width = 320
                    new_height = int(height * new_width / width)
                    img_resized = cv2.resize(img_rgb, (new_width, new_height))

                    video_placeholder.image(img_resized, channels="RGB", width='stretch')

                    # Обновление метрик
                    if results:
                        result = results[0]  # Первое обнаруженное лицо

                        # Эмоция
                        emotion_metric.info(
                            f"**{result.get('emotion', '—')}** "
                            f"(уверенность: {result.get('confidence', 0):.2f})"
                        )
                        st.session_state.emotion_history.append(result.get("emotion", "unknown"))

                        # Вовлечённость
                        engagement = result.get("engagement")
                        if engagement:
                            level = engagement.get("level", "—")
                            score = engagement.get("score", 0)
                            trend = engagement.get("trend", "stable")
                            trend_icon = {"rising": "↑", "falling": "↓", "stable": "→"}.get(trend, "")
                            engagement_metric.success(
                                f"**Вовлечённость:** {level} ({score:.0%}) {trend_icon}"
                            )
                        else:
                            engagement_metric.empty()

                        # Положение головы
                        if result.get("head_pose"):
                            hp = result["head_pose"]
                            pitch_metric.metric("Pitch", f"{hp.get('pitch', 0):.1f}°")
                            yaw_metric.metric("Yaw", f"{hp.get('yaw', 0):.1f}°")
                            roll_metric.metric("Roll", f"{hp.get('roll', 0):.1f}°")

                            st.session_state.head_pose_history["pitch"].append(hp.get("pitch", 0))
                            st.session_state.head_pose_history["yaw"].append(hp.get("yaw", 0))
                            st.session_state.head_pose_history["roll"].append(hp.get("roll", 0))
                        else:
                            pitch_metric.metric("Pitch", "—")
                            yaw_metric.metric("Yaw", "—")
                            roll_metric.metric("Roll", "—")

                        # EAR
                        ear = result.get("ear")
                        if ear and ear.get("avg_ear") is not None:
                            st.session_state.ear_history.append(ear["avg_ear"])
                    else:
                        emotion_metric.warning("Лицо не обнаружено")
                        engagement_metric.empty()
                        pitch_metric.metric("Pitch", "—")
                        yaw_metric.metric("Yaw", "—")
                        roll_metric.metric("Roll", "—")

                    # Обновление графиков (каждый 5-й кадр)
                    if st.session_state.frame_count % 5 == 0:
                        # Круговая диаграмма эмоций
                        pie_fig = create_emotion_pie_chart(st.session_state.emotion_history)
                        if pie_fig:
                            pie_placeholder.plotly_chart(
                                pie_fig,
                                width='stretch',
                                key=f"pie_{st.session_state.frame_count}",
                            )

                        # График положения головы
                        pose_fig = create_head_pose_chart(
                            st.session_state.timestamps,
                            st.session_state.head_pose_history["pitch"],
                            st.session_state.head_pose_history["yaw"],
                            st.session_state.head_pose_history["roll"],
                        )
                        if pose_fig:
                            pose_placeholder.plotly_chart(
                                pose_fig,
                                width='stretch',
                                key=f"pose_{st.session_state.frame_count}",
                            )

                        # График EAR
                        ear_fig = create_ear_chart(
                            st.session_state.timestamps,
                            st.session_state.ear_history,
                        )
                        if ear_fig:
                            ear_placeholder.plotly_chart(
                                ear_fig,
                                width='stretch',
                                key=f"ear_{st.session_state.frame_count}",
                            )

            except Exception as e:
                st.error(f"Ошибка: {e}")
            finally:
                cap.release()
                # WebSocket НЕ отключается при rerun Streamlit — соединение
                # сохраняется в session_state для переиспользования.
                # Отключение только при явной остановке webcam.
                if not st.session_state.get("webcam_running", True):
                    api_client.disconnect()
                video_placeholder.empty()


def main():
    """Основная функция"""
    display_header()
    create_webcam_section()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Ошибка приложения: {str(e)}")
