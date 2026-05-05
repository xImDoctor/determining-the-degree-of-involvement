import csv
import io
import os
from collections import deque
from time import time as current_time

import cv2
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from api_client import EngagementAPIClient
from components.video_player import video_player

load_dotenv()

APP_TITLE = "Анализ вовлечённости при просмотре видео"
APP_ICON = "🎬"

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
# Порог красной линии на графике EAR (только визуализация)
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", "0.25"))

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

if "chart_update_count" not in st.session_state:
    st.session_state.chart_update_count = 0

# Временные метки видео и история вовлечённости
if "video_timestamps" not in st.session_state:
    st.session_state.video_timestamps = deque(maxlen=100)

if "engagement_history" not in st.session_state:
    st.session_state.engagement_history = deque(maxlen=100)

# Полная история для CSV-экспорта (сохраняется, пока живёт Streamlit-сессия)
if "export_data" not in st.session_state:
    st.session_state.export_data = []

# Камера хранится в session_state, чтобы rerun от плеера не переоткрывал её
if "camera" not in st.session_state:
    st.session_state.camera = None

# Снимок состояния плеера и wall-clock момент его получения
# для интерполяции video_timestamp между редкими rerun-ами от плеера.
if "player_snapshot" not in st.session_state:
    st.session_state.player_snapshot = None

if "player_snapshot_wall" not in st.session_state:
    st.session_state.player_snapshot_wall = 0.0


HEALTH_CHECK_INTERVAL = 10.0  # Интервал проверки доступности бэкенда (секунды)
CHART_UPDATE_INTERVAL = 15    # Интервал обновления графиков (в кадрах)


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

    fig.add_hline(y=EAR_THRESHOLD, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_annotation(
        x=0.5,
        y=EAR_THRESHOLD + 0.02,
        text="Порог закрытия",
        showarrow=False,
        xref="paper",
        yref="y",
        font=dict(size=10),
    )

    return fig


def create_engagement_timeline(video_timestamps, engagement_history):
    """Создание графика вовлечённости по временной шкале видео"""
    if not video_timestamps or not engagement_history:
        return None

    vt = list(video_timestamps)[-50:]
    eng = list(engagement_history)[-50:]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=vt,
            y=eng,
            mode="lines+markers",
            name="Вовлечённость",
            line=dict(color="#4CAF50", width=2),
            marker=dict(size=4),
        )
    )

    # Цветовые зоны уровней вовлечённости
    fig.add_hrect(y0=0.75, y1=1.0, fillcolor="green", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.50, y1=0.75, fillcolor="orange", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.25, y1=0.50, fillcolor="yellow", opacity=0.08, line_width=0)
    fig.add_hrect(y0=0.0, y1=0.25, fillcolor="red", opacity=0.08, line_width=0)

    fig.update_layout(
        title="Вовлечённость по времени видео",
        xaxis_title="Время видео (с)",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )

    return fig


# ============================================
# ОТРИСОВКА ГРАФИКОВ (in-place, без мерцания)
# ============================================


def render_charts(placeholders: dict, suffix: str) -> None:
    """
    Отрисовка всех графиков в переданные placeholder'ы.

    Args:
        placeholders: dict с ключами 'eng', 'pie', 'pose', 'ear'
        suffix: Суффикс для уникального key plotly_chart (избегает DuplicateWidgetID
            при многократных вызовах внутри одного script run)
    """
    eng_fig = create_engagement_timeline(
        st.session_state.video_timestamps,
        st.session_state.engagement_history,
    )
    if eng_fig:
        placeholders["eng"].plotly_chart(eng_fig, key=f"eng_{suffix}", width="stretch")

    pie_fig = create_emotion_pie_chart(st.session_state.emotion_history)
    if pie_fig:
        placeholders["pie"].plotly_chart(pie_fig, key=f"pie_{suffix}", width="stretch")

    pose_fig = create_head_pose_chart(
        st.session_state.timestamps,
        st.session_state.head_pose_history["pitch"],
        st.session_state.head_pose_history["yaw"],
        st.session_state.head_pose_history["roll"],
    )
    if pose_fig:
        placeholders["pose"].plotly_chart(pose_fig, key=f"pose_{suffix}", width="stretch")

    ear_fig = create_ear_chart(
        st.session_state.timestamps,
        st.session_state.ear_history,
    )
    if ear_fig:
        placeholders["ear"].plotly_chart(ear_fig, key=f"ear_{suffix}", width="stretch")


# ============================================
# CSV-ЭКСПОРТ
# ============================================


def export_csv():
    """Формирование CSV-файла с данными анализа"""
    if not st.session_state.export_data:
        return None

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "video_time_s", "emotion", "confidence", "engagement_score",
        "engagement_level", "ear", "pitch", "yaw", "roll",
    ])

    for row in st.session_state.export_data:
        writer.writerow([
            f"{row.get('video_time', 0):.2f}",
            row.get("emotion", ""),
            f"{row.get('confidence', 0):.3f}",
            f"{row.get('engagement_score', 0):.3f}",
            row.get("engagement_level", ""),
            f"{row.get('ear', 0):.3f}" if row.get("ear") is not None else "",
            f"{row.get('pitch', 0):.1f}" if row.get("pitch") is not None else "",
            f"{row.get('yaw', 0):.1f}" if row.get("yaw") is not None else "",
            f"{row.get('roll', 0):.1f}" if row.get("roll") is not None else "",
        ])

    return buf.getvalue()


# ============================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# ============================================


def display_header():
    """Отображение заголовка приложения"""
    st.markdown(f'<h1 class="main-header">{APP_ICON} {APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown("---")


def create_main_section():
    """Создание основной секции приложения"""

    # Проверка доступности бэкенда (с кэшированием)
    backend_available = check_backend_health()

    if not backend_available:
        st.warning("Бэкенд недоступен. Убедитесь, что сервер запущен и доступен.")
        st.info(f"Адрес бэкенда: {BACKEND_HTTP_URL}")
        return

    api_client: EngagementAPIClient = st.session_state.api_client

    # Ввод URL видео
    video_url = st.text_input(
        "URL видео (.mp4, .webm, .ogg)",
        placeholder="https://example.com/lecture.mp4",
        key="video_url_input",
    )

    # Верхняя строка: видеоплеер (2/3) + веб-камера (1/3)
    video_col, webcam_col = st.columns([2, 1])

    # Плейсхолдер подписи плеера (обновляется в while-цикле по wall-clock интерполяции)
    caption_placeholder = None
    with video_col:
        st.markdown("#### 🎬 Видео")
        if video_url:
            # Компонент с key="main_player" автоматически кладёт value в session_state
            player_state_initial = video_player(video_url, height=360, key="main_player")
            caption_placeholder = st.empty()
            if player_state_initial:
                duration = player_state_initial.get("duration", 0)
                current_t = player_state_initial.get("currentTime", 0)
                playing = player_state_initial.get("playing", False)
                status_text = "▶️ Воспроизведение" if playing else "⏸️ Пауза"
                caption_placeholder.caption(f"{status_text} — {current_t:.1f}с / {duration:.1f}с")
        else:
            st.info("Введите URL видеофайла для начала просмотра")

    with webcam_col:
        st.markdown("#### 📹 Веб-камера")

        # Контейнер для камеры
        camera_container = st.container()
        with camera_container:
            video_placeholder = st.empty()

        # Кнопки управления
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.webcam_running:
                if st.button("▶️ Запустить", width="stretch"):
                    st.session_state.webcam_running = True
                    st.session_state.needs_reset = True
                    st.rerun()

        with col2:
            if st.session_state.webcam_running:
                if st.button("⏹️ Стоп", width="stretch"):
                    st.session_state.webcam_running = False
                    st.rerun()

        # Текущие метрики
        st.markdown("---")
        st.markdown("#### 📊 Показатели")

        # Частные engagement по компонентам (в строку)
        comp_cols = st.columns(3)
        comp_eye_metric = comp_cols[0].empty()
        comp_hpe_metric = comp_cols[1].empty()
        comp_emo_metric = comp_cols[2].empty()

        engagement_metric = st.empty()
        emotion_metric = st.empty()

        # Метрики положения головы (в строку)
        pose_cols = st.columns(3)
        pitch_metric = pose_cols[0].empty()
        yaw_metric = pose_cols[1].empty()
        roll_metric = pose_cols[2].empty()

    # Разделитель перед графиками
    st.markdown("---")
    st.markdown("#### 📈 Аналитика в реальном времени")

    # Плейсхолдеры для графиков (обновляются in-place)
    eng_col, pie_col = st.columns(2)
    pose_col, ear_col = st.columns(2)
    chart_placeholders = {
        "eng": eng_col.empty(),
        "pie": pie_col.empty(),
        "pose": pose_col.empty(),
        "ear": ear_col.empty(),
    }

    # Отрисовка последних данных только когда камера выключена.
    # При активной камере графики обновляет while-цикл ниже — стартовый вызов
    # пропускается, чтобы избежать дублирования ключей с цикловыми вызовами.
    if not st.session_state.webcam_running:
        render_charts(chart_placeholders, suffix="last")

    # CSV-экспорт
    if st.session_state.export_data:
        csv_data = export_csv()
        if csv_data:
            st.download_button(
                "📥 Экспорт данных (CSV)",
                csv_data,
                "engagement_data.csv",
                "text/csv",
                width="stretch",
            )

    # Запуск веб-камеры
    if st.session_state.webcam_running:
        # Получение веб-камеры из сессии
        cap = st.session_state.camera
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                cap.set(cv2.CAP_PROP_FPS, 15)
                st.session_state.camera = cap

        if not cap.isOpened():
            st.error("Не удалось открыть веб-камеру")
            st.session_state.webcam_running = False
            st.session_state.camera = None
            return

        # Подключение к бэкенду через WebSocket
        if not api_client.is_connected:
            try:
                api_client.connect(room_id="video-engagement-app", name="video-engagement-user")
            except ConnectionError as e:
                st.error(f"Не удалось подключиться к бэкенду: {e}")
                cap.release()
                st.session_state.camera = None
                st.session_state.webcam_running = False
                return

        try:
            # Очистка истории графиков при новом запуске
            if st.session_state.get("needs_reset", False):
                st.session_state.emotion_history.clear()
                st.session_state.head_pose_history["pitch"].clear()
                st.session_state.head_pose_history["yaw"].clear()
                st.session_state.head_pose_history["roll"].clear()
                st.session_state.ear_history.clear()
                st.session_state.timestamps.clear()
                st.session_state.video_timestamps.clear()
                st.session_state.engagement_history.clear()
                st.session_state.export_data.clear()
                st.session_state.frame_count = 0
                st.session_state.chart_update_count = 0
                st.session_state.needs_reset = False

            start_time = current_time()

            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    break

                st.session_state.frame_count += 1
                current_timestamp = current_time() - start_time
                st.session_state.timestamps.append(current_timestamp)

                # Интерполяция video_timestamp по wall-clock.
                # Плеер отправляет snapshot только на play/pause/seeked/loadedmetadata,
                # чтобы не триггерить rerun Streamlit чаще необходимого (rerun рвёт камеру и WS).
                # Между снимками currentTime дорисовывается по прошедшему реальному времени.
                player_state = st.session_state.get("main_player")
                if player_state and player_state != st.session_state.player_snapshot:
                    # Новый снимок от плеера — зафиксировать момент получения
                    st.session_state.player_snapshot = player_state
                    st.session_state.player_snapshot_wall = current_time()

                video_ts = None
                snap = st.session_state.player_snapshot
                if snap and snap.get("currentTime") is not None:
                    base_ts = snap["currentTime"]
                    if snap.get("playing"):
                        video_ts = base_ts + (current_time() - st.session_state.player_snapshot_wall)
                    else:
                        video_ts = base_ts

                # Обновление подписи плеера интерполированным video_ts
                if caption_placeholder is not None and snap is not None and video_ts is not None:
                    duration = snap.get("duration", 0) or 0
                    status_text = "▶️ Воспроизведение" if snap.get("playing") else "⏸️ Пауза"
                    caption_placeholder.caption(
                        f"{status_text} — {video_ts:.1f}с / {duration:.1f}с"
                    )

                # Отправка кадра на бэкенд с временной меткой видео
                processed_frame, results, echoed_ts = api_client.send_frame(
                    frame, video_timestamp=video_ts
                )

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

                video_placeholder.image(img_resized, channels="RGB", width="stretch")

                # Обновление метрик
                if results:
                    result = results[0]  # Первое обнаруженное лицо

                    # Эмоция
                    emotion = result.get("emotion", "unknown")
                    confidence = result.get("confidence", 0)
                    emotion_metric.info(
                        f"**{emotion}** "
                        f"(уверенность: {confidence:.2f})"
                    )
                    st.session_state.emotion_history.append(emotion)

                    # Вовлечённость
                    engagement = result.get("engagement")
                    ear = result.get("ear")
                    hp = result.get("head_pose")
                    engagement_score = 0
                    engagement_level = ""
                    if engagement:
                        components = engagement.get("components") or {}

                        # Eye (EAR) компонент
                        if ear and ear.get("attention_state"):
                            eye_s = components.get("eye_score", 0)
                            comp_eye_metric.info(
                                f"**Eye:** {ear['attention_state']} ({eye_s:.2f})"
                            )
                        else:
                            comp_eye_metric.empty()

                        # HPE компонент
                        if hp and hp.get("attention_state"):
                            hp_s = components.get("head_pose_score", 0)
                            comp_hpe_metric.info(
                                f"**HPE:** {hp['attention_state']} ({hp_s:.2f})"
                            )
                        else:
                            comp_hpe_metric.empty()

                        # Emotion компонент
                        emo_s = components.get("emotion_score", 0)
                        if emo_s:
                            comp_emo_metric.info(f"**Emo:** {emo_s:.2f}")
                        else:
                            comp_emo_metric.empty()

                        # Общий engagement
                        level = engagement.get("level", "—")
                        score = engagement.get("score", 0)
                        trend = engagement.get("trend", "stable")
                        trend_icon = {"rising": "↑", "falling": "↓", "stable": "→"}.get(trend, "")
                        engagement_metric.success(
                            f"**Вовлечённость:** {level} ({score:.0%}) {trend_icon}"
                        )
                        engagement_score = score
                        engagement_level = level

                        # Сохранение для графика вовлечённости по видео
                        if echoed_ts is not None:
                            st.session_state.video_timestamps.append(echoed_ts)
                            st.session_state.engagement_history.append(score)
                    else:
                        comp_eye_metric.empty()
                        comp_hpe_metric.empty()
                        comp_emo_metric.empty()
                        engagement_metric.empty()

                    # Положение головы
                    pitch_val = yaw_val = roll_val = None
                    if result.get("head_pose"):
                        hp = result["head_pose"]
                        pitch_val = hp.get("pitch", 0)
                        yaw_val = hp.get("yaw", 0)
                        roll_val = hp.get("roll", 0)
                        pitch_metric.metric("Pitch", f"{pitch_val:.1f}°")
                        yaw_metric.metric("Yaw", f"{yaw_val:.1f}°")
                        roll_metric.metric("Roll", f"{roll_val:.1f}°")

                        st.session_state.head_pose_history["pitch"].append(pitch_val)
                        st.session_state.head_pose_history["yaw"].append(yaw_val)
                        st.session_state.head_pose_history["roll"].append(roll_val)
                    else:
                        pitch_metric.metric("Pitch", "—")
                        yaw_metric.metric("Yaw", "—")
                        roll_metric.metric("Roll", "—")

                    # EAR
                    ear_val = None
                    ear = result.get("ear")
                    if ear and ear.get("avg_ear") is not None:
                        ear_val = ear["avg_ear"]
                        st.session_state.ear_history.append(ear_val)

                    # Сохранение строки для CSV-экспорта
                    st.session_state.export_data.append({
                        "video_time": echoed_ts if echoed_ts is not None else current_timestamp,
                        "emotion": emotion,
                        "confidence": confidence,
                        "engagement_score": engagement_score,
                        "engagement_level": engagement_level,
                        "ear": ear_val,
                        "pitch": pitch_val,
                        "yaw": yaw_val,
                        "roll": roll_val,
                    })
                else:
                    comp_eye_metric.empty()
                    comp_hpe_metric.empty()
                    comp_emo_metric.empty()
                    emotion_metric.warning("Лицо не обнаружено")
                    engagement_metric.empty()
                    pitch_metric.metric("Pitch", "—")
                    yaw_metric.metric("Yaw", "—")
                    roll_metric.metric("Roll", "—")

                # Обновление графиков с фиксированной частотой (in-place)
                if st.session_state.frame_count % CHART_UPDATE_INTERVAL == 0:
                    n = st.session_state.chart_update_count
                    st.session_state.chart_update_count = n + 1
                    render_charts(chart_placeholders, suffix=str(n))

        except Exception as e:
            st.error(f"Ошибка: {e}")
        finally:
            # Камера и WebSocket сохраняются в session_state для переиспользования
            # между rerun-ами (которые триггерит компонент плеера).
            # Освобождение ресурсов только при явной остановке webcam.
            if not st.session_state.get("webcam_running", True):
                if st.session_state.camera is not None:
                    st.session_state.camera.release()
                    st.session_state.camera = None
                api_client.disconnect()
                video_placeholder.empty()


def main():
    """Основная функция"""
    display_header()
    create_main_section()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Ошибка приложения: {str(e)}")
