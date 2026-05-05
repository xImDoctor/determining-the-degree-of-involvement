# Frontend — Engagement Detection

Streamlit application for real-time student engagement analysis (while watching educational video content), frontend of the analysis system.

## Structure

```
frontend/
├── engagement_app.py           # Base application (webcam -> analysis)
├── video_engagement_app.py     # Extended application (video + webcam -> analysis with video sync)
├── api_client.py               # WebSocket client for FastAPI backend
├── styles.css                  # UI styles
├── components/
│   └── video_player/           # Custom Streamlit video player component
│       ├── __init__.py         # Python wrapper of the player
│       └── index.html          # HTML5 player + JS (currentTime feedback)
├── tools/
│   └── param_testing_app.py    # Standalone monolith for system parameter experimentation
├── pyproject.toml              # Project configuration and dependencies
├── requirements.txt            # Dependencies for Docker
└── Dockerfile
```

## Applications

### `engagement_app.py` — Base Analysis

Webcam captures the student's face, frames are sent to the backend via WebSocket, and results (emotions, engagement, head pose, EAR) are displayed in real time.

```bash
cd frontend
streamlit run engagement_app.py
```

### `video_engagement_app.py` — Video Viewing Analysis

The student watches an educational video (by URL) while the system simultaneously analyzes engagement via webcam. Metrics are synchronized with the video timeline.

```bash
cd frontend
streamlit run video_engagement_app.py
```

Features:
- Video file URL input (.mp4, .webm, .ogg)
- Custom video player with current position feedback
- Engagement chart along the video timeline
- CSV data export


## How to start

### Option 1: Local (requires running backend)

```bash
cd frontend
pip install -r requirements.txt
streamlit run engagement_app.py
# or
streamlit run video_engagement_app.py
```

The backend must be available at `ws://localhost:8000`. If the backend is at a different address, configure via `.env` (recommended — see [Environment Variables](#environment-variables)) or export the variables inline:

```bash
BACKEND_WS_URL=ws://192.168.1.10:8000 \
BACKEND_HTTP_URL=http://192.168.1.10:8000 \
streamlit run video_engagement_app.py
```

### Option 2: Parameter Testing Tool (standalone)

`param_testing_app.py` works as a monolith — it directly imports the backend ML pipeline without WebSocket. Used for experimenting with thresholds and parameters.

```bash
cd frontend
pip install -r tools/requirements.txt
streamlit run tools/param_testing_app.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BACKEND_WS_URL` | `ws://localhost:8000` | Backend WebSocket URL |
| `BACKEND_HTTP_URL` | `http://localhost:8000` | Backend HTTP URL (health check) |

Variables are loaded via `python-dotenv` from a `.env` file in the `frontend/` directory. Copy `.env.example` to `.env` and edit values as needed:

```bash
cd frontend
cp .env.example .env
# edit .env
```

System environment variables take precedence over `.env` entries.
