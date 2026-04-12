# Real-time Engagement Detection Backend

Real-time engagement detection system with FastAPI backend and WebSocket support.

## Features

| Feature                 | Description                                  |
|-------------------------|----------------------------------------------|
| **Face Detection**      | MediaPipe Face Detection                     |
| **Emotion Recognition** | EmotiEffLib (PyTorch)                        |
| **EAR Analysis**        | Eye Aspect Ratio — blink & fatigue detection |
| **Head Pose**           | Gaze direction estimation                    |
| **WebSocket Streaming** | Real-time video processing                   |
| **Room Management**     | Isolated sessions for multiple clients       |

## Requirements

- Python 3.12+
- CUDA (optional, for PyTorch acceleration)

## Installation

```bash
pip install -r requirements.txt
```

## Running

### Local

```bash
# Start Redis server first (required)
redis-server

# Run the backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build image
docker build -t engagement-detection-backend .

# Run with GPU
docker run -d -p 8000:8000 --gpus all engagement-detection-backend

# Run without GPU
docker run -d -p 8000:8000 engagement-detection-backend
```

See root `docker-compose.yaml` for full stack deployment.

---

## API Endpoints

### REST

| Method | Endpoint                   | Description       |
|--------|----------------------------|-------------------|
| `GET`  | `/health`                  | Health check      |
| `GET`  | `/rooms`                   | List active rooms |
| `GET`  | `/rooms/{room_id}/clients` | Clients in room   |

### WebSocket

| Endpoint                                                | Description              |
|---------------------------------------------------------|--------------------------|
| `/ws/rooms/{room_id}/stream`                            | Send frames for analysis |
| `/ws/rooms/{room_id}/clients/{client_id}/output_stream` | Receive processed stream |

---

## Configuration

Environment variables (or `.env` file):

| Parameter                       | Description              | Default                           |
|---------------------------------|--------------------------|-----------------------------------|
| `app_version`                   | App version              | `1.0.0`                           |
| `cors_allowed_origins`          | CORS origins             | `localhost:8501, localhost:63342` |
| `face_detection_min_confidence` | Face detection threshold | `0.5`                             |
| `emotion_model_name`            | Emotion model            | `enet_b2_8`                       |
| `emotion_device`                | Device (cpu/cuda/auto)   | `auto`                            |
| `ear_threshold`                 | EAR threshold            | `0.25`                            |
| `head_pitch_attentive`          | Head pitch threshold     | `20.0`                            |

---

## Architecture

```
backend/
├── app/
│   ├── api/
│   │   ├── stream.py     # WebSocket endpoints
│   │   └── room.py      # REST room endpoints
│   ├── core/
│   │   └── config.py    # Configuration
│   ├── db/              # Database
│   ├── schemas/         # Pydantic models
│   └── services/
│       ├── room.py              # Room management
│       └── video_processing/   # Video processing
│           ├── face_analysis_pipeline.py
│           ├── face_detection.py
│           ├── analyze_emotion.py
│           ├── analyze_ear.py
│           └── analyze_head_pose.py
└── tests/
```

---

## Testing

```bash
pytest
```

## GitHub Actions CI/CD

The project includes GitHub Actions workflows for continuous integration and deployment:

### Workflows

1. **CI/CD Pipeline** (`.github/workflows/ci-cd.yml`):
   - Runs tests, linting, type checking on push to main and pull requests
   - Builds and pushes Docker images to GitHub Container Registry (GHCR) on push to main or version tags (v*)
   - Docker images are only built/pushed after successful test completion

### Local Development with Docker Compose

To use the pre-built images from GHCR:

```bash
# Set your GitHub username (lowercase) as owner
export GHCR_REPOSITORY_OWNER=your_github_username  # or edit .env

# Start the stack
docker compose up
```

For local development without GHCR, you can override the image fields in docker-compose.yaml to build from source.

### Manual Docker Build and Push

```bash
# Login to GHCR
echo $GHCR_TOKEN | docker login ghcr.io -u $GHCR_USERNAME --password-stdin

# Build and push backend
docker build -t ghcr.io/your_github_username/engagement-detection-backend:latest backend
docker push ghcr.io/your_github_username/engagement-detection-backend:latest
```
