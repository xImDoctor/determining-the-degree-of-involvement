FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY ./frontend/pyproject.toml .
RUN pip install --no-cache-dir .

COPY ./frontend/src ./src
COPY ./backend/app/services/video_processing ./src/app/services/video_processing
COPY ./backend/app/core ./src/app/core

EXPOSE 8501

CMD ["streamlit", "run", "src/main.py"]
