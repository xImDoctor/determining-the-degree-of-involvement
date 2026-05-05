# Переменные окружения и `.env` для фронтенда

`engagement_app.py` и `video_engagement_app.py` загружают переменные через `python-dotenv` без явного пути:

```python
# frontend/engagement_app.py:14
load_dotenv()
```

`load_dotenv()` ищет `.env` в текущей директории запуска и поднимается вверх по дереву каталогов до первого совпадения. При запуске `cd frontend && streamlit run engagement_app.py` это будет `frontend/.env`. Системные переменные окружения имеют приоритет над `.env`.

`tools/param_testing_app.py` использует переменные через `os.getenv` напрямую, без `load_dotenv`, – для этого приложения env-переменные нужно либо экспортировать в shell, либо запускать через `dotenv` CLI.

---

## Frontend-специфичные переменные

| Переменная | Default | Описание |
|------------|---------|----------|
| `BACKEND_WS_URL` | `ws://localhost:8000` | WebSocket-адрес backend для `/ws/rooms/{room_id}/stream` |
| `BACKEND_HTTP_URL` | `http://localhost:8000` | HTTP-адрес backend для `GET /health`, `GET /rooms` |
| `EAR_THRESHOLD` | `0.25` | **Только** позиция горизонтальной красной линии на графике EAR в `engagement_app.py`. **Не** влияет на реальный порог детекции моргания в backend (его задаёт одноимённая переменная backend) |

---

## Пример минимального `.env` для запуска фронтенда

```bash
# Frontend → Backend
BACKEND_WS_URL=ws://localhost:8000
BACKEND_HTTP_URL=http://localhost:8000

# Только для отображения графика EAR
EAR_THRESHOLD=0.25
```

---

## Backend-переменные, влияющие на восприятие фронтендом

Эти переменные задаются для backend, но определяют, что фронтенд получит в WS-ответе. Подробное описание см. в [../backend/environment-variables.md](../backend/environment-variables.md).

| Переменная | Что меняется в UI |
|------------|-------------------|
| `EMOTION_WINDOW_SIZE` | Скорость отклика индикатора эмоции (большие значения = плавнее, медленнее) |
| `EMOTION_CONFIDENCE_THRESHOLD` | Частота fallback на `Neutral` при низкой уверенности |
| `EAR_THRESHOLD` (backend) | Реальный порог моргания, влияет на blink_count |
| `EAR_ALERT_THRESHOLD` / `EAR_DROWSY_THRESHOLD` / `EAR_VERY_DROWSY_THRESHOLD` | Какой `attention_state` придёт от backend |
| `HEAD_PITCH_*` / `HEAD_YAW_*` | Какой `attention_state` HPE придёт от backend |

> `EAR_THRESHOLD` присутствует и для backend, и для frontend – **они независимы**. Если хотите, чтобы линия на графике совпадала с реальным порогом, держите эти значения синхронными в `.env`.

---

## Совместный запуск через Docker Compose

Compose-файл проекта (см. [../backend/deployment.md](../backend/deployment.md)) поднимает Redis и backend в одной сети. Фронтенд запускается **локально**, не в контейнере, и подключается по `localhost`:

```bash
docker compose up -d         # redis + backend
cd frontend
streamlit run engagement_app.py
```

В этой схеме фронтенду достаточно default-значений `BACKEND_WS_URL` и `BACKEND_HTTP_URL`.

Если фронтенд запускается из другого контейнера или удалённой машины, тогда задайте `BACKEND_WS_URL=ws://<backend_host>:8000` и `BACKEND_HTTP_URL=http://<backend_host>:8000` в окружении или `.env` соответствующей машины.

---

## Изменение `.env` в рантайме

Streamlit **не подхватывает** изменения `.env` без перезапуска. После правки `.env`:

1. Остановить streamlit (`Ctrl+C` в терминале).
2. Перезапустить `streamlit run engagement_app.py`.
