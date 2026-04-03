# Переменные окружения

| Переменная             | По умолчанию                                 | Описание                               |
|------------------------|----------------------------------------------|----------------------------------------|
| `REDIS_HOST`           | localhost                                    | Хост Redis                             |
| `REDIS_PORT`           | 6379                                         | Порт Redis                             |
| `REDIS_PASSWORD`       | (пустая строка)                              | Пароль Redis                           |
| `CORS_ALLOWED_ORIGINS` | http://localhost:8501,http://localhost:63342 | Разрешённые CORS-источники             |
| `EMOTION_DEVICE`       | auto                                         | Устройство для PyTorch (cpu/cuda/auto) |

Подробнее в `.env.example`.