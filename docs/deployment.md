# Развёртывание бэкенда

## Docker Compose (рекомендуемый способ)

Запускает Redis и бэкенд одной командой:

```bash
docker compose up -d
```

Остановка:
```bash
docker compose down
```

---

## Ручная установка

### Требования

- **Python**: 3.12+
- **Redis**: 7.0+

### Установка зависимостей

```bash
cd backend
pip install .
```

### Запуск Redis

```bash
sudo systemctl start redis-server
# или
redis-server --requirepass password
```

### Запуск бэкенда

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Переменные окружения

См. [environment-variables.md](environment-variables.md).

---

## Проверка работоспособности

```bash
curl http://localhost:8000/health
# {"status":"healthy","version":"1.0.0"}
```
