# Разработка

Проект использует SemVer.
Измените GHCR_REPOSITORY_OWNER при создании форка.

## Тесты

### Доступные тесты

| Файл                         | Кол-во | Описание                                            |
|------------------------------|--------|-----------------------------------------------------|
| `tests/test_api.py`          | 5      | HTTP endpoints (health, rooms, clients)             |
| `tests/test_room_service.py` | 13     | Room lifecycle (создание, удаление клиентов/комнат) |

### Запуск

```bash
# Все тесты
python -m pytest backend/tests/ -v

# Конкретный файл
python -m pytest backend/tests/test_api.py -v

# Один тест
python -m pytest backend/tests/test_api.py::test_health_check -v

# По названию
python -m pytest -k "test_get_rooms" -v

# С покрытием
python -m pytest backend/tests/ --cov=backend/app --cov-report=html
```

### Пример вывода

```
backend/tests/test_api.py::test_health_check PASSED                    [  5%]
backend/tests/test_api.py::test_get_rooms_empty PASSED                 [ 11%]
backend/tests/test_api.py::test_get_rooms_with_data PASSED             [ 16%]
backend/tests/test_api.py::test_get_clients_in_room_success PASSED     [ 22%]
backend/tests/test_api.py::test_get_clients_in_room_not_found PASSED   [ 27%]
...
===================== 18 passed in 0.45s =====================
```

---

## Качество кода

### Линтинг (Ruff)

```bash
cd backend
ruff check .          # Проверить
ruff check --fix .    # Исправить автоматические
ruff format .         # Форматировать
```

### Типизация (MyPy)

```bash
cd backend
mypy app/ --warn-return-any --no-implicit-optional
```

### Полная проверка

```bash
cd backend
ruff check . && mypy app/ && pytest
```

---

## Зависимости

### Установка

```bash
# Backend
cd backend
pip install -e ".[dev]"

# Frontend
cd frontend
pip install -e .
```

### Требования

- **Python**: 3.12+
- **Redis**: 7.0+

---

## Запуск

### Backend

```bash
# Redis должен быть запущен
redis-server

# В другом терминале
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
streamlit run engagement_app.py
```

---

## Переменные окружения

См. [environment-variables.md](environment-variables.md).
