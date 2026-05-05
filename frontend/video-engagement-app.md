# Streamlit-приложение `video_engagement_app.py`

**Файл:** [`frontend/video_engagement_app.py`](../../frontend/video_engagement_app.py)


Приложение позволяет пользователю смотреть видео (подразумевается **учебное**, учитывая цель и задачи системы) по URL, одновременно анализируя его вовлечённость через веб-камеру. Метрики синхронизируются с временной шкалой видео, а отдельный график показывает, как менялась вовлечённость на конкретных секундах ролика.

---

## Отличия от `engagement_app.py`

| Признак | `engagement_app.py` | `video_engagement_app.py` |
|---------|---------------------|---------------------------|
| Источник стимула (просматриваемое видео) | Нет (только наблюдение потока веб-камеры) | Видеозапись по URL (`.mp4`, `.webm`, `.ogg`) |
| Кастомный компонент | – | HTML5-плеер с обратной связью по `currentTime` |
| Синхронизация с временной шкалой видео | – | Да, отдельный график engagement по секундам видео |
| CSV-экспорт метрик | – | Да |
| Поле `video_timestamp` в WS | Не используется | Передаётся с каждым кадром, эхо в ответе |
| `room_id` / `name` | `engagement-app` / `engagement-user` | `video-engagement-app` / `video-engagement-user` |

Загрузка `.env`, разрешение веб-камеры (320×240, 15 FPS), health-check backend и общая структура `session_state` совпадают с [`engagement_app.py`](engagement-app.md).

---

## Запуск

### Предварительные требования

1. Запущен Redis (см. [../backend/deployment.md](../backend/deployment.md))
2. Запущен backend: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000`
3. Установлены frontend-зависимости: `cd frontend && pip install -r requirements.txt`
4. Подключена веб-камера (camera 0)
5. Доступ к видео по HTTPS-URL **с разрешающими CORS-заголовками** или локальный файл, отданный со встроенного сервера

### Команда

```bash
cd frontend
streamlit run video_engagement_app.py
```

Откроется на `http://localhost:8501`.

### Конфигурация через `.env`

Приложение читает `BACKEND_WS_URL` и `BACKEND_HTTP_URL` через `python-dotenv` (`load_dotenv()` без аргументов, [video_engagement_app.py:16](../../frontend/video_engagement_app.py#L16)). При запуске из `frontend/` подхватится `frontend/.env`. Шаблон – [`frontend/.env.example`](../../frontend/.env.example):

```
BACKEND_WS_URL=ws://localhost:8000
BACKEND_HTTP_URL=http://localhost:8000
```

Системные переменные окружения имеют приоритет над `.env`. Подробнее – [environment.md](environment.md).

---

## Интерфейс

### Компоновка

Интерфейс имеет следующую структуру, представленную на ascii-схеме ниже:
```
┌─────────────────────────────────────────────────────────────┐
│            URL видео: [text input              ]            │
└─────────────────────────────────────────────────────────────┘
┌──────────────────────────────────┬──────────────────────────┐
│  Видео                           │  Веб-камера              |
│  (2/3)                           │  (1/3)                   |
│                                  │                          │
│  [HTML5 player]                  │  [камера]                │
│  Воспроизведение – 12.3с/180с    │ [Запустить] [Стоп]       │
│                                  │  ──────                  │
│                                  │   Показатели             │
│                                  │  Eye | HPE | Emo         │
│                                  │  Вовлечённость           │
│                                  │  Эмоция                  │
│                                  │  Pitch | Yaw | Roll      │
└──────────────────────────────────┴──────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                 Аналитика в реальном времени                │ 
├──────────────────────────────────┬──────────────────────────┤
│  [engagement по времени видео]   │  [распределение эмоций]  │
├──────────────────────────────────┼──────────────────────────┤
│  [положение головы]              │  [EAR]                   │
└──────────────────────────────────┴──────────────────────────┘
                       [Экспорт CSV]
```

### Поле URL видео

Простой `st.text_input`. Видеоплеер не показывается до ввода URL. При смене URL компонент перезагружает `<video>` и сбрасывает позицию.

### Видеоплеер

Кастомный Streamlit-компонент `frontend/components/video_player/`. HTML5 `<video controls preload="metadata">` внутри iframe. Подробности см. [ниже](#кастомный-компонент-video_player).

### Управление веб-камерой

Кнопки **"▶️ Запустить"** / **"⏹️ Стоп"**. Поведение аналогично `engagement_app.py`:
- старт: `webcam_running=True`, `needs_reset=True`, `cv2.VideoCapture(0)` с разрешением 320×240, 15 FPS;
- стоп: освобождение камеры и WS-соединения; графики остаются на экране.

### Метрики (правая колонка)

Три блока компонентов в строку:
- `Eye: {attention_state} ({eye_score})`
- `HPE: {attention_state} ({head_pose_score})`
- `Emo: {emotion_score}`

Под ними – общая вовлечённость, эмоция и три отдельных метрики `Pitch`/`Yaw`/`Roll` через `st.metric` (по сравнению с `engagement_app.py`, где это одна строка caption-ов – здесь три полноценных metric-блока).

### Графики

Четыре Plotly-графика в сетке 2×2:

| Слот | График |
|------|--------|
| Верхний левый | **Engagement по времени видео** – `score` vs `video_timestamp`, с цветовыми зонами уровней `High`/`Medium`/`Low`/`Very Low` |
| Верхний правый | Распределение эмоций (donut) |
| Нижний левый | Положение головы (Pitch/Yaw/Roll) |
| Нижний правый | EAR с пунктирной линией порога 0.25 |

> Уникальность video-приложения – **первый график**. Точки на оси X идут не по wall-clock от старта камеры, а по реальной позиции в видеоролике. Если пользователь перемотал видео назад, то следующая точка ляжет назад по X. Если поставил на паузу – новые точки будут накапливаться на той же X-координате.

### CSV-экспорт

Кнопка `📥 Экспорт данных (CSV)` появляется, когда есть хотя бы одна запись. Скачивается `engagement_data.csv` с полями:

```
video_time_s, emotion, confidence, engagement_score,
engagement_level, ear, pitch, yaw, roll
```

История экспорта (`st.session_state.export_data`) накапливается до конца Streamlit-сессии. Она **не очищается** кнопкой "⏹️ Стоп", обнуляется только при следующем "▶️ Запустить" (`needs_reset = True`).

---

## Архитектура: основной цикл + плеер в `session_state`

Структура совпадает с `engagement_app.py`: один блокирующий `while st.session_state.webcam_running:` цикл ([video_engagement_app.py:553](../../frontend/video_engagement_app.py#L553)), внутри которого читается кадр камеры, отправляется на backend и обновляются метрики.

Особенность данного video-приложения – **синхронизация с плеером**, который существует параллельно:

1. Кастомный компонент `video_player(url, height, key="main_player")` отрисовывается до начала цикла ([video_engagement_app.py:421](../../frontend/video_engagement_app.py#L421)). Его возвращаемое значение Streamlit автоматически кладёт в `st.session_state["main_player"]`.

2. Компонент пушит `setComponentValue` **только** на событиях `play` / `pause` / `seeked` / `loadedmetadata`. Между этими событиями `currentTime` в Python-коде "застывает" – чтобы не дёргать rerun слишком часто и не рвать захват камеры.

3. Внутри цикла на каждом кадре читается `st.session_state["main_player"]` ([video_engagement_app.py:566](../../frontend/video_engagement_app.py#L566)). Если значение изменилось – обновляется снимок (`player_snapshot`) и wall-clock метка его получения (`player_snapshot_wall`).

4. `video_timestamp`, отправляемый в backend, **интерполируется** между снимками по реальному времени ([video_engagement_app.py:572-579](../../frontend/video_engagement_app.py#L572-L579)):

   ```python
   snap = st.session_state.player_snapshot
   if snap and snap.get("currentTime") is not None:
       base_ts = snap["currentTime"]
       if snap.get("playing"):
           video_ts = base_ts + (current_time() - st.session_state.player_snapshot_wall)
       else:
           video_ts = base_ts
   ```

   На паузе `video_ts` фиксируется. При воспроизведении – линейно растёт по wall-clock от момента последнего снимка.

5. `api_client.send_frame(frame, video_timestamp=video_ts)` возвращает 3-tuple: `(processed_frame, results, echoed_timestamp)`. Поле `video_timestamp` приходит обратно эхом, чтобы клиент мог надёжно сопоставить ответ с моментом видео (ответ может прийти позже отправки).

6. Камера и `api_client` сохраняются в `st.session_state` (`camera`, `api_client._ws`). При rerun, который триггерит компонент плеера, ресурсы **переиспользуются** – освобождение происходит только при явной остановке (`webcam_running = False`).

### Ключи у `plotly_chart`

Графики отрисовываются через [`render_charts(placeholders, suffix)`](../../frontend/video_engagement_app.py#L311) с `suffix="last"` при выключенной камере и `suffix=str(n)` в цикле, где `n` – `chart_update_count`, инкрементируется каждые `CHART_UPDATE_INTERVAL = 15` кадров:

```python
placeholders["eng"].plotly_chart(eng_fig, key=f"eng_{suffix}", width="stretch")
```

Counter-suffix нужен, чтобы избежать `DuplicateWidgetID` при многократных вызовах внутри одного script run. Стартовая отрисовка (`suffix="last"`) и цикловая (`suffix=str(n)`) не пересекаются благодаря разному префиксу.

---

## Кастомный компонент `video_player`

**Файл компонента:** `frontend/components/video_player/index.html`
**Python-обёртка:** `frontend/components/video_player/__init__.py`

### Зачем нужен кастомный компонент

`st.video()` показывает видео, но **не отдаёт обратно** текущую позицию воспроизведения. Чтобы синхронизировать показатель engagement с временной шкалой видео, нужно знать `currentTime` в Python.

В качестве решения используется собственный компонент через `streamlit.components.v1.declare_component`. Он рендерит `<video>` в iframe и пушит на хост `currentTime` / `playing` / `duration` через `postMessage`.

### Возвращаемое значение

```python
video_player(url: str, height: int = 400, key: str | None = None) -> dict | None
# {"currentTime": float, "playing": bool, "duration": float}
```

До первого взаимодействия – `None`.

### Когда отправляются события

Компонент шлёт `setComponentValue` **только** на:
- `play`
- `pause`
- `seeked`
- `loadedmetadata`

`timeupdate` (~4 раза/сек) **не используется**, чтобы не триггерить rerun слишком часто и не нагружать Streamlit. Между событиями Python-сторона интерполирует `currentTime` сама (см. ниже).

### Wall-clock интерполяция currentTime

Между событиями плеера значение `video_timestamp`, отправляемое в backend, дорисовывается на графиках по реальному времени. Полная схема описана выше – см. [Архитектура: основной цикл + плеер в `session_state`](#архитектура-основной-цикл--плеер-в-session_state).

### Свойства iframe

`postMessage` на `streamlit:setFrameHeight` подгоняет высоту iframe под размер видео + контролов. Дополнительно `setInterval(setFrameHeight, 1000)` страхует от смещения при `<video controls>` resize.

---

## Связь с backend: поле `video_timestamp`

Pydantic-схемы backend поддерживают сквозной проброс временной метки видео:

- `FrameRequest.video_timestamp: float | None` – приходит с каждым кадром.
- `FrameResponse.video_timestamp: float | None` – backend возвращает то же значение **эхом**.

API-клиент [`frontend/api_client.py`](../../frontend/api_client.py):

```python
def send_frame(
    self, frame: np.ndarray, video_timestamp: float | None = None,
) -> tuple[np.ndarray | None, list[dict], float | None]:
    ...
```

Возвращает 3-tuple `(processed_frame, results, echoed_timestamp)`. `echoed_timestamp` используется как X-координата на графике вовлечённости по времени видео ([video_engagement_app.py:667-669](../../frontend/video_engagement_app.py#L667-L669)).

> Если backend не вернул поле, `echoed_timestamp = None`. В этом случае точка в `video_timestamps` не добавляется – график вовлечённости по времени видео останется пустым, остальные функции по-прежнему работают.

---

## Жизненный цикл сессии

```
1. Открыть http://localhost:8501
        │
        ▼
2. health-check backend → если жив, открывается секция
        │
        ▼
3. Ввести URL видео → отрисовка кастомного video_player (key="main_player")
        │
        ▼
4. "▶️ Запустить" веб-камеру:
        - cv2.VideoCapture(0), 320x240, 15 FPS
        - api_client.connect("video-engagement-app", "video-engagement-user")
        - очистка истории
        - вход в основной while-цикл захвата кадров
        │
        ▼
5. Пользователь смотрит видео, ставит на паузу, перематывает –
   плеер пушит setComponentValue (только на play / pause / seeked / loadedmetadata) →
   следующий шаг while-цикла читает свежий snapshot из session_state →
   video_ts интерполируется по wall-clock → send_frame(frame, video_timestamp=video_ts)
        │
        ▼
6. "⏹️ Стоп": webcam_running=False → выход из цикла →
   finally-блок: cap.release(), api_client.disconnect()
        │
        ▼
7. CSV доступен на скачивание (export_data сохранён до следующего "Запустить"
   или закрытия вкладки)
```

---

## Тонкости и ограничения

> Специфичные ограничения данного приложения, которые **дополняют** общий список из [limitations.md](limitations.md).

### Видео и CORS

Кастомный плеер запрашивает видео через стандартный `<video>` HTML5. Источник видео должен:
- быть доступен по HTTPS (или HTTP в локальной сети);
- разрешать запросы из origin-а Streamlit (`localhost:8501`) через CORS-заголовки;
- отдавать `Accept-Ranges: bytes` для корректной перемотки.

Видео с YouTube/RuTube/Vimeo **не поддерживаются**, т.к. это не прямые URL на файл. Используйте сервисы прямого хостинга или локальные файлы за reverse-proxy.

### Wall-clock интерполяция – приближение, не источник истины

Между событиями `play/pause/seeked` Python считает "сколько секунд прошло" по `current_time()` и прибавляет к последнему `currentTime`. Это работает, **если воспроизведение не было прервано пользователем без события**. Например:
- Браузер заблокировал autoplay и поставил на паузу – компонент пришлёт `pause`, всё ок.
- Скорость воспроизведения изменилась через `playbackRate` – Python об этом не узнает, оценка `video_ts` исказится.
- Видео буферизуется и плеер ждёт – `playing=true` остаётся, но реально продвижение замедлилось.

Для точного анализа таких случаев нужно расширить компонент событиями `ratechange` / `waiting` / `stalled`.

### CSV-экспорт сохраняется в RAM

`st.session_state.export_data` – обычный list of dict. Для длинного видео (1 час, 10 FPS = 36 000 записей) это около 5 МБ JSON-эквивалента в памяти браузера/Streamlit. Не критично, но для сессий часами лучше периодически экспортировать и нажимать "▶️ Запустить" заново (это очистит).

### Camera 0 фиксирована

Как и в основном приложении – `cv2.VideoCapture(0)` без параметризации.

### Не использовать одновременно с `engagement_app.py`

Оба приложения подключаются к разным комнатам, но обе используют **одну и ту же камеру**. Запустить параллельно нельзя.

---

## Файлы

| Файл | Назначение |
|------|-----------|
| `frontend/video_engagement_app.py` | UI и оркестрация (основной цикл, метрики, графики, CSV) |
| `frontend/components/video_player/__init__.py` | Python-обёртка кастомного компонента |
| `frontend/components/video_player/index.html` | HTML5-плеер + JS-мост к Streamlit через `postMessage` |
| `frontend/api_client.py` | WS-клиент с поддержкой `video_timestamp` (3-tuple возврата) |
| `frontend/.env.example` | Шаблон конфигурации фронтенда |
| `frontend/.env` | Локальная конфигурация (создаётся пользователем) |
| `backend/app/schemas/stream.py` | `FrameRequest.video_timestamp`, `FrameResponse.video_timestamp` |
