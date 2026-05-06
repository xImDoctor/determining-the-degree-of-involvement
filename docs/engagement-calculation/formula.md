# Формула engagement и веса компонентов

---

## Итоговая формула

```
engagement_raw = 0.42 * emotion_score  +  0.33 * eye_score  +  0.25 * head_pose_score
engagement     = clip(engagement_raw, 0, 1)
```

Источник: [`EngagementCalculator.calculate`](../../backend/app/services/video_processing/engagement_calculator.py#L266-L273), константа `WEIGHTS` в [`engagement_calculator.py:43-47`](../../backend/app/services/video_processing/engagement_calculator.py#L43-L47).

После взвешенной суммы сырой score проходит через адаптивное временное сглаживание (см. [smoothing-and-trend.md](smoothing-and-trend.md)).

---

## Веса компонентов

| Компонент | Вес | Источник обоснования |
|-----------|-----|----------------------|
| `emotion_score` | **0.42** (42%) | Buono et al., 2022 (*Multimedia Tools and Applications*) – корреляция subjective engagement с эмоциями составляет ρ ≈ 0.38, показатель базируется на LSTM-предсказаниях эмоций |
| `eye_score` | **0.33** (33%) | Dewi et al., 2022 (*Electronics*, 11(19), 3183) – снижение показателя EAR надёжно ассоциируется с усталостью; второй по силе сигнал |
| `head_pose_score` | **0.25** (25%) | Raca, Kidzinski & Dillenbourg, 2015 (EDM); Sümer et al., 2021 (*IEEE TAffC*) – метрика является надёжным индикатором визуального внимания в классной среде и переносится в т.ч. на гаджеты |

Сумма весов соответственно: `0.42 + 0.33 + 0.25 = 1.00`.

Подробности по публикациям и методологии см. в [literature.md](literature.md).

---

## Почему такое распределение

- **Эмоции (42%)** – прямой сигнал эмоционального состояния. Улыбка/удивление (активные вовлечённые эмоции) сильнее всего коррелируют с *subjective engagement*. Даёт устойчивый базовый сигнал и при отсутствии движения.
- **Состояние глаз (33%)** – критично для детекции усталости *(drowsiness)* и потери фокуса. Менее информативно, чем эмоции, при коротких сессиях, но становится важным показателем при длительных (усталость глаз, учащённое/редкое моргания).
- **Положение/поза головы (25%)** – вспомогательный индикатор, штраф при явном отвороте головы. Отклонение взгляда от экрана не всегда означает отсутствие вовлечённости (учащийся может обдумывать материал, смотреть в тетрадь или другие источники), но в общем случае отвлечение от экрана свидетельствует о переключении фокуса и снижении вовлечённости. Таким образом, метрика имеет наименьший вес в формуле.

---

## Диапазоны и клиппинг

- `emotion_score ∈ [0, 1]` – произведение `emotion_weight * confidence * confidence_penalty`, оба множителя в `[0, 1]`.
- `eye_score ∈ [0, 1]` – `min(1.0, base_score * blink_modifier)`, где `base_score ∈ {0.1, 0.4, 0.7, 1.0}`.
- `head_pose_score ∈ {0.2, 0.5, 0.8, 1.0}` – четыре дискретных уровня из `HEAD_POSE_STATE_SCORES`.

`engagement_raw` теоретически лежит в `[0, 1]`, но код применяет `max(0, min(1, engagement_raw))` страховочно ([engagement_calculator.py:273](../../backend/app/services/video_processing/engagement_calculator.py#L273)).

`score` и `score_raw` округляются до **3 знаков после запятой** перед сериализацией ([engagement_calculator.py:305-306](../../backend/app/services/video_processing/engagement_calculator.py#L305-L306)).

---

## Пример полного расчёта

> **Конфигурация для расчётов ниже** (default-значения констант на момент написания):
>
> - **`WEIGHTS`:** `emotion=0.42`, `eye=0.33`, `head_pose=0.25`
> - **`EMOTION_WEIGHTS`** (релевантные): `Happiness=1.0`, `Anger=0.1`. **`confidence_threshold = 0.55`** → `confidence_penalty = 1.0` при `conf ≥ 0.55`
> - **EAR thresholds** в [`analyze_ear.classify_attention_by_ear`](../../backend/app/services/video_processing/analyze_ear.py#L185): `alert=0.30`, `drowsy=0.20`, `very_drowsy=0.15`; на верхней ветке дополнительная развилка по `blink_count ∈ [10, 25]` (см. [../pipeline/eye-aspect-ratio.md](../pipeline/eye-aspect-ratio.md#классификация-attention_state))
> - **`EAR_STATE_SCORES`:** `Alert=1.0`, `Normal=0.7`, `Drowsy=0.4`, `Very Drowsy=0.1`
> - **Blink rate modifier:** `1.10` для `10 ≤ rate ≤ 25`, `0.95` при `rate < 5`, `0.90` при `rate > 30`, иначе `1.00`. В текущей версии пайплайна `timestamp=datetime.now()` передаётся автоматически и modifier активен, но при автономном вызове без `timestamp` возникает `elapsed_time = None` и значение остаётся `1.00` (модификатор не применяется)
> - **HPE thresholds** в [`analyze_head_pose.classify_attention_state`](../../backend/app/services/video_processing/analyze_head_pose.py#L160): `Highly Attentive` при `|pitch|<10°, |yaw|<15°`; `Distracted` при `|pitch|<30°, |yaw|<40°` (полная таблица - в [../pipeline/head-pose-estimation.md](../pipeline/head-pose-estimation.md#классификация-attention_state))
> - **`HEAD_POSE_STATE_SCORES`:** `Highly Attentive=1.0`, `Attentive=0.8`, `Distracted=0.5`, `Very Distracted=0.2`
> - **`THRESHOLDS` engagement:** `high=0.75`, `medium=0.50`, `low=0.25`
>
> При изменении любой из этих констант в коде расчёты ниже могут не воспроизвестись.

**Вход:** `Happiness, conf=0.85`; EAR `attention_state=Alert`, `blink_count=12`, `elapsed_time=60s`; HPE `attention_state=Highly Attentive`.

| Этап | Расчёт | Значение |
|------|--------|----------|
| `emotion_score` | `1.0 (Happiness) * 0.85 * 1.0 (conf ≥ 0.55)` | `0.850` |
| `base_score` EAR | `EAR_STATE_SCORES["Alert"]` | `1.0` |
| `blink_rate_per_min` | `(12 / 60) * 60` | `12` |
| `blink_modifier` | `10 ≤ rate ≤ 25` → `1.1` | `1.10` |
| `eye_score` | `min(1.0, 1.0 * 1.1)` | `1.000` (clip) |
| `head_pose_score` | `HEAD_POSE_STATE_SCORES["Highly Attentive"]` | `1.000` |
| `engagement_raw` | `0.42·0.85 + 0.33·1.0 + 0.25·1.0` | `0.937` |
| После clip | `max(0, min(1, 0.937))` | `0.937` |
| `score` (сглаженный) | стабильное состояние → `mean(history[-15:])` | `~0.93` |
| `level` | `0.937 ≥ 0.75` | `High` |
| `trend` | зависит от истории | `stable`/`rising` |

**Вход «негативный»** (иллюстрация автономного вызова без `timestamp` – в текущем пайплайне `timestamp` явно передаётся): `Anger, conf=0.92`; EAR `Drowsy` (без `timestamp` → `blink_modifier = 1.0`); HPE `Distracted`.

| Этап | Расчёт | Значение |
|------|--------|----------|
| `emotion_score` | `0.1 (Anger) * 0.92 * 1.0 (conf ≥ 0.55)` | `0.092` |
| `base_score` EAR | `EAR_STATE_SCORES["Drowsy"]` | `0.4` |
| `blink_modifier` | `elapsed_time` не передан → modifier неактивен | `1.00` |
| `eye_score` | `min(1.0, 0.4 * 1.0)` | `0.400` |
| `head_pose_score` | `HEAD_POSE_STATE_SCORES["Distracted"]` | `0.500` |
| `engagement_raw` | `0.42·0.092 + 0.33·0.4 + 0.25·0.5` | `0.296` |
| `level` | `0.25 ≤ 0.296 < 0.50` | `Low` |

---

## Ограничение модели

Веса `{0.42, 0.33, 0.25}` определены в коде как class-attribute:

```python
# engagement_calculator.py:43-47
WEIGHTS = {"emotion": 0.42, "eye": 0.33, "head_pose": 0.25}
```

> Переопределить можно только правкой кода. Нет env-переменных или аргументов конструктора для их настройки. Это намеренно – веса калибровались на академических данных и не предназначены для произвольного тюнинга в рантайме.
