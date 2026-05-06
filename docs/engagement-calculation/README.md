# Расчёт вовлечённости (Engagement)

Данная документация описывает модуль преобразования распознанных эмоций, EAR и HPE метрик в итоговый показатель вовлечённости `engagement score ∈ [0, 1]` с классификацией уровня и тенденции.

**Файл:** [`backend/app/services/video_processing/engagement_calculator.py`](../../backend/app/services/video_processing/engagement_calculator.py)

---

## Содержание

1. [Формула и веса](formula.md) – итоговая взвешенная сумма, обоснование весов
2. [Частные scores компонентов](component-scores.md) – как emotion, eye, head_pose превращаются в числа
3. [Модификаторы](modifiers.md) – штрафы и бонусы (blink rate, confidence penalty)
4. [Сглаживание и тренд](smoothing-and-trend.md) – адаптивное окно, определение тренда как `rising`/`falling`/`stable`
5. [Уровни и диапазоны](levels-and-ranges.md) – классификация `High`/`Medium`/`Low`/`Very Low`
6. [Научные источники](literature.md) – публикации, на которых основаны веса и модификаторы, методы оценки вовлечённости

---

## Входные данные

```python
EngagementCalculator.calculate(
    emotion: str,                      # из EmotionRecognizer (уже сглаженное значение)
    emotion_confidence: float,         # из EmotionRecognizer
    ear_data: EyeAspectRatioAnalyzeResult | None,
    head_pose_data: HeadPoseEstimateResult | None,
    timestamp: datetime | None = None, # опционально, для blink rate
) -> EngagementCalculateResult
```

Если `ear_data` или `head_pose_data = None` – соответствующий компонент получает нейтральное значение `0.5`. Поведение и влияние такого компонента на итоговую метрику описано в [данном разделе](README.md#семантика-нейтрального-значения-05).

---

## Выход

```python
@dataclass
class EngagementCalculateResult:
    score: float        # сглаженное значение в [0, 1], округлённое до 3 знаков
    score_raw: float    # сырое значение по текущему кадру
    level: Literal["High", "Medium", "Low", "Very Low"]     # уровень вовлечённости
    trend: Literal["rising", "falling", "stable"]   # тренд: растущая, убывающая, стабильная
    components: EngagementComponents    # отдельные компонентные scores, см. dataclass ниже
    frame_count: int    # порядковый номер кадра в сессии

@dataclass
class EngagementComponents:
    emotion_score: float
    eye_score: float
    head_pose_score: float
```

---

## Упрощённая схема

```
emotion, confidence     ─► calculate_emotion_score()   ─► emotion_score × 0.42  ┐
                                                                                │
ear_data + elapsed_time ─► calculate_eye_score()       ─► eye_score     × 0.33 ─┼─► sum ─► clip[0,1] ─► engagement_raw
                                                                                │
head_pose_data          ─► calculate_head_pose_score() ─► head_pose_score × 0.25┘

engagement_raw ──► adaptive smoothing ──► score
engagement_raw ──► trend window (30)  ──► trend
score          ──► threshold map      ──► level
```

---

## Семантика нейтрального значения `0.5`

Если EAR или HPE недоступны, подставляется `0.5` – не `0.0` и не `1.0`. Это  означает:

- **Модуль не включён** → не создаёт ни положительного, ни отрицательного вклада.
- `engagement_raw` при полностью отключённых EAR и HPE сводится к `0.42*emotion + 0.33*0.5 + 0.25*0.5 = 0.42*emotion + 0.29`.
- Для `Happiness (conf=1.0)` при отключённых EAR/HPE: `0.42*1.0 + 0.29 = 0.71` – попадает в `Medium`, но близко к `High`.

Поведение заложено в [`engagement_calculator.py:260, 263`](../../backend/app/services/video_processing/engagement_calculator.py#L260).

---

## Сброс состояния

`EngagementCalculator.reset()` очищает:
- `engagement_history` (окно сглаживания)
- `trend_history`
- `session_start_time`
- `frame_count`

Вызывается при начале новой сессии. В текущей WS-архитектуре сессия равносильна времени жизни одного `Client` в комнате; для каждого клиента создаётся свой экземпляр `EngagementCalculator` в handler'е WS-эндпоинта (см. [api/stream.py](../../backend/app/api/stream.py) и [architecture.md](../backend/architecture.md)).
