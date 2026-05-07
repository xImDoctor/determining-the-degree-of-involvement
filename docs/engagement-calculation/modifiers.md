# Модификаторы engagement

Помимо базовых весов и scores, система применяет точечные модификаторы, корректирующие отдельные компоненты. Задача – учесть факторы, не покрываемые дискретной классификацией `attention_state`.

---

## Blink rate modifier (для eye_score)

Реализация: [`calculate_eye_score`](../../backend/app/services/video_processing/engagement_calculator.py#L172-L208).

### Почему отдельный модификатор

`attention_state` из `analyze_ear.py` берёт текущий `avg_ear` и общий `blink_count`. Это не даёт прямого сигнала об **активности** глаз за минуту. Гиперконцентрация (редкое моргание) и стресс (частое моргание) – оба состояния с открытыми глазами и одинаковым `attention_state = Alert/Normal`, но имеющие разную семантику с точки зрения вовлечённости.

Модификатор даёт явный сигнал о активности глаз поверх базовой классификации.

### Формула

```python
blink_rate_per_min = (blink_count / elapsed_time) * 60

if 10 <= rate <= 25:    modifier = 1.10   # нормально, бонус
elif rate < 5:          modifier = 0.95   # редкое - гиперфокус или усталость
elif rate > 30:         modifier = 0.90   # частое - стресс
else:                   modifier = 1.00   # переходные диапазоны 5..10, 25..30
```

### Применение

```python
eye_score = min(1.0, base_score * modifier)
```

Clip к `1.0` нужен, чтобы бонус `x1.10` не выбрасывал `Alert` (base=1.0) за границу диапазона. Эффективно бонус работает только при `base_score ∈ {0.7, 0.4, 0.1}`.

### Требование

Модификатор активен **только** если `elapsed_time > 0` (известно время начала сессии). Для этого в первый вызов `calculate()` нужно передать `timestamp`:

```python
engagement_calculator.calculate(
    emotion="Happiness", emotion_confidence=0.9,
    ear_data=ear, head_pose_data=hp,
    timestamp=datetime.now(),    # важно!
)
```

`session_start_time` сохраняется при первой передаче `timestamp` ([engagement_calculator.py:248-249](../../backend/app/services/video_processing/engagement_calculator.py#L248-L249)).

В текущей версии пайплайна [`face_analysis_pipeline.py:183-189`](../../backend/app/services/video_processing/face_analysis_pipeline.py#L183-L189) `timestamp=datetime.now()` передаётся на каждом кадре, поэтому blink modifier активен по умолчанию. При автономном использовании калькулятора вовлечённости без `timestamp` (например, из тестов или утилит) `elapsed_time = None` и blink modifier остаётся `1.0`.

### Значения порогов и обоснование

- **10–25 морг./мин** – диапазон нормальной частоты моргания у бодрствующего человека.
- **< 5** – гиперфокус (сильная концентрация на задаче) или начальная стадия усталости глаз.
- **> 30** – признак стресса, раздражения или дискомфорта (яркое освещение, сухой воздух).

Данные значения определены в научных публикациях, подробнее см. в [literature.md](literature.md#частота-моргания).

---

## Confidence penalty (для emotion_score)

Не является модификатором поверх базового score – встроен в формулу `emotion_score` напрямую. Описан отдельно, т. к. логически выполняет ту же роль штрафа.

```
confidence < 0.55  →  multiplier = confidence / 0.55    (линейный штраф, до 0)
confidence ≥ 0.55  →  multiplier = 1.0                  (нейтрально)
```

Реализация: [`engagement_calculator.py:165-168`](../../backend/app/services/video_processing/engagement_calculator.py#L165-L168).

### Зачем

Если модель эмоций отдала `Happiness` с низкой уверенностью (`conf = 0.3`), наивный расчёт даст `emotion_score = 1.0 * 0.3 * 1.0 = 0.30`. С линейным штрафом: `1.0 * 0.3 * (0.3/0.55) = 0.164`. То есть **значительно** снижается вклад неуверенных предсказаний.

Порог `0.55` совпадает с `emotion_confidence_threshold`, что обеспечивает консистентность: ниже порога эмоция всё равно была бы заменена на `Neutral` в `EmotionRecognizer`.

---

## Что модификатором **не является**

### Temporal smoothing на уровне вычисления вовлечённости

Это не модификатор, а отдельный этап обработки после взвешенной суммы. Меняет не компоненты, а итоговый `score`. Описан в [smoothing-and-trend.md](smoothing-and-trend.md).

### Neutral fallback `0.5` для отсутствующих компонентов

Это не модификатор, а **заглушка** для случая `ear_data=None` / `head_pose_data=None`. Подставляется до формулы, задаёт *"нейтральный"* вклад компонента в сумму (подробнее в [README.md](README.md#семантика-нейтрального-значения-05)).

---

## Как расширять

Текущая реализация не имеет API для добавления пользовательских модификаторов – нужна правка кода. Места для вставки:

- `calculate_emotion_score` (`engagement_calculator.py:147`) – перед возвратом.
- `calculate_eye_score` (`engagement_calculator.py:172`) – после `base_score * blink_modifier`.
- `calculate_head_pose_score` (`engagement_calculator.py:210`) – перед возвратом.

Рекомендация: любой новый модификатор держать в диапазоне `[0.8, 1.2]` с clip к `[0, 1]` – иначе баланс итоговой формулы сместится непредсказуемо.
