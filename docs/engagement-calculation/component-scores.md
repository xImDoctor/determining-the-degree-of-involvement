# Расчёт частных scores компонентов

Три метода `EngagementCalculator` переводят полученные сырые ML-данные в числовые scores `[0, 1]`:

1. [Emotion score](component-scores.md#1-emotion-score)
2. [EAR score + модификатор по `blink_rate_per_min`](component-scores.md#2-eye-score)
3. [Head pose score](component-scores.md#3-head-pose-score)

---

## 1. Emotion score

Реализация: [`calculate_emotion_score`](../../backend/app/services/video_processing/engagement_calculator.py#L147-L170).

```
emotion_score = emotion_weight * confidence * confidence_penalty
```

### Веса эмоций

`EMOTION_WEIGHTS` ([engagement_calculator.py:51-60](../../backend/app/services/video_processing/engagement_calculator.py#L51-L60)) определены следующим образом:

| Эмоция | Вес | Интерпретация |
|--------|-----|---------------|
| `Happiness` | **1.0** | Максимальная *позитивная* вовлечённость |
| `Surprise` | **0.8** | Интерес, удивление – *"продуктивная"* эмоция |
| `Neutral` | **0.6** | Спокойное внимание |
| `Contempt` | **0.4** | Скептицизм, *частичная* вовлечённость |
| `Fear` | **0.3** | Беспокойство |
| `Sadness` | **0.2** | Грусть, усталость |
| `Anger` | **0.1** | Фрустрация |
| `Disgust` | **0.1** | Отторжение |

Указанные веса инициированы на основании стандартной шкалы Ekman-based эмоций и их связи с engagement. Подробнее см. в [literature.md](literature.md).

### Confidence penalty

Штраф при неуверенном предсказании (вероятность, что модель *"угадывает"*):

```
confidence < 0.55:  penalty = confidence / 0.55     # линейный штраф
confidence ≥ 0.55:  penalty = 1.0                   # без штрафа
```

Пороговое значение в `0.55` по умолчанию совпадает с `confidence_threshold` в `EmotionRecognizer`. В текущей реализации до `EngagementCalculator` уже попадает отфильтрованный confidence (эмоции ниже порога уже заменены на `Neutral` с `conf = 0.495 = 0.55 · 0.9`) – поэтому ветка с penalty срабатывает редко, но всё ещё является инструментом отсеивания неуверенных предсказаний, в основном при ambiguous-fallback `("Neutral", 0.5)` из ambiguity-фильтра (отсеивающего *неоднозначные предсказания*). 

> Это необходимо также учитывать при перенастройке порога `confidence` здесь и в ML-пайплайне.


### Примеры

| Вход | Расчёт | `emotion_score` |
|------|--------|-----------------|
| `Happiness`, `conf=0.87` | `1.0 * 0.87 * 1.0` | `0.870` |
| `Neutral`, `conf=0.50` (ambiguous) | `0.6 * 0.50 * (0.50/0.55)` | `0.273` |
| `Neutral`, `conf=0.495` (threshold fallback) | `0.6 * 0.495 * (0.495/0.55)` | `0.267` |
| `Anger`, `conf=0.92` | `0.1 * 0.92 * 1.0` | `0.092` |
| `Surprise`, `conf=0.70` | `0.8 * 0.70 * 1.0` | `0.560` |

---

## 2. Eye score

Реализация: [`calculate_eye_score`](../../backend/app/services/video_processing/engagement_calculator.py#L172-L208).

```
eye_score = min(1.0, base_score * blink_modifier)
```

### Базовый score из `attention_state`

`EAR_STATE_SCORES` ([engagement_calculator.py:68-73](../../backend/app/services/video_processing/engagement_calculator.py#L68-L73)):

| `attention_state` | Base score | Порог EAR |
|-------------------|------------|-----------|
| `Alert` | **1.0** | `avg_ear ≥ 0.30`|
| `Normal` | **0.7** | `avg_ear ≥ 0.22`|
| `Drowsy` | **0.4** | `avg_ear ≥ 0.17` |
| `Very Drowsy` | **0.1** | `avg_ear < 0.17` |

`attention_state` вычисляется **в `analyze_ear.py`** через [`classify_attention_by_ear`](../../backend/app/services/video_processing/analyze_ear.py#L185) – калькулятор его не пересчитывает, а только берёт готовую строку и ищет её в map.

> См. пояснение расчёту `attention_state` и по поведению `blink_count` vs `blink_rate` в [../pipeline/eye-aspect-ratio.md](../pipeline/eye-aspect-ratio.md#классификация-attention_state).


### Blink modifier

Дополнительный коэффициент из реальной частоты моргания в минуту (считается по `elapsed_time`, и это не то же самое, что `blink_rate` из исходного EAR-модуля, см. ссылку выше):

```python
blink_rate_per_min = (blink_count / elapsed_time) * 60
```

| Диапазон | Модификатор | Источник |
|----------|-------------|----------|
| `10 ≤ rate ≤ 25` | **x1.10** (бонус) | Нормальная частота, бодрствование |
| `rate < 5` | x0.95 | Гиперфокус или усталость глаз |
| `rate > 30` | x0.90 | Стресс, раздражение |
| остальное (`5..10`, `25..30`) | x1.00 | без изменений |

Подробнее в [заметке по модификаторам (modifiers.md)](modifiers.md).

### Fallback при `ear_data=None`

`eye_score = 0.5`. Нейтральное значение: модуль не тянет engagement ни вверх, ни вниз. 

> Интерпретация такого значения описана в [README.md](README.md#семантика-нейтрального-значения-05).

---

## 3. Head pose score

Реализация: [`calculate_head_pose_score`](../../backend/app/services/video_processing/engagement_calculator.py#L210-L224).

```
head_pose_score = HEAD_POSE_STATE_SCORES[attention_state]
```

`HEAD_POSE_STATE_SCORES` ([engagement_calculator.py:93-98](../../backend/app/services/video_processing/engagement_calculator.py#L93-L98)):

| `attention_state` | Score | Пороги |
|-------------------|-------|--------|
| `Highly Attentive` | **1.0** | `\|pitch\| < 10°`, `\|yaw\| < 15°` |
| `Attentive` | **0.8** | `\|pitch\| < 20°`, `\|yaw\| < 25°` |
| `Distracted` | **0.5** | `\|pitch\| < 30°`, `\|yaw\| < 40°` |
| `Very Distracted` | **0.2** | иначе |

Без модификаторов. Пороги определяются в [`analyze_head_pose.py`](../../backend/app/services/video_processing/analyze_head_pose.py#L160).

Fallback при `head_pose_data=None`: `head_pose_score = 0.5`.

> Интерпретация fallback'а подчиняется той же логике, что и в случае EAR ([README.md](README.md#семантика-нейтрального-значения-05)).

`roll` в классификацию не включён (см. [../pipeline/head-pose-estimation.md](../pipeline/head-pose-estimation.md#классификация-attention_state)).
