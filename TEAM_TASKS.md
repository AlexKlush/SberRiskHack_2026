# Задачи для команды — SberRiskHack 2026

Система будет проверяться на **скрытом датасете**. Нельзя хардкодить ничего под конкретные данные. Всё должно быть универсальным.

---

## Участник 1 — Улучшение промтов GigaChat (FeatureIdeator)

**Цель:** сделать так, чтобы GigaChat генерировал качественные идеи фичей на ЛЮБОМ датасете.

**Файл:** `src/agents/feature_ideator.py`

**Что делать:**
1. Открой `SYSTEM_PROMPT` и `USER_PROMPT_TEMPLATE`
2. Улучши few-shot примеры — добавь разнообразные примеры фичей:
   - Агрегация по ключу из доп. таблицы (mean, count, nunique)
   - Взаимодействие двух числовых колонок (ratio, difference)
   - Frequency encoding категориальной колонки
   - Количество записей в доп. таблице для данного ключа
3. Добавь в `<thinking>` подсказку: "Посмотри на join_keys — какие агрегации из доп. таблиц можно построить?"
4. Протестируй: `python run.py` — смотри что генерирует GigaChat в логе

**Важно:** НЕ упоминай конкретные имена таблиц/колонок (users, orders и т.д.) — всё через переменные из schema.

---

## Участник 2 — Улучшение промтов GigaChat (FeatureCoder)

**Цель:** сделать так, чтобы код от GigaChat реже падал.

**Файл:** `src/agents/feature_coder.py`

**Что делать:**
1. Открой `SYSTEM_PROMPT` и `USER_PROMPT_TEMPLATE`
2. Добавь в constraints больше safety-правил:
   - "Оборачивай каждый признак в try/except — если один не считается, пропусти его"
   - "Всегда проверяй что колонка существует перед обращением к ней"
   - "После merge проверяй что DataFrame не пустой"
3. Добавь в output_format пример полной функции-шаблона:
```python
def generate_features(df_train, df_test, extra_tables=None):
    import pandas as pd
    import numpy as np
    features = []
    # для каждого признака: try/except
    try:
        # ... вычисление
        features.append("feat_name")
    except Exception:
        pass
    # итоговые df
    df_train_out = df_train[[id_column, target_column] + features]
    df_test_out = df_test[[id_column] + features]
    return df_train_out, df_test_out
```
4. Протестируй на тестовом датасете

---

## Участник 3 — Универсальный fallback

**Цель:** усилить автоматические фичи, которые генерируются БЕЗ LLM.

**Файл:** `src/utils/fallback_features.py`

**Что делать:**
Текущий fallback уже универсальный. Можно добавить:
1. **Числовые статистики** — для каждой числовой колонки в train: z-score, квантильный бин
2. **Count encoding** — сколько раз каждое значение категориальной колонки встречается (абсолютно, не normalize)
3. **Missing flags** — флаг пропуска (1/0) для колонок с NaN
4. **Rank features** — ранг значения внутри группировки по ключевой колонке

**Важно:**
- НЕ хардкодь имена таблиц или колонок
- Используй `df_train.columns`, `df_train.dtypes` для автоопределения
- Агрегации считай по train или extra_tables, применяй к test через `.map()`
- Все фичи — числовые, fillna(0)

---

## Участник 4 — Тестирование на разных датасетах

**Цель:** убедиться что пайплайн работает на ЛЮБЫХ данных, не только на текущем.

**Что делать:**
1. Запусти на текущем датасете — запиши ROC-AUC
2. Создай минимальный синтетический датасет для быстрого теста:
```python
import pandas as pd
import numpy as np
n = 1000
train = pd.DataFrame({"id": range(n), "cat_a": np.random.choice(["x","y","z"], n), "num_b": np.random.randn(n), "target": np.random.randint(0,2,n)})
test = pd.DataFrame({"id": range(n, n+200), "cat_a": np.random.choice(["x","y","z"], 200), "num_b": np.random.randn(200), "target": np.random.randint(0,2,200)})
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
# Без доп. таблиц и readme — pipeline должен справиться
```
3. Запусти `python run.py` — должен завершиться без ошибок
4. Запусти `python src/utils/check_submission.py` — 13 чеков
5. Если что-то упало — запиши traceback, создай issue

---

## Участник 5 — Подготовка сабмита + CI

**Цель:** подготовить финальный архив и убедиться что он запускается с нуля.

**Что делать:**
1. Склонируй репо в ЧИСТУЮ папку (не ту где разрабатывали)
2. Пройди весь README с нуля: `uv venv` → `uv sync` → настрой `.env` → положи данные → `python run.py`
3. Убедись что `check_submission.py` проходит
4. Собери zip:
   - ВКЛЮЧИТЬ: `run.py`, `pyproject.toml`, `.env` (с токеном!), `src/`, `data/.gitkeep`, `output/.gitkeep`
   - ИСКЛЮЧИТЬ: `.venv/`, `catboost_info/`, `__pycache__/`, `.git/`, `data/*.csv`, `output/*.csv`
5. Проверь что в `.env` стоит реальный токен, не placeholder

---

## Общие правила

- **Всё должно быть dataset-agnostic** — никаких хардкодов под конкретные данные
- Перед коммитом: `git pull origin main`
- Не коммить `.env` с токеном, `data/`, `output/`, `.venv/`
- Не меняй `src/state.py`, `src/graph.py`, `run.py` без согласования

## Приоритеты

1. 🔴 Промты FeatureCoder (Уч. 2) — LLM-фичи сейчас падают, это главная проблема
2. 🔴 Промты FeatureIdeator (Уч. 1) — качество идей = качество фичей
3. 🟡 Универсальный fallback (Уч. 3) — страховка
4. 🟡 Тестирование (Уч. 4) — ловит баги до проверки
5. 🟢 Сабмит (Уч. 5) — финальный шаг
