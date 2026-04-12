# Задачи для команды — SberRiskHack 2026

Система будет проверяться на **скрытом датасете**. Нельзя хардкодить ничего под конкретные данные. Всё должно быть универсальным.

## Архитектура (3 агента)

```
DataAnalyst → FeatureEngineer → EvaluatorWriter
```

| Агент | Файл | Роль |
|-------|------|------|
| DataAnalyst | `src/agents/data_analyst.py` | Чтение данных, определение схемы, ключей, таблиц |
| FeatureEngineer | `src/agents/feature_engineer.py` | LLM выбирает из меню операций + авто-пул |
| EvaluatorWriter | `src/agents/evaluator_writer.py` | Forward selection top-5, оценка, сохранение |

Ключевые утилиты:
- `src/utils/operations.py` — 13 безопасных операций (без exec/sandbox)
- `src/utils/scoring.py` — CatBoost CV + forward selection

---

## Участник 1 — Улучшение промтов FeatureEngineer

**Цель:** LLM выбирает операции из меню — нужно чтобы выбор был умнее.

**Файл:** `src/agents/feature_engineer.py`

**Что делать:**
1. Улучши `SYSTEM_PROMPT` и `USER_PROMPT_TEMPLATE`
2. Добавь few-shot примеры хороших комбинаций операций для разных типов данных
3. Подумай как передать больше контекста из readme (сейчас обрезается до 3000 символов)
4. Тестируй: запускай `python run.py` на разных датасетах, смотри какие операции предлагает LLM

**Важно:** НЕ упоминай конкретные имена таблиц/колонок — всё через schema.

---

## Участник 2 — Расширение библиотеки операций

**Цель:** добавить новые типы операций в пул.

**Файл:** `src/utils/operations.py`

**Что добавить:**
1. **QUANTILE_BIN** — разбиение числовой колонки на квантильные бины (pd.qcut)
2. **TIME_FEATURES** — если есть колонка-дата: день недели, месяц, час
3. **GROUP_RANK** — ранг значения внутри группы (groupby + rank)
4. **STD_AGG** — std агрегация из доп. таблицы (сейчас есть только через AGG, но можно выделить)

**Шаблон операции:**
```python
def my_operation(df_train, df_test, column, **kw):
    # Проверь что column существует
    if column not in df_train.columns:
        return None, None, None
    name = f"fe_{column}_myop"
    tr = ...  # numpy array длины len(df_train)
    te = ...  # numpy array длины len(df_test)
    return name, tr, te
```
После добавления: зарегистрируй в `OPERATIONS` словаре и обнови промпт в feature_engineer.py.

---

## Участник 3 — Тестирование на разных датасетах

**Цель:** убедиться что пайплайн работает на ЛЮБЫХ данных.

**Что делать:**
1. Запусти на каждом тестовом датасете — запиши ROC-AUC
2. Создай минимальный синтетический датасет:
```python
import pandas as pd, numpy as np
n = 1000
train = pd.DataFrame({"id": range(n), "cat_a": np.random.choice(["x","y","z"], n),
                       "num_b": np.random.randn(n), "target": np.random.randint(0,2,n)})
test = pd.DataFrame({"id": range(n, n+200), "cat_a": np.random.choice(["x","y","z"], 200),
                      "num_b": np.random.randn(200)})
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
```
3. Запусти `python run.py` — должен завершиться без ошибок
4. Запусти `python src/utils/check_submission.py` — 13 чеков
5. Если что-то упало — запиши traceback

---

## Участник 4 — Подготовка сабмита

**Цель:** подготовить финальный архив.

**Что делать:**
1. Склонируй репо в ЧИСТУЮ папку
2. Пройди README с нуля: `python -m uv venv` → `python -m uv sync` → `.env` → `python run.py`
3. Собери zip:
   - ВКЛЮЧИТЬ: `run.py`, `pyproject.toml`, `.env`, `src/`, `data/.gitkeep`, `output/.gitkeep`
   - ИСКЛЮЧИТЬ: `.venv/`, `catboost_info/`, `__pycache__/`, `.git/`, `data/*.csv`, `output/*.csv`
4. Проверь что в `.env` стоит реальный токен

---

## Общие правила

- **Всё должно быть dataset-agnostic** — никаких хардкодов
- Перед коммитом: `git pull origin main`
- Не коммить `.env` с токеном, `data/`, `output/`, `.venv/`
- Не меняй `src/state.py`, `src/graph.py`, `run.py` без согласования
