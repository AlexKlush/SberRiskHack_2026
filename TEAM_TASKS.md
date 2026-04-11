# Задачи для команды — SberRiskHack 2026

Пайплайн уже работает. Ниже — что каждый может делать параллельно, чтобы поднять ROC-AUC.

---

## Участник 1 — EDA и ручные фичи

**Цель:** найти закономерности в данных, которые LLM мог пропустить.

**Что делать:**
```bash
# В папке feature-agent, с активированным .venv
python
```
```python
import pandas as pd

train = pd.read_csv("data/train.csv")
users = pd.read_csv("data/users.csv")
oi = pd.read_csv("data/order_items.csv")
orders = pd.read_csv("data/orders.csv")
products = pd.read_csv("data/products.csv")

# 1. Посмотри распределение таргета
print(train["target"].value_counts(normalize=True))

# 2. Какие колонки users коррелируют с target?
merged = train.merge(users, on="user_id")
for col in users.columns:
    if col != "user_id":
        print(f"{col}: corr={merged[col].corr(merged['target']):.4f}")

# 3. Топ-товары по reorder rate
prod_stats = oi.groupby("product_id")["reordered"].agg(["mean","count"])
print(prod_stats.sort_values("mean", ascending=False).head(20))

# 4. Проверь гипотезы:
# - Влияет ли day_of_week на повторные покупки?
# - Есть ли зависимость от количества заказов пользователя?
# - Какие aisle/department чаще повторно покупают?
```

**Результат:** список из 3-5 гипотез для новых признаков → передай Участнику 2.

---

## Участник 2 — Улучшение промтов для GigaChat

**Цель:** сделать так, чтобы GigaChat генерировал более качественные идеи фичей.

**Файл:** `src/agents/feature_ideator.py`

**Что делать:**
1. Открой файл, найди `SYSTEM_PROMPT` и `USER_PROMPT_TEMPLATE`
2. Добавь в `<few_shot_example>` конкретные примеры хороших фичей для ЭТОГО датасета:
```
Хорошая идея:
{"name": "user_prod_buy_count", "description": "Сколько раз user покупал этот product в прошлых заказах",
 "columns_used": ["user_id", "product_id"], "extra_tables_used": ["order_items", "orders"],
 "category": "CROSS_TABLE", "hypothesis": "Чем чаще покупал — тем вероятнее купит снова"}
```
3. Добавь в `<feature_categories>` новую категорию:
```
- USER_PRODUCT_HISTORY: признаки на уровне пары (user_id, product_id) — count покупок, последний заказ, среднее position in cart
```
4. Используй результаты EDA от Участника 1 для подсказок

**Результат:** коммит с улучшенными промтами → `git add ... && git commit && git push`

---

## Участник 3 — Ручные фичи в fallback

**Цель:** добавить вручную самые сильные фичи, не завися от LLM.

**Файл:** `src/utils/fallback_features.py`

**Что делать:**
Добавь новые фичи в функцию `generate_fallback_features`. Примеры что стоит добавить:

```python
# 1. Среднее position in cart для товара (add_to_cart_order)
# Товары которые добавляют первыми — базовые, их чаще покупают
prod_cart_pos = oi.groupby("product_id")["add_to_cart_order"].mean()

# 2. Сколько дней прошло с последнего заказа пользователя
# (из orders.csv — последний days_since_prior_order)
last_order = orders.sort_values("order_number").groupby("user_id").last()

# 3. Доля повторных покупок по категории товара (aisle_id)
# Популярные категории чаще покупают повторно
aisle_reorder = oi.merge(products, on="product_id").groupby("aisle_id")["reordered"].mean()

# 4. Количество заказов содержащих этот товар / общее число заказов user
# (мера "лояльности" к товару)
```

**Важно:**
- Считай агрегации по TRAIN или extra_tables
- Применяй к test через `.map()` или `.merge()`
- Заполняй NaN через `.fillna(0)`
- Каждая фича — числовая (int/float)

**Результат:** коммит → push

---

## Участник 4 — Тестирование и стабильность

**Цель:** убедиться что пайплайн не падает и проходит все проверки.

**Что делать:**
1. Склонируй репо, настрой окружение (по README)
2. Запусти `python run.py` и запиши:
   - Сколько фичей сгенерировано?
   - Какой ROC-AUC?
   - Были ли ошибки в логе?
3. Запусти `python src/utils/check_submission.py` — все 13 чеков должны пройти
4. Попробуй запустить **второй раз** — результат должен быть стабильным
5. Если есть другой датасет — протестируй на нём

**Если что-то упало:**
- Скопируй полный traceback
- Создай issue в GitHub или напиши в чат команды

**Результат:** отчёт по стабильности, список багов если есть

---

## Участник 5 — Подготовка сабмита

**Цель:** подготовить финальный архив для загрузки на платформу.

**Что делать:**
1. Убедись что `python src/utils/check_submission.py` проходит ✓
2. Проверь что `.env` содержит реальный токен (не placeholder)
3. Собери zip-архив:
```bash
# Из папки feature-agent:
cd ..
zip -r submission.zip feature-agent/ \
  -x "feature-agent/.venv/*" \
  -x "feature-agent/catboost_info/*" \
  -x "feature-agent/__pycache__/*" \
  -x "feature-agent/src/__pycache__/*" \
  -x "feature-agent/src/agents/__pycache__/*" \
  -x "feature-agent/src/utils/__pycache__/*" \
  -x "feature-agent/data/*.csv" \
  -x "feature-agent/data/*.txt" \
  -x "feature-agent/output/*" \
  -x "feature-agent/.git/*"
```
Или на Windows — вручную заархивируй папку, исключив `.venv`, `data/`, `output/`, `.git/`

4. Проверь что в архиве ЕСТЬ:
   - `.env` (с токеном!)
   - `run.py`
   - `pyproject.toml`
   - все файлы `src/`
   - `data/.gitkeep`
   - `output/.gitkeep`

5. Загрузи на платформу

---

## Общие правила

- **Перед коммитом** всегда делай `git pull origin main` — кто-то мог запушить раньше
- **Не коммить** `.env` с реальным токеном, `data/`, `output/`, `.venv/`
- **Не меняй** `src/state.py`, `src/graph.py`, `run.py` без согласования — это ядро пайплайна
- **Тестируй локально** перед пушем: `python run.py` должен завершиться без ошибок

## Приоритеты по влиянию на ROC-AUC

1. 🔴 Ручные фичи в fallback (Участник 3) — самый быстрый прирост
2. 🔴 EDA (Участник 1) — даёт инсайты для всех остальных
3. 🟡 Улучшение промтов (Участник 2) — если LLM заработает, это +0.02-0.05
4. 🟢 Тестирование (Участник 4) — страховка от провала на платформе
5. 🟢 Сабмит (Участник 5) — финальный шаг
