# SberRiskHack_2026
Hackathon "Код Риска" 2026. Presented by: Klushin Alexey, Karpov Alexey, Belyavskiy Denis, Kim Andrew, Yakimushkin Alexander

## Feature Agent — автогенерация признаков

Мультиагентная система на LangGraph + GigaChat-2-Max для автоматической генерации признаков в задаче бинарной классификации.

## Быстрый старт (пошагово)

### 1. Клонируй репозиторий

```bash
git clone https://github.com/AlexKlush/SberRiskHack_2026.git
cd SberRiskHack_2026
```

### 2. Убедись что Python 3.10+ установлен

```bash
python --version
```

### 3. Установи uv и создай окружение

```bash
pip install uv
python -m uv venv
python -m uv sync
```

### 4. Активируй окружение

**Windows PowerShell:**
```powershell
# Если выдаёт ошибку про Execution Policy — сначала выполни:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Затем активируй:
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 5. Настрой токен GigaChat

Открой файл `.env` и вставь токен от организаторов:
```
GIGACHAT_CREDENTIALS='ВСТАВЬ_СЮДА_СВОЙ_ТОКЕН'
GIGACHAT_SCOPE='GIGACHAT_API_CORP'
```

### 6. Положи данные

Скопируй файлы датасета в папку `data/`:
```
data/
  train.csv      <- обязательно
  test.csv       <- обязательно
  readme.txt     <- если есть
  users.csv      <- доп. таблицы (если есть)
  orders.csv
  ...
```

### 7. Запусти

```bash
python run.py
```

Время работы: **5-10 минут** в зависимости от размера данных.

### 8. Проверь результат

```bash
python src/utils/check_submission.py
```

Если все 13 проверок пройдены — готово к сабмиту.

Результаты:
```
output/train.csv  — [ID, target, feature_1, ..., feature_5]
output/test.csv   — [ID, feature_1, ..., feature_5]
```

## Частые проблемы

| Проблема | Решение |
|---|---|
| `uv: command not found` | Используй `python -m uv venv` и `python -m uv sync` |
| `activate: Execution Policy` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ModuleNotFoundError` | Убедись что активировал `.venv` перед запуском |
| Работает дольше 10 минут | CatBoost считает CV — подожди, лимит 570 сек |

## Архитектура

```
START → DataAnalyst → FeatureIdeator → FeatureCoder → FeatureEvaluator → OutputWriter → END
```

- **DataAnalyst** — читает CSV, определяет схему, агрегирует тяжёлые таблицы
- **FeatureIdeator** — 2 раунда по 5 идей признаков через GigaChat
- **FeatureCoder** — генерирует Python-код, исполняет в sandbox (retry при ошибке)
- **FeatureEvaluator** — объединяет все фичи + fallback, forward selection топ-5 по ROC-AUC
- **OutputWriter** — сохраняет итоговые CSV
