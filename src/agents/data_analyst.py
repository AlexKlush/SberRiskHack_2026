"""Агент DataAnalyst — читает CSV, определяет схему, формирует контекст для LLM."""
from pathlib import Path

import pandas as pd

from src.state import AgentState

DATA_DIR = Path("data")


def _validate_columns(df_train, df_test, id_col, target_col, extra_tables):
    """Валидация определённых столбцов. Падает если что-то не так."""
    errors = []

    # 1. id и target не должны совпадать
    if id_col == target_col:
        errors.append(f"id_column и target_column совпадают: '{id_col}'")

    # 2. id_column должен быть в train
    if id_col not in df_train.columns:
        errors.append(f"id_column '{id_col}' не найден в train. Колонки train: {list(df_train.columns)[:10]}")

    # 3. id_column должен быть в test (после переименований)
    if id_col not in df_test.columns:
        errors.append(f"id_column '{id_col}' не найден в test. Колонки test: {list(df_test.columns)[:10]}")

    # 4. target_column должен быть в train
    if target_col not in df_train.columns:
        errors.append(f"target_column '{target_col}' не найден в train. Колонки train: {list(df_train.columns)[:10]}")

    # 5. target должен быть бинарным (бинарная классификация)
    if target_col in df_train.columns:
        unique_vals = set(df_train[target_col].dropna().unique())
        if not unique_vals <= {0, 1, 0.0, 1.0}:
            errors.append(
                f"target_column '{target_col}' не бинарный. "
                f"Уникальные значения ({len(unique_vals)}): {sorted(list(unique_vals))[:10]}"
            )

    # 6. id_column должен иметь высокую уникальность в обоих датасетах
    if id_col in df_test.columns:
        nunique = df_test[id_col].nunique()
        ratio = nunique / len(df_test) if len(df_test) > 0 else 0
        if ratio < 0.5:
            errors.append(
                f"id_column '{id_col}' имеет низкую уникальность в test: "
                f"{nunique}/{len(df_test)} ({ratio:.1%}). Вероятно, это не ID-столбец"
            )
    if id_col in df_train.columns:
        nunique_tr = df_train[id_col].nunique()
        ratio_tr = nunique_tr / len(df_train) if len(df_train) > 0 else 0
        if ratio_tr < 0.5:
            errors.append(
                f"id_column '{id_col}' имеет низкую уникальность в train: "
                f"{nunique_tr}/{len(df_train)} ({ratio_tr:.1%}). Вероятно, это не ID-столбец"
            )

    # 7. Должны быть фичи: в train ИЛИ в дополнительных таблицах
    feature_cols = [c for c in df_train.columns if c not in (id_col, target_col)]
    if not feature_cols and not extra_tables:
        errors.append("Нет столбцов-фичей ни в train, ни в дополнительных таблицах")

    if errors:
        msg = "COLUMN VALIDATION FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        print(f"  [DataAnalyst] {msg}")
        raise ValueError(msg)

    src = f"{len(feature_cols)} train cols"
    if extra_tables:
        src += f" + {len(extra_tables)} extra tables"
    id_uniq_train = df_train[id_col].nunique() if id_col in df_train.columns else 0
    id_uniq_test = df_test[id_col].nunique() if id_col in df_test.columns else 0
    print(f"  [DataAnalyst] Validation OK: id='{id_col}' "
          f"(train uniq={id_uniq_train}/{len(df_train)}, test uniq={id_uniq_test}/{len(df_test)}), "
          f"target='{target_col}' (binary), features from {src}")


def _find_id_column(df_train, df_test, readme_text):
    """Автоопределение столбца-идентификатора по нескольким стратегиям."""
    train_cols = list(df_train.columns)
    test_cols = list(df_test.columns)
    common_cols = [c for c in train_cols if c in test_cols]

    # ВСЕГДА пробуем case-insensitive матч для ещё не совпавших колонок
    already_matched = set(common_cols)
    test_lower = {}
    for c in test_cols:
        if c not in already_matched:
            test_lower[c.lower().strip()] = c
    renamed_count = 0
    for tc in train_cols:
        if tc in already_matched:
            continue
        matched = test_lower.get(tc.lower().strip())
        if matched:
            df_test.rename(columns={matched: tc}, inplace=True)
            common_cols.append(tc)
            already_matched.add(tc)
            renamed_count += 1
    if renamed_count > 0:
        print(f"  [DataAnalyst] Fixed {renamed_count} column name mismatches, common: {common_cols[:5]}")

    # Ищем столбец с "id" в названии
    id_keywords = ["_id", "id_", "row_id", "client_id", "sample_id", "index",
                   "application_id", "app_id", "user_id", "customer_id"]
    for c in common_cols:
        cl = c.lower()
        if cl == "id" or any(kw in cl for kw in id_keywords):
            return c

    # Ищем подсказки в readme
    if readme_text:
        for c in common_cols:
            for line in readme_text.split("\n"):
                cl, ll = c.lower(), line.lower()
                if cl in ll and ("идентификатор" in ll or "identifier" in ll
                                 or "unique" in ll or "уникальн" in ll):
                    return c

    # Первый общий столбец, уникальный в обоих датафреймах
    for c in common_cols:
        if df_train[c].nunique() == len(df_train) and df_test[c].nunique() == len(df_test):
            return c

    # Фоллбэк — ТОЛЬКО общий столбец (гарантируем что есть в обоих)
    if common_cols:
        return common_cols[0]

    # Крайний случай: первый столбец train, добавляем его в test
    fallback = train_cols[0]
    if fallback not in test_cols:
        # Переименовываем первый столбец test чтобы совпал
        df_test.rename(columns={test_cols[0]: fallback}, inplace=True)
        print(f"  [DataAnalyst] WARNING: no common cols, renamed test '{test_cols[0]}' -> '{fallback}'")
    return fallback


def _find_target_column(df_train, df_test, id_column, readme_text):
    """Автоопределение целевой переменной по нескольким стратегиям."""
    train_cols = list(df_train.columns)
    test_cols = list(df_test.columns)

    # Столбцы в train, которых нет в test — самый надёжный признак
    candidates = [c for c in train_cols if c not in test_cols and c != id_column]

    # Если кандидат один — точно он
    if len(candidates) == 1:
        return candidates[0]

    # Если кандидатов несколько — сначала ищем по известным именам среди них
    target_names = ["target", "label", "y", "class", "is_fraud", "default",
                    "default_flag", "churn", "is_default", "fraud", "outcome",
                    "result", "flag"]
    if candidates:
        for name in target_names:
            if name in candidates:
                return name
        # Среди кандидатов ищем бинарный столбец (0/1)
        for c in candidates:
            if df_train[c].dtype in ("int64", "float64"):
                vals = set(df_train[c].dropna().unique())
                if vals <= {0, 1, 0.0, 1.0}:
                    return c
        # Fallback: первый кандидат (не id-подобный)
        for c in candidates:
            cl = c.lower()
            if not (cl == "id" or "_id" in cl or "id_" in cl):
                return c
        return candidates[0]

    # Нет кандидатов (target есть и в train и в test) — ищем по имени
    for name in target_names:
        if name in train_cols and name != id_column:
            return name

    # Бинарный столбец (только 0/1)
    for c in train_cols:
        if c == id_column:
            continue
        if df_train[c].dtype in ("int64", "float64"):
            vals = set(df_train[c].dropna().unique())
            if vals <= {0, 1, 0.0, 1.0}:
                return c

    # Подсказки в readme
    if readme_text:
        for c in train_cols:
            if c == id_column:
                continue
            for line in readme_text.split("\n"):
                if c.lower() in line.lower() and (
                    "целев" in line.lower() or "target" in line.lower()
                ):
                    return c

    raise ValueError("Cannot determine target column")


def _detect_separator(path):
    """Определяем разделитель CSV-файла."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    for sep in [",", ";", "\t", "|"]:
        if sep in first_line:
            return sep
    return ","


def _safe_read_csv(path):
    """Безопасное чтение CSV: пробуем запятую, потом другие разделители."""
    for sep in [",", ";", "\t", None]:
        try:
            kw = {"encoding": "utf-8-sig"}
            if sep is None:
                kw["sep"] = sep
                kw["engine"] = "python"
            else:
                kw["sep"] = sep
            df = pd.read_csv(path, **kw)
            if len(df.columns) >= 2:
                return df
        except Exception:
            continue
    # Последняя попытка — дефолтные параметры
    return pd.read_csv(path)


def run(state: AgentState) -> dict:
    print("  [DataAnalyst] Reading data files...")
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    df_train = _safe_read_csv(train_path)
    df_test = _safe_read_csv(test_path)

    # Чистим имена столбцов от пробелов и BOM
    df_train.columns = [c.strip().strip("\ufeff") for c in df_train.columns]
    df_test.columns = [c.strip().strip("\ufeff") for c in df_test.columns]
    print(f"  [DataAnalyst] train: {df_train.shape}, test: {df_test.shape}")

    # Загружаем дополнительные таблицы
    extra_tables = {}
    for csv_file in sorted(DATA_DIR.glob("*.csv")):
        if csv_file.name in ("train.csv", "test.csv"):
            continue
        try:
            tdf = _safe_read_csv(csv_file)
            tdf.columns = [c.strip().strip("\ufeff") for c in tdf.columns]
            extra_tables[csv_file.stem] = tdf
            print(f"  [DataAnalyst] extra: {csv_file.stem} {tdf.shape}")
        except Exception as e:
            print(f"  [DataAnalyst] skip {csv_file.name}: {e}")

    # Читаем readme
    readme_text = ""
    for name in ("readme.txt", "README.txt", "readme.md", "README.md"):
        p = DATA_DIR / name
        if p.exists():
            try:
                readme_text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                readme_text = p.read_text(encoding="cp1251")
            break

    # Определяем id и target
    train_cols = list(df_train.columns)
    test_cols = list(df_test.columns)

    id_column = _find_id_column(df_train, df_test, readme_text)
    target_column = _find_target_column(df_train, df_test, id_column, readme_text)
    print(f"  [DataAnalyst] id_column={id_column}, target_column={target_column}")

    # Обновляем test_cols после возможных переименований в _find_id_column
    test_cols = list(df_test.columns)

    # ===== ВАЛИДАЦИЯ: гарантируем корректность или падаем с ясной ошибкой =====
    _validate_columns(df_train, df_test, id_column, target_column, extra_tables)

    test_has_features = any(
        c not in (id_column, target_column) for c in test_cols
    )

    feature_cols = [c for c in train_cols if c not in (id_column, target_column)]

    # Зарезервированные имена (чтобы не было коллизий)
    reserved_names = set(train_cols + test_cols)
    for tdf in extra_tables.values():
        reserved_names.update(tdf.columns.tolist())

    # Предагрегация тяжёлых таблиц (>100K строк)
    MAX_RAW_ROWS = 100_000
    new_tables = {}
    for tname, tdf in extra_tables.items():
        if len(tdf) <= MAX_RAW_ROWS:
            continue
        join_keys = [c for c in tdf.columns if c in train_cols or c in test_cols]
        if not join_keys:
            continue
        numeric_cols = [c for c in tdf.select_dtypes(include="number").columns
                        if c not in join_keys and not c.startswith("Unnamed")][:4]
        if not numeric_cols:
            continue
        for key in join_keys[:2]:
            agg_dict = {f"{tname}_{col}_mean": (col, "mean") for col in numeric_cols}
            agg_dict[f"{tname}_count"] = (numeric_cols[0], "count")
            try:
                agg_df = tdf.groupby(key).agg(**agg_dict).reset_index()
                new_tables[f"{tname}_by_{key}"] = agg_df
                print(f"  [DataAnalyst] pre-aggregated: {tname}_by_{key} {agg_df.shape}")
            except Exception:
                pass
    extra_tables.update(new_tables)

    # Формируем схему доп. таблиц
    extra_tables_schema = {}
    for tname, tdf in extra_tables.items():
        extra_tables_schema[tname] = {
            "columns": {c: str(tdf[c].dtype) for c in tdf.columns
                        if not c.startswith("Unnamed")},
            "shape": list(tdf.shape),
            "join_keys": [c for c in tdf.columns if c in train_cols or c in test_cols],
        }

    # Базовая статистика для контекста LLM
    stat_sources = []
    for c in feature_cols:
        if df_train[c].dtype in ("int64", "float64"):
            stat_sources.append((c, df_train[c], "train"))
    for tname, tdf in extra_tables.items():
        if len(tdf) > MAX_RAW_ROWS:
            continue
        if id_column in tdf.columns and tdf[id_column].nunique() == len(tdf):
            for c in tdf.select_dtypes(include="number").columns:
                if c != id_column and not c.startswith("Unnamed"):
                    stat_sources.append((f"{tname}.{c}", tdf[c], tname))

    basic_stats = {}
    for label, series, source in stat_sources[:20]:
        basic_stats[label] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "nunique": int(series.nunique()),
            "source": source,
        }
        if source == "train":
            try:
                basic_stats[label]["corr_target"] = round(
                    float(series.corr(df_train[target_column])), 4
                )
            except Exception:
                pass

    sample_rows = df_train.head(3).to_dict(orient="records")

    schema_info = {
        "target_column": target_column,
        "id_column": id_column,
        "readme_text": readme_text,
        "train_shape": list(df_train.shape),
        "test_shape": list(df_test.shape),
        "column_dtypes": {c: str(df_train[c].dtype) for c in feature_cols},
        "null_percentages": {c: round(float(df_train[c].isna().mean() * 100), 1)
                             for c in feature_cols},
        "basic_stats": basic_stats,
        "sample_rows": sample_rows,
        "extra_table_names": list(extra_tables.keys()),
        "extra_tables_schema": extra_tables_schema,
        "reserved_names": list(reserved_names),
        "test_has_features": test_has_features,
    }

    print(f"  [DataAnalyst] Schema built: {len(feature_cols)} feature cols, "
          f"{len(extra_tables)} extra tables")

    return {
        "schema_info": schema_info,
        "df_train": df_train,
        "df_test": df_test,
        "extra_tables": extra_tables,
    }
