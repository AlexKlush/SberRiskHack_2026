"""Агент DataAnalyst — читает CSV, определяет схему, формирует контекст для LLM."""
from pathlib import Path

import pandas as pd

from src.state import AgentState

DATA_DIR = Path("data")


def _find_id_column(df_train, df_test, readme_text):
    """Автоопределение столбца-идентификатора по нескольким стратегиям."""
    train_cols = list(df_train.columns)
    test_cols = list(df_test.columns)
    common_cols = [c for c in train_cols if c in test_cols]

    # Ищем столбец с "id" в названии
    id_keywords = ["_id", "id_", "row_id", "client_id", "sample_id", "index"]
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

    # Фоллбэк — первый общий или первый столбец train
    return common_cols[0] if common_cols else train_cols[0]


def _find_target_column(df_train, df_test, id_column, readme_text):
    """Автоопределение целевой переменной по нескольким стратегиям."""
    train_cols = list(df_train.columns)
    test_cols = list(df_test.columns)

    # Столбцы в train, которых нет в test — самый надёжный признак
    candidates = [c for c in train_cols if c not in test_cols and c != id_column]
    if candidates:
        return candidates[0]

    # Известные названия таргета
    target_names = ["target", "label", "y", "class", "is_fraud", "default",
                    "default_flag", "churn", "is_default", "fraud", "outcome",
                    "result", "flag"]
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
