"""DataAnalyst agent — reads data files and builds schema info."""
from pathlib import Path

import pandas as pd

from src.state import AgentState

DATA_DIR = Path("data")


def _find_id_column(df_train, df_test, readme_text):
    """Auto-detect the ID column using multiple strategies."""
    train_cols = list(df_train.columns)
    test_cols = list(df_test.columns)
    common_cols = [c for c in train_cols if c in test_cols]

    # Strategy 1: column with "id" in name (prefer exact matches like "id", "row_id", "client_id")
    id_keywords = ["_id", "id_", "row_id", "client_id", "sample_id", "index"]
    for c in common_cols:
        cl = c.lower()
        if cl == "id" or any(kw in cl for kw in id_keywords):
            return c

    # Strategy 2: check readme for hints about which column is the identifier
    if readme_text:
        for c in common_cols:
            for line in readme_text.split("\n"):
                cl = c.lower()
                ll = line.lower()
                if cl in ll and ("идентификатор" in ll or "identifier" in ll or "unique" in ll or "уникальн" in ll):
                    return c

    # Strategy 3: first common column that is unique in both train and test
    for c in common_cols:
        if df_train[c].nunique() == len(df_train) and df_test[c].nunique() == len(df_test):
            return c

    # Strategy 4: first common column, or first column of train
    if common_cols:
        return common_cols[0]
    return train_cols[0]


def _find_target_column(df_train, df_test, id_column, readme_text):
    """Auto-detect the target column using multiple strategies."""
    train_cols = list(df_train.columns)
    test_cols = list(df_test.columns)

    # Strategy 1: columns in train but not in test
    candidates = [c for c in train_cols if c not in test_cols and c != id_column]
    if candidates:
        return candidates[0]

    # Strategy 2: common target names
    target_names = ["target", "label", "y", "class", "is_fraud", "default", "churn",
                    "is_default", "fraud", "outcome", "result", "flag"]
    for name in target_names:
        if name in train_cols and name != id_column:
            return name

    # Strategy 3: binary column (only 0/1 values) that's not the id
    for c in train_cols:
        if c == id_column:
            continue
        if df_train[c].dtype in ("int64", "float64"):
            unique_vals = set(df_train[c].dropna().unique())
            if unique_vals <= {0, 1, 0.0, 1.0}:
                return c

    # Strategy 4: check readme for "целевая" or "target"
    if readme_text:
        for c in train_cols:
            if c == id_column:
                continue
            for line in readme_text.split("\n"):
                if c.lower() in line.lower() and ("целев" in line.lower() or "target" in line.lower()):
                    return c

    raise ValueError("Cannot determine target column")


def run(state: AgentState) -> dict:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    df_train = pd.read_csv(train_path, sep=None, engine="python")
    df_test = pd.read_csv(test_path, sep=None, engine="python")

    extra_tables = {}
    for csv_file in DATA_DIR.glob("*.csv"):
        if csv_file.name not in ("train.csv", "test.csv"):
            extra_tables[csv_file.stem] = pd.read_csv(csv_file, sep=None, engine="python")

    readme_text = ""
    readme_path = DATA_DIR / "readme.txt"
    if readme_path.exists():
        try:
            readme_text = readme_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            readme_text = readme_path.read_text(encoding="cp1251")

    train_cols = list(df_train.columns)
    test_cols = list(df_test.columns)

    id_column = _find_id_column(df_train, df_test, readme_text)
    target_column = _find_target_column(df_train, df_test, id_column, readme_text)

    # Check if test has features beyond id_column
    test_feature_cols = [c for c in test_cols if c not in (id_column, target_column)]
    test_has_features = len(test_feature_cols) > 0

    # Feature columns in train (excluding id and target)
    feature_cols = [c for c in train_cols if c not in (id_column, target_column)]

    # Reserved names: all existing column names across train, test, and extra tables
    reserved_names = list(set(train_cols + test_cols))
    for tdf in extra_tables.values():
        reserved_names.extend(tdf.columns.tolist())
    reserved_names = list(set(reserved_names))

    # Pre-aggregate heavy tables (>100K rows) by join keys
    MAX_RAW_ROWS = 100_000
    for tname in list(extra_tables.keys()):
        tdf = extra_tables[tname]
        if len(tdf) > MAX_RAW_ROWS:
            join_keys = [c for c in tdf.columns if c in train_cols or c in test_cols]
            if join_keys:
                numeric_cols = tdf.select_dtypes(include="number").columns.tolist()
                numeric_cols = [c for c in numeric_cols if c not in join_keys and not c.startswith("Unnamed")][:4]
                if numeric_cols:
                    for key in join_keys[:2]:
                        agg_dict = {f"{tname}_{col}_mean": (col, "mean") for col in numeric_cols}
                        agg_dict[f"{tname}_count"] = (numeric_cols[0], "count")
                        agg_df = tdf.groupby(key).agg(**agg_dict).reset_index()
                        extra_tables[f"{tname}_by_{key}"] = agg_df

    # Build detailed schema for extra tables
    extra_tables_schema = {}
    for tname, tdf in extra_tables.items():
        extra_tables_schema[tname] = {
            "columns": {c: str(tdf[c].dtype) for c in tdf.columns if not c.startswith("Unnamed")},
            "shape": list(tdf.shape),
            "join_keys": [c for c in tdf.columns if c in train_cols or c in test_cols],
        }

    # Compute basic stats for LLM context — from train AND from extra tables (1-to-1 with train)
    all_numeric_for_stats = []
    # Stats from train feature columns
    for c in feature_cols:
        if df_train[c].dtype in ("int64", "float64"):
            all_numeric_for_stats.append((c, df_train[c], "train"))
    # Stats from extra tables that join 1-to-1 with train via id_column
    for tname, tdf in extra_tables.items():
        if len(tdf) > MAX_RAW_ROWS:
            continue
        if id_column in tdf.columns and tdf[id_column].nunique() == len(tdf):
            for c in tdf.select_dtypes(include="number").columns:
                if c != id_column and not c.startswith("Unnamed"):
                    all_numeric_for_stats.append((f"{tname}.{c}", tdf[c], tname))

    basic_stats = {}
    for label, series, source in all_numeric_for_stats[:15]:
        basic_stats[label] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "nunique": int(series.nunique()),
            "source": source,
        }
        if source == "train":
            try:
                basic_stats[label]["corr_target"] = round(float(series.corr(df_train[target_column])), 4)
            except Exception:
                pass

    # Sample rows for LLM context
    sample_rows = df_train.head(3).to_dict(orient="records")

    schema_info = {
        "target_column": target_column,
        "id_column": id_column,
        "readme_text": readme_text,
        "train_shape": list(df_train.shape),
        "test_shape": list(df_test.shape),
        "column_dtypes": {c: str(df_train[c].dtype) for c in feature_cols},
        "null_percentages": {c: float(df_train[c].isna().mean() * 100) for c in feature_cols},
        "basic_stats": basic_stats,
        "sample_rows": sample_rows,
        "extra_table_names": list(extra_tables.keys()),
        "extra_tables_schema": extra_tables_schema,
        "reserved_names": reserved_names,
        "test_has_features": test_has_features,
    }

    return {
        "schema_info": schema_info,
        "df_train": df_train,
        "df_test": df_test,
        "extra_tables": extra_tables,
    }
