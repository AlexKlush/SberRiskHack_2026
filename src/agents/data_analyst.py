"""DataAnalyst agent — reads data files and builds schema info."""
from pathlib import Path

import pandas as pd

from src.state import AgentState

DATA_DIR = Path("data")


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

    # Determine id_column: shared column between train and test that looks like an identifier
    train_cols = list(df_train.columns)
    test_cols = list(df_test.columns)
    common_cols = [c for c in train_cols if c in test_cols]

    id_column = None
    for c in common_cols:
        if "id" in c.lower():
            id_column = c
            break
    if id_column is None and common_cols:
        if train_cols[0] == test_cols[0]:
            id_column = train_cols[0]
        else:
            id_column = common_cols[0]

    # Determine target_column:
    # Strategy 1: columns in train but not in test
    target_candidates = [c for c in train_cols if c not in test_cols and c != id_column]
    # Strategy 2: if all columns are shared, look for common target names
    if not target_candidates:
        target_names = ["target", "label", "y", "class", "is_fraud", "default", "churn"]
        for name in target_names:
            if name in train_cols and name != id_column:
                target_candidates = [name]
                break
    # Strategy 3: fallback — check readme for clues about target
    if not target_candidates and readme_text:
        for col in train_cols:
            if col != id_column and ("целевая" in readme_text.lower() or "target" in readme_text.lower()):
                if col.lower() in readme_text.lower():
                    for line in readme_text.split("\n"):
                        if col.lower() in line.lower() and ("целев" in line.lower() or "target" in line.lower()):
                            target_candidates = [col]
                            break
                if target_candidates:
                    break
    if not target_candidates:
        raise ValueError("Cannot determine target column")

    target_column = target_candidates[0]
    if len(target_candidates) > 1:
        print(f"WARNING: multiple target candidates {target_candidates}, using '{target_column}'")

    # Check if test has features beyond id_column
    test_feature_cols = [c for c in test_cols if c != id_column]
    test_has_features = len(test_feature_cols) > 0
    if not test_has_features:
        print("WARNING: test.csv contains only ID column, no source features available")

    # Feature columns in train (excluding id and target)
    feature_cols = [c for c in train_cols if c not in (id_column, target_column)]

    # Reserved names: all existing column names across train and test
    reserved_names = list(set(train_cols + test_cols))

    # Pre-aggregate heavy tables (>100K rows) to speed up sandbox execution
    MAX_RAW_ROWS = 100_000
    for tname in list(extra_tables.keys()):
        tdf = extra_tables[tname]
        if len(tdf) > MAX_RAW_ROWS:
            join_keys = [c for c in tdf.columns if c in train_cols or c in test_cols]
            if join_keys:
                numeric_cols = tdf.select_dtypes(include="number").columns.tolist()
                numeric_cols = [c for c in numeric_cols if c not in join_keys]
                if numeric_cols:
                    agg_dict = {}
                    for col in numeric_cols:
                        agg_dict[f"{tname}_{col}_mean"] = (col, "mean")
                        agg_dict[f"{tname}_{col}_std"] = (col, "std")
                    agg_dict[f"{tname}_count"] = (numeric_cols[0], "count")
                    for key in join_keys:
                        agg_df = tdf.groupby(key).agg(**agg_dict).reset_index()
                        extra_tables[f"{tname}_by_{key}"] = agg_df
            # Keep original for fallback but note it's heavy
            # Don't remove — LLM code might need raw access

    # Build detailed schema for extra tables so LLM knows what's available to join
    extra_tables_schema = {}
    for tname, tdf in extra_tables.items():
        extra_tables_schema[tname] = {
            "columns": {c: str(tdf[c].dtype) for c in tdf.columns},
            "shape": list(tdf.shape),
            "join_keys": [c for c in tdf.columns if c in train_cols or c in test_cols],
        }

    # Compute basic stats and correlations with target for LLM context
    numeric_features = [c for c in feature_cols if df_train[c].dtype in ("int64", "float64")]
    basic_stats = {}
    for c in numeric_features[:10]:
        basic_stats[c] = {
            "mean": round(float(df_train[c].mean()), 4),
            "std": round(float(df_train[c].std()), 4),
            "nunique": int(df_train[c].nunique()),
        }
        try:
            basic_stats[c]["corr_target"] = round(float(df_train[c].corr(df_train[target_column])), 4)
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
