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

    # Determine target_column: columns in train but not in test, minus id_column
    target_candidates = [c for c in train_cols if c not in test_cols and c != id_column]
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

    schema_info = {
        "target_column": target_column,
        "id_column": id_column,
        "readme_text": readme_text,
        "train_shape": list(df_train.shape),
        "test_shape": list(df_test.shape),
        "column_dtypes": {c: str(df_train[c].dtype) for c in feature_cols},
        "null_percentages": {c: float(df_train[c].isna().mean() * 100) for c in feature_cols},
        "extra_table_names": list(extra_tables.keys()),
        "reserved_names": reserved_names,
        "test_has_features": test_has_features,
    }

    return {
        "schema_info": schema_info,
        "df_train": df_train,
        "df_test": df_test,
        "extra_tables": extra_tables,
    }
