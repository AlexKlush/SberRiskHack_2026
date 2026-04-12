"""Pre-built feature operations library.

Each operation is safe, deterministic, and dataset-agnostic.
Returns (feature_name, train_values, test_values) or (None, None, None) on failure.
No exec(), no sandbox — just reliable pandas/numpy.
"""
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Individual operations
# ---------------------------------------------------------------------------

def freq_encode(df_train, df_test, column, **kw):
    """Frequency encoding — proportion of each value in train."""
    if column not in df_train.columns:
        return None, None, None
    freq = df_train[column].value_counts(normalize=True)
    name = f"fe_{column}_freq"
    tr = df_train[column].map(freq).fillna(0).values
    te = df_test[column].map(freq).fillna(0).values if column in df_test.columns else np.zeros(len(df_test))
    return name, tr, te


def target_encode(df_train, df_test, column, target_col, **kw):
    """Smoothed target-mean encoding (no leakage — global mean as prior)."""
    if column not in df_train.columns:
        return None, None, None
    smoothing = max(20, min(200, len(df_train) // 100))
    gm = df_train[target_col].mean()
    stats = df_train.groupby(column)[target_col].agg(["mean", "count"])
    stats["s"] = (stats["count"] * stats["mean"] + smoothing * gm) / (stats["count"] + smoothing)
    mapping = stats["s"]
    name = f"fe_{column}_tmean"
    tr = df_train[column].map(mapping).fillna(gm).values
    te = df_test[column].map(mapping).fillna(gm).values if column in df_test.columns else np.full(len(df_test), gm)
    return name, tr, te


def agg_feature(df_train, df_test, extra_tables, table, key, column, func, **kw):
    """Aggregate a numeric column from an extra table grouped by key."""
    if table not in extra_tables:
        return None, None, None
    tdf = extra_tables[table]
    if key not in tdf.columns or column not in tdf.columns or key not in df_train.columns:
        return None, None, None

    agg_funcs = {
        "mean": lambda g: g.mean(),
        "std": lambda g: g.std().fillna(0),
        "sum": lambda g: g.sum(),
        "max": lambda g: g.max(),
        "min": lambda g: g.min(),
        "count": lambda g: g.count(),
        "nunique": lambda g: g.nunique(),
        "median": lambda g: g.median(),
    }
    if func not in agg_funcs:
        return None, None, None

    mapping = agg_funcs[func](tdf.groupby(key)[column])
    name = f"fe_{table}_{column}_{func}"
    tr = df_train[key].map(mapping).fillna(0).values
    te = df_test[key].map(mapping).fillna(0).values if key in df_test.columns else np.zeros(len(df_test))
    return name, tr, te


def count_feature(df_train, df_test, extra_tables, table, key, **kw):
    """Row count per key in an extra table."""
    if table not in extra_tables:
        return None, None, None
    tdf = extra_tables[table]
    if key not in tdf.columns or key not in df_train.columns:
        return None, None, None
    counts = tdf.groupby(key).size()
    name = f"fe_{table}_cnt_{key}"
    tr = df_train[key].map(counts).fillna(0).values
    te = df_test[key].map(counts).fillna(0).values if key in df_test.columns else np.zeros(len(df_test))
    return name, tr, te


def interaction(df_train, df_test, col1, op_type, col2, **kw):
    """Arithmetic interaction between two numeric columns."""
    if col1 not in df_train.columns or col2 not in df_train.columns:
        return None, None, None

    def _calc(df, c1, c2, op):
        a = pd.to_numeric(df[c1], errors="coerce").fillna(0)
        b = pd.to_numeric(df[c2], errors="coerce").fillna(0)
        if op == "mul":
            return a * b
        elif op == "div":
            return a / b.replace(0, np.nan).fillna(1)
        elif op == "add":
            return a + b
        elif op == "sub":
            return a - b
        return a

    name = f"fe_{col1}_{op_type}_{col2}"
    tr = _calc(df_train, col1, col2, op_type).values
    te = (_calc(df_test, col1, col2, op_type).values
          if col1 in df_test.columns and col2 in df_test.columns
          else np.zeros(len(df_test)))
    return name, tr, te


def rank_feature(df_train, df_test, column, **kw):
    """Percentile rank transform."""
    if column not in df_train.columns:
        return None, None, None
    if not pd.api.types.is_numeric_dtype(df_train[column]):
        return None, None, None
    name = f"fe_{column}_rank"
    tr = df_train[column].rank(pct=True).fillna(0.5).values
    te = df_test[column].rank(pct=True).fillna(0.5).values if column in df_test.columns else np.full(len(df_test), 0.5)
    return name, tr, te


def is_null(df_train, df_test, column, **kw):
    """Binary indicator: 1 if value is null, 0 otherwise."""
    if column not in df_train.columns:
        return None, None, None
    if df_train[column].isna().sum() == 0:
        return None, None, None
    name = f"fe_{column}_isnull"
    tr = df_train[column].isna().astype(int).values
    te = df_test[column].isna().astype(int).values if column in df_test.columns else np.zeros(len(df_test))
    return name, tr, te


def label_encode(df_train, df_test, column, **kw):
    """Integer label encoding for categorical columns."""
    if column not in df_train.columns:
        return None, None, None
    if pd.api.types.is_numeric_dtype(df_train[column]):
        return None, None, None
    label_map = {v: i for i, v in enumerate(df_train[column].dropna().unique())}
    name = f"fe_{column}_label"
    tr = df_train[column].map(label_map).fillna(-1).values
    te = df_test[column].map(label_map).fillna(-1).values if column in df_test.columns else np.full(len(df_test), -1)
    return name, tr, te


def direct_numeric(df_train, df_test, extra_tables, table, key, column, **kw):
    """Direct merge of a numeric column from a 1-to-1 extra table."""
    if table not in extra_tables:
        return None, None, None
    tdf = extra_tables[table]
    if key not in tdf.columns or column not in tdf.columns or key not in df_train.columns:
        return None, None, None
    if not pd.api.types.is_numeric_dtype(tdf[column]):
        return None, None, None
    # Must be 1-to-1 (unique key)
    if tdf[key].nunique() != len(tdf):
        return None, None, None
    mapping = tdf.set_index(key)[column]
    name = f"fe_{table}_{column}"
    tr = df_train[key].map(mapping).fillna(0).values
    te = df_test[key].map(mapping).fillna(0).values if key in df_test.columns else np.zeros(len(df_test))
    return name, tr, te


def ratio_to_group(df_train, df_test, extra_tables, column, table, key, ref_column, **kw):
    """Ratio of a value to the group mean (value / group_mean)."""
    if table not in extra_tables:
        return None, None, None
    tdf = extra_tables[table]
    if key not in tdf.columns or ref_column not in tdf.columns or key not in df_train.columns:
        return None, None, None
    if column not in df_train.columns:
        return None, None, None
    group_mean = tdf.groupby(key)[ref_column].mean()
    name = f"fe_{column}_ratio_{table}_{ref_column}"
    tr_vals = pd.to_numeric(df_train[column], errors="coerce").fillna(0)
    tr_group = df_train[key].map(group_mean).fillna(1)
    tr = (tr_vals / tr_group.replace(0, 1)).values
    if column in df_test.columns and key in df_test.columns:
        te_vals = pd.to_numeric(df_test[column], errors="coerce").fillna(0)
        te_group = df_test[key].map(group_mean).fillna(1)
        te = (te_vals / te_group.replace(0, 1)).values
    else:
        te = np.zeros(len(df_test))
    return name, tr, te


def extra_freq_encode(df_train, df_test, extra_tables, table, key, column, **kw):
    """Frequency encoding of a categorical column from an extra table."""
    if table not in extra_tables:
        return None, None, None
    tdf = extra_tables[table]
    if key not in tdf.columns or column not in tdf.columns or key not in df_train.columns:
        return None, None, None
    # Map key -> column value, then frequency encode
    mapping = tdf.drop_duplicates(key).set_index(key)[column]
    train_vals = df_train[key].map(mapping)
    freq = train_vals.value_counts(normalize=True)
    name = f"fe_{table}_{column}_freq"
    tr = train_vals.map(freq).fillna(0).values
    if key in df_test.columns:
        test_vals = df_test[key].map(mapping)
        te = test_vals.map(freq).fillna(0).values
    else:
        te = np.zeros(len(df_test))
    return name, tr, te


def extra_target_encode(df_train, df_test, extra_tables, table, key, column,
                        target_col, **kw):
    """Smoothed target encoding of a categorical column from an extra table."""
    if table not in extra_tables:
        return None, None, None
    tdf = extra_tables[table]
    if key not in tdf.columns or column not in tdf.columns or key not in df_train.columns:
        return None, None, None
    smoothing = max(20, min(200, len(df_train) // 100))
    # Map key -> column value
    mapping = tdf.drop_duplicates(key).set_index(key)[column]
    train_vals = df_train[key].map(mapping)
    gm = df_train[target_col].mean()
    stats = pd.DataFrame({"val": train_vals, "y": df_train[target_col].values})
    grp = stats.groupby("val")["y"].agg(["mean", "count"])
    grp["s"] = (grp["count"] * grp["mean"] + smoothing * gm) / (grp["count"] + smoothing)
    te_map = grp["s"]
    name = f"fe_{table}_{column}_tmean"
    tr = train_vals.map(te_map).fillna(gm).values
    if key in df_test.columns:
        test_vals = df_test[key].map(mapping)
        te = test_vals.map(te_map).fillna(gm).values
    else:
        te = np.full(len(df_test), gm)
    return name, tr, te


def cross_agg(df_train, df_test, extra_tables, table, keys, column, func, **kw):
    """Aggregate a column from an extra table grouped by composite key."""
    if table not in extra_tables:
        return None, None, None
    tdf = extra_tables[table]
    if not isinstance(keys, list) or len(keys) < 2:
        return None, None, None
    for k in keys:
        if k not in tdf.columns or k not in df_train.columns:
            return None, None, None
    if column not in tdf.columns:
        return None, None, None
    agg_funcs = {"mean": "mean", "std": "std", "sum": "sum", "max": "max",
                 "min": "min", "count": "count", "nunique": "nunique", "median": "median"}
    if func not in agg_funcs:
        return None, None, None
    grouped = tdf.groupby(keys)[column].agg(agg_funcs[func]).reset_index()
    grouped.columns = list(keys) + ["__val__"]
    name = f"fe_cross_{'_'.join(keys)}_{table}_{column}_{func}"
    tr = df_train[keys].merge(grouped, on=keys, how="left")["__val__"].fillna(0).values
    if len(tr) != len(df_train):
        return None, None, None
    test_keys_ok = all(k in df_test.columns for k in keys)
    if test_keys_ok:
        te = df_test[keys].merge(grouped, on=keys, how="left")["__val__"].fillna(0).values
        if len(te) != len(df_test):
            return None, None, None
    else:
        te = np.zeros(len(df_test))
    return name, tr, te


def extra_label_encode(df_train, df_test, extra_tables, table, key, column, **kw):
    """Label encoding of a categorical column from an extra table."""
    if table not in extra_tables:
        return None, None, None
    tdf = extra_tables[table]
    if key not in tdf.columns or column not in tdf.columns or key not in df_train.columns:
        return None, None, None
    mapping = tdf.drop_duplicates(key).set_index(key)[column]
    label_map = {v: i for i, v in enumerate(mapping.dropna().unique())}
    name = f"fe_{table}_{column}_label"
    tr = df_train[key].map(mapping).map(label_map).fillna(-1).values
    if key in df_test.columns:
        te = df_test[key].map(mapping).map(label_map).fillna(-1).values
    else:
        te = np.full(len(df_test), -1)
    return name, tr, te


# ---------------------------------------------------------------------------
# Operation registry
# ---------------------------------------------------------------------------

OPERATIONS = {
    "FREQ_ENCODE": freq_encode,
    "TARGET_ENCODE": target_encode,
    "AGG": agg_feature,
    "COUNT": count_feature,
    "INTERACTION": interaction,
    "RANK": rank_feature,
    "IS_NULL": is_null,
    "LABEL_ENCODE": label_encode,
    "DIRECT_NUMERIC": direct_numeric,
    "RATIO_TO_GROUP": ratio_to_group,
    "EXTRA_FREQ_ENCODE": extra_freq_encode,
    "EXTRA_TARGET_ENCODE": extra_target_encode,
    "EXTRA_LABEL_ENCODE": extra_label_encode,
    "CROSS_AGG": cross_agg,
}


def execute_operation(op_dict, df_train, df_test, extra_tables, target_col):
    """Execute one operation from the menu. Returns (name, tr, te) or None."""
    op_type = op_dict.get("op")
    if op_type not in OPERATIONS:
        return None
    fn = OPERATIONS[op_type]
    try:
        params = {k: v for k, v in op_dict.items() if k != "op"}
        result = fn(
            df_train=df_train,
            df_test=df_test,
            extra_tables=extra_tables,
            target_col=target_col,
            **params,
        )
        if result[0] is None:
            return None
        # Validate output
        name, tr, te = result
        if len(tr) != len(df_train) or len(te) != len(df_test):
            return None
        # Check for inf/nan
        tr = np.nan_to_num(tr, nan=0.0, posinf=0.0, neginf=0.0)
        te = np.nan_to_num(te, nan=0.0, posinf=0.0, neginf=0.0)
        return name, tr, te
    except Exception:
        return None
