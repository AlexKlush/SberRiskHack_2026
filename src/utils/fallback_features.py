"""Generic fallback features — dataset-agnostic baseline when LLM fails."""
import pandas as pd
import numpy as np


def generate_fallback_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    extra_tables: dict,
    schema_info: dict,
) -> tuple:
    """Generate generic features from any dataset structure. No hardcoded table/column names."""
    id_column = schema_info["id_column"]
    target_column = schema_info["target_column"]

    train_out = df_train[[id_column, target_column]].copy()
    test_out = df_test[[id_column]].copy()
    features_added = []

    # Key columns in train/test (excluding id and target)
    key_cols = [c for c in df_train.columns if c not in (id_column, target_column)]

    # --- 1. Frequency encoding for categorical/low-cardinality columns in train ---
    for col in key_cols:
        nunique = df_train[col].nunique()
        if nunique < len(df_train) * 0.5:
            feat_name = f"fb_{col}_freq"
            freq = df_train[col].value_counts(normalize=True)
            train_out[feat_name] = df_train[col].map(freq).fillna(0).values
            test_out[feat_name] = df_test[col].map(freq).fillna(0).values if col in df_test.columns else 0
            features_added.append(feat_name)

    # --- 2. Merge features from extra tables ---
    # IMPORTANT: use id_column as join key too — some datasets have all features in extra tables
    for tname, tdf in extra_tables.items():
        if len(tdf) > 500_000:
            continue

        # Find join keys: columns shared between this table and train (INCLUDING id_column)
        shared_keys = [c for c in tdf.columns if c in df_train.columns and c != target_column]
        if not shared_keys:
            continue

        join_key = shared_keys[0]

        # Get numeric columns from extra table (exclude join keys and garbage columns)
        numeric_cols = tdf.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in shared_keys
                        and c != target_column
                        and not c.startswith("Unnamed")]

        # Get categorical columns for encoding
        cat_cols = tdf.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in shared_keys and c != target_column]

        # Numeric features: direct merge if 1-to-1, otherwise aggregate
        is_unique_key = tdf[join_key].nunique() == len(tdf)

        if is_unique_key:
            # 1-to-1 mapping: merge directly (e.g. client_data keyed by client_id)
            merge_cols = numeric_cols[:8]
            if merge_cols:
                subset = tdf[[join_key] + merge_cols].copy()
                rename_map = {c: f"fb_{tname}_{c}" for c in merge_cols}
                subset = subset.rename(columns=rename_map)
                train_out = train_out.merge(subset, on=join_key, how="left")
                test_out = test_out.merge(subset, on=join_key, how="left")
                features_added.extend(rename_map.values())

            # Categorical features: label encode
            for col in cat_cols[:5]:
                feat_name = f"fb_{tname}_{col}_enc"
                label_map = {v: i for i, v in enumerate(tdf[col].dropna().unique())}
                encoded = tdf.set_index(join_key)[col].map(label_map)
                train_out[feat_name] = df_train[join_key].map(encoded).fillna(-1).values
                test_out[feat_name] = df_test[join_key].map(encoded).fillna(-1).values
                features_added.append(feat_name)
        else:
            # Many-to-1: aggregate
            for col in numeric_cols[:3]:
                feat_name = f"fb_{tname}_{col}_mean"
                mapping = tdf.groupby(join_key)[col].mean()
                train_out[feat_name] = df_train[join_key].map(mapping).fillna(0).values
                if join_key in df_test.columns:
                    test_out[feat_name] = df_test[join_key].map(mapping).fillna(0).values
                else:
                    test_out[feat_name] = 0
                features_added.append(feat_name)

            feat_name = f"fb_{tname}_count"
            counts = tdf.groupby(join_key).size()
            train_out[feat_name] = df_train[join_key].map(counts).fillna(0).values
            if join_key in df_test.columns:
                test_out[feat_name] = df_test[join_key].map(counts).fillna(0).values
            else:
                test_out[feat_name] = 0
            features_added.append(feat_name)

    # --- 3. Target-mean encoding per key columns (smoothed) ---
    # Include id_column if it's a reasonable grouping key
    target_enc_cols = list(key_cols)
    id_nunique = df_train[id_column].nunique()
    if id_nunique < len(df_train) * 0.8 and id_column not in target_enc_cols:
        target_enc_cols.append(id_column)

    global_mean = df_train[target_column].mean()
    for col in target_enc_cols:
        nunique = df_train[col].nunique()
        if 2 < nunique < len(df_train) * 0.3:
            feat_name = f"fb_{col}_target_mean"
            stats = df_train.groupby(col)[target_column].agg(["mean", "count"])
            smoothing = 20
            stats["smoothed"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
            mapping = stats["smoothed"]
            train_out[feat_name] = df_train[col].map(mapping).fillna(global_mean).values
            if col in df_test.columns:
                test_out[feat_name] = df_test[col].map(mapping).fillna(global_mean).values
            else:
                test_out[feat_name] = global_mean
            features_added.append(feat_name)

    # Deduplicate and keep only valid features
    feature_cols = [c for c in dict.fromkeys(features_added) if c in train_out.columns and c in test_out.columns]
    train_out = train_out[[id_column, target_column] + feature_cols]
    test_out = test_out[[id_column] + feature_cols]

    train_out[feature_cols] = train_out[feature_cols].fillna(0)
    test_out[feature_cols] = test_out[feature_cols].fillna(0)

    return train_out, test_out
