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

    # Key columns in train/test that can be used for joins and aggregations
    key_cols = [c for c in df_train.columns if c not in (id_column, target_column)]

    # --- 1. Frequency encoding for all categorical/low-cardinality columns in train ---
    for col in key_cols:
        nunique = df_train[col].nunique()
        if nunique < len(df_train) * 0.5:  # categorical-like
            feat_name = f"fb_{col}_freq"
            freq = df_train[col].value_counts(normalize=True)
            train_out[feat_name] = df_train[col].map(freq).fillna(0).values
            test_out[feat_name] = df_test[col].map(freq).fillna(0).values if col in df_test.columns else 0
            features_added.append(feat_name)

    # --- 2. Merge numeric features from extra tables via shared keys ---
    for tname, tdf in extra_tables.items():
        if len(tdf) > 500_000:
            continue  # skip raw heavy tables, use pre-aggregated only

        # Find join keys: columns shared between this table and train
        shared_keys = [c for c in tdf.columns if c in df_train.columns and c not in (id_column, target_column)]
        if not shared_keys:
            continue

        join_key = shared_keys[0]  # use first shared key
        numeric_cols = tdf.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in shared_keys and c != id_column]

        for col in numeric_cols[:3]:  # max 3 features per extra table
            feat_name = f"fb_{tname}_{col}"
            if feat_name in features_added:
                continue
            mapping = tdf.groupby(join_key)[col].mean()
            train_out[feat_name] = df_train[join_key].map(mapping).fillna(0).values
            if join_key in df_test.columns:
                test_out[feat_name] = df_test[join_key].map(mapping).fillna(0).values
            else:
                test_out[feat_name] = 0
            features_added.append(feat_name)

    # --- 3. Cross-key aggregations: for each pair of key columns, count from extra tables ---
    for tname, tdf in extra_tables.items():
        if len(tdf) > 500_000:
            continue

        shared_keys = [c for c in tdf.columns if c in df_train.columns and c not in (id_column, target_column)]
        if len(shared_keys) >= 2:
            k1, k2 = shared_keys[0], shared_keys[1]
            pair_counts = tdf.groupby([k1, k2]).size().reset_index(name="fb_pair_count")
            feat_name = f"fb_{tname}_{k1}_{k2}_count"

            for df_src, df_out, label in [(df_train, train_out, "train"), (df_test, test_out, "test")]:
                if k1 in df_src.columns and k2 in df_src.columns:
                    merged = df_src[[k1, k2]].merge(pair_counts, on=[k1, k2], how="left")
                    df_out[feat_name] = merged["fb_pair_count"].fillna(0).values

            features_added.append(feat_name)

    # --- 4. Target-mean encoding per key (only from train, apply to both) ---
    for col in key_cols:
        nunique = df_train[col].nunique()
        if 2 < nunique < len(df_train) * 0.3:
            feat_name = f"fb_{col}_target_mean"
            global_mean = df_train[target_column].mean()
            # Smoothed target encoding to reduce overfitting
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
