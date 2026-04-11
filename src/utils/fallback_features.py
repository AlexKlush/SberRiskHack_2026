"""Hardcoded fallback features — guaranteed baseline when LLM fails."""
import pandas as pd
import numpy as np


def generate_fallback_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    extra_tables: dict,
    schema_info: dict,
) -> tuple:
    """Generate basic features from extra tables without LLM. Returns (df_train_out, df_test_out)."""
    id_column = schema_info["id_column"]
    target_column = schema_info["target_column"]

    train_out = df_train[[id_column, target_column]].copy()
    test_out = df_test[[id_column]].copy()

    features_added = []

    # --- Features from users table ---
    if "users" in extra_tables:
        users = extra_tables["users"]
        if "user_id" in df_train.columns and "user_id" in users.columns:
            user_cols = [c for c in users.columns if c != "user_id"]
            for col in user_cols[:5]:
                feat_name = f"user_{col}"
                mapping = users.set_index("user_id")[col]
                train_out[feat_name] = df_train["user_id"].map(mapping)
                test_out[feat_name] = df_test["user_id"].map(mapping)
                features_added.append(feat_name)

    # --- Features from order_items table ---
    if "order_items" in extra_tables and "product_id" in df_train.columns:
        oi = extra_tables["order_items"]
        if "product_id" in oi.columns:
            prod_stats = oi.groupby("product_id").agg(
                fb_prod_order_count=("order_id", "nunique") if "order_id" in oi.columns else ("product_id", "count"),
            )
            if "reordered" in oi.columns:
                prod_reorder = oi.groupby("product_id")["reordered"].mean()
                prod_stats["fb_prod_reorder_rate"] = prod_reorder

            for col in prod_stats.columns:
                train_out[col] = df_train["product_id"].map(prod_stats[col])
                test_out[col] = df_test["product_id"].map(prod_stats[col])
                features_added.append(col)

    # --- Features from orders table ---
    if "orders" in extra_tables and "user_id" in df_train.columns:
        orders = extra_tables["orders"]
        if "user_id" in orders.columns:
            user_order_stats = orders.groupby("user_id").agg(
                fb_user_order_count=("order_id", "nunique") if "order_id" in orders.columns else ("user_id", "count"),
            )
            if "days_since_prior_order" in orders.columns:
                user_days = orders.groupby("user_id")["days_since_prior_order"].mean()
                user_order_stats["fb_user_avg_days"] = user_days

            for col in user_order_stats.columns:
                if col not in features_added:
                    train_out[col] = df_train["user_id"].map(user_order_stats[col])
                    test_out[col] = df_test["user_id"].map(user_order_stats[col])
                    features_added.append(col)

    # --- Frequency encoding for key columns ---
    for col in ["user_id", "product_id"]:
        if col in df_train.columns:
            feat_name = f"fb_{col}_freq"
            freq = df_train[col].value_counts(normalize=True)
            train_out[feat_name] = df_train[col].map(freq)
            test_out[feat_name] = df_test[col].map(freq).fillna(0)
            features_added.append(feat_name)

    # Keep only top 5 features
    feature_cols = [c for c in features_added if c in train_out.columns][:5]
    train_out = train_out[[id_column, target_column] + feature_cols]
    test_out = test_out[[id_column] + feature_cols]

    train_out[feature_cols] = train_out[feature_cols].fillna(0)
    test_out[feature_cols] = test_out[feature_cols].fillna(0)

    return train_out, test_out
