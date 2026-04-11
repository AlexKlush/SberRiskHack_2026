"""Hardcoded fallback features — guaranteed baseline when LLM fails."""
import pandas as pd
import numpy as np


def generate_fallback_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    extra_tables: dict,
    schema_info: dict,
) -> tuple:
    """Generate features from extra tables without LLM. Returns (df_train_out, df_test_out)."""
    id_column = schema_info["id_column"]
    target_column = schema_info["target_column"]

    train_out = df_train[[id_column, target_column]].copy()
    test_out = df_test[[id_column]].copy()

    features_added = []

    def _add_feature(name, train_series, test_series):
        train_out[name] = train_series.values
        test_out[name] = test_series.values
        features_added.append(name)

    # --- 1. User features from users.csv ---
    if "users" in extra_tables and "user_id" in df_train.columns:
        users = extra_tables["users"]
        if "user_id" in users.columns:
            useful_cols = ["total_orders", "reordered_share", "total_distinct_products",
                           "avg_days_between_orders", "avg_basket_size"]
            for col in useful_cols:
                if col in users.columns:
                    mapping = users.set_index("user_id")[col]
                    _add_feature(
                        f"user_{col}",
                        df_train["user_id"].map(mapping).fillna(0),
                        df_test["user_id"].map(mapping).fillna(0),
                    )

    # --- 2. Product reorder stats from order_items ---
    if "order_items" in extra_tables and "product_id" in df_train.columns:
        oi = extra_tables["order_items"]
        if "product_id" in oi.columns and "reordered" in oi.columns:
            prod_reorder = oi.groupby("product_id")["reordered"].agg(["mean", "count"])
            prod_reorder.columns = ["fb_prod_reorder_rate", "fb_prod_order_count"]
            for col in prod_reorder.columns:
                mapping = prod_reorder[col]
                _add_feature(
                    col,
                    df_train["product_id"].map(mapping).fillna(0),
                    df_test["product_id"].map(mapping).fillna(0),
                )

    # --- 3. User-product interaction from order_items ---
    if "order_items" in extra_tables and "orders" in extra_tables:
        oi = extra_tables["order_items"]
        orders = extra_tables["orders"]
        if "order_id" in oi.columns and "order_id" in orders.columns and "user_id" in orders.columns:
            oi_with_user = oi.merge(orders[["order_id", "user_id"]], on="order_id", how="left")

            if "product_id" in oi_with_user.columns:
                # How many times this user bought this product before
                user_prod_counts = oi_with_user.groupby(["user_id", "product_id"]).size().reset_index(name="fb_user_prod_count")
                for df, label in [(df_train, "train"), (df_test, "test")]:
                    merged = df[["user_id", "product_id"]].merge(
                        user_prod_counts, on=["user_id", "product_id"], how="left"
                    )
                    if label == "train":
                        train_out["fb_user_prod_count"] = merged["fb_user_prod_count"].fillna(0).values
                    else:
                        test_out["fb_user_prod_count"] = merged["fb_user_prod_count"].fillna(0).values
                features_added.append("fb_user_prod_count")

                # How many distinct products this user bought / product popularity ratio
                user_total = oi_with_user.groupby("user_id")["product_id"].nunique()
                prod_total = oi_with_user.groupby("product_id")["user_id"].nunique()

                _add_feature(
                    "fb_user_nunique_products",
                    df_train["user_id"].map(user_total).fillna(0),
                    df_test["user_id"].map(user_total).fillna(0),
                )
                _add_feature(
                    "fb_prod_nunique_users",
                    df_train["product_id"].map(prod_total).fillna(0),
                    df_test["product_id"].map(prod_total).fillna(0),
                )

    # --- 4. Product category features ---
    if "products" in extra_tables and "product_id" in df_train.columns:
        products = extra_tables["products"]
        if "product_id" in products.columns and "aisle_id" in products.columns:
            mapping = products.set_index("product_id")["aisle_id"]
            _add_feature(
                "fb_prod_aisle_id",
                df_train["product_id"].map(mapping).fillna(0),
                df_test["product_id"].map(mapping).fillna(0),
            )

    # --- 5. Frequency encoding ---
    for col in ["user_id", "product_id"]:
        if col in df_train.columns:
            freq = df_train[col].value_counts(normalize=True)
            _add_feature(
                f"fb_{col}_freq",
                df_train[col].map(freq).fillna(0),
                df_test[col].map(freq).fillna(0),
            )

    # Deduplicate and limit to available columns
    feature_cols = [c for c in features_added if c in train_out.columns and c in test_out.columns]
    train_out = train_out[[id_column, target_column] + feature_cols]
    test_out = test_out[[id_column] + feature_cols]

    train_out[feature_cols] = train_out[feature_cols].fillna(0)
    test_out[feature_cols] = test_out[feature_cols].fillna(0)

    return train_out, test_out
