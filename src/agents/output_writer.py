"""OutputWriter agent — saves final train.csv and test.csv."""
from pathlib import Path

import pandas as pd

from src.state import AgentState

OUTPUT_DIR = Path("output")


def run(state: AgentState) -> dict:
    best_idx = state["best_set_idx"]
    df_train_out = state["computed_train_dfs"][best_idx]
    df_test_out = state["computed_test_dfs"][best_idx]

    if df_train_out is None or df_test_out is None:
        raise RuntimeError(
            f"Best feature set {best_idx + 1} is None — "
            "all code generation attempts failed. Cannot produce output."
        )

    id_column = state["schema_info"]["id_column"]
    target_column = state["schema_info"]["target_column"]

    feature_cols = [c for c in df_train_out.columns if c not in (id_column, target_column)][:5]

    train_out = df_train_out[[id_column, target_column] + feature_cols].copy()
    test_out = df_test_out[[id_column] + feature_cols].copy()

    train_out[feature_cols] = train_out[feature_cols].fillna(0)
    test_out[feature_cols] = test_out[feature_cols].fillna(0)

    assert list(train_out.columns[2:]) == list(test_out.columns[1:]), \
        "Feature columns mismatch between train and test"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_out.to_csv(OUTPUT_DIR / "test.csv", index=False)

    cv_score = state["cv_scores"][best_idx]
    print(f"\n{'='*50}")
    print(f"  Selected set: {best_idx + 1}")
    print(f"  ROC-AUC:      {cv_score:.4f}")
    print(f"  Features:     {feature_cols}")
    print(f"  Train shape:  {train_out.shape}")
    print(f"  Test shape:   {test_out.shape}")
    print(f"  Saved to:     {OUTPUT_DIR / 'train.csv'}, {OUTPUT_DIR / 'test.csv'}")
    print(f"{'='*50}\n")

    return {}
