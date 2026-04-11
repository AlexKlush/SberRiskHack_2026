"""OutputWriter agent — saves final train.csv and test.csv."""
from pathlib import Path

import pandas as pd

from src.state import AgentState
from src.utils.fallback_features import generate_fallback_features

OUTPUT_DIR = Path("output")


def run(state: AgentState) -> dict:
    best_idx = state["best_set_idx"]
    df_train_out = None
    df_test_out = None

    if state["computed_train_dfs"]:
        df_train_out = state["computed_train_dfs"][best_idx]
        df_test_out = state["computed_test_dfs"][best_idx]

    # Last-resort fallback: generate features directly if everything else failed
    if df_train_out is None or df_test_out is None:
        try:
            df_train_out, df_test_out = generate_fallback_features(
                state["df_train"], state["df_test"], state["extra_tables"], state["schema_info"]
            )
        except Exception as e:
            raise RuntimeError(
                f"All feature generation failed including fallback: {e}"
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

    # Final score is the last element (mixed/selected score from evaluator)
    cv_scores = state["cv_scores"]
    cv_score = cv_scores[-1] if cv_scores else 0.0
    print(f"\n{'='*50}")
    print(f"  ROC-AUC:      {cv_score:.4f}")
    print(f"  Features:     {feature_cols}")
    print(f"  Train shape:  {train_out.shape}")
    print(f"  Test shape:   {test_out.shape}")
    print(f"  Saved to:     {OUTPUT_DIR / 'train.csv'}, {OUTPUT_DIR / 'test.csv'}")
    print(f"{'='*50}\n")

    return {}
