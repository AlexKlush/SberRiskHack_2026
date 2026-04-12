"""EvaluatorWriter agent — forward-selects top 5 features, evaluates, saves output."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.state import AgentState
from src.utils.scoring import evaluate_features, forward_select_features

OUTPUT_DIR = Path("output")


def run(state: AgentState) -> dict:
    schema = state["schema_info"]
    id_col = schema["id_column"]
    target_col = schema["target_column"]

    train_df = state["candidate_features_train"]
    test_df = state["candidate_features_test"]
    candidates = list(state["candidate_names"])

    # --- Sanity: if no candidates at all, save bare minimum ---
    if not candidates or train_df is None or test_df is None:
        state["errors_log"].append("CRITICAL: no candidate features")
        train_out = state["df_train"][[id_col, target_col]].copy()
        test_out = state["df_test"][[id_col]].copy()
        train_out["fb_constant"] = 0
        test_out["fb_constant"] = 0
        _save(train_out, test_out, ["fb_constant"], 0.0)
        return {"selected_features": ["fb_constant"], "cv_score": 0.0}

    # Fill any remaining NaN / inf
    train_df[candidates] = train_df[candidates].fillna(0)
    test_df[candidates] = test_df[candidates].fillna(0)
    for c in candidates:
        train_df[c] = np.nan_to_num(train_df[c], nan=0.0, posinf=0.0, neginf=0.0)
        test_df[c] = np.nan_to_num(test_df[c], nan=0.0, posinf=0.0, neginf=0.0)

    # --- Forward selection: pick top 5 ---
    selected = forward_select_features(
        train_df, candidates, target_col, max_features=5,
    )

    # Fallback: if forward selection returns nothing, take first 5 candidates
    if not selected:
        state["errors_log"].append("Forward selection returned 0 features, taking first 5")
        selected = candidates[:5]

    # --- Final 5-fold evaluation ---
    final_score = evaluate_features(train_df, selected, target_col)

    # --- Build and save output ---
    train_out = train_df[[id_col, target_col] + selected].copy()
    test_out = test_df[[id_col] + selected].copy()
    train_out[selected] = train_out[selected].fillna(0)
    test_out[selected] = test_out[selected].fillna(0)

    _save(train_out, test_out, selected, final_score)

    return {"selected_features": selected, "cv_score": final_score}


def _save(train_out, test_out, features, score):
    """Write CSVs and print summary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_out.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print(f"\n{'=' * 50}")
    print(f"  ROC-AUC:      {score:.4f}")
    print(f"  Features ({len(features)}): {features}")
    print(f"  Train shape:  {train_out.shape}")
    print(f"  Test shape:   {test_out.shape}")
    print(f"  Saved to:     {OUTPUT_DIR / 'train.csv'}, {OUTPUT_DIR / 'test.csv'}")
    print(f"{'=' * 50}\n")
