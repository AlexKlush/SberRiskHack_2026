"""FeatureEvaluator agent — merges all feature sets and picks best via forward selection."""
import json

import pandas as pd
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState
from src.utils.scoring import evaluate_features, forward_select_features
from src.utils.fallback_features import generate_fallback_features


def run(state: AgentState) -> dict:
    schema = state["schema_info"]
    id_column = schema["id_column"]
    target_column = schema["target_column"]

    # --- Phase 1: Merge all valid feature sets into one combined DataFrame ---
    combined_train = state["df_train"][[id_column, target_column]].copy()
    combined_test = state["df_test"][[id_column]].copy()
    all_feature_cols = []
    per_set_scores = []

    for idx, (df_train_out, df_test_out) in enumerate(
        zip(state["computed_train_dfs"], state["computed_test_dfs"])
    ):
        if df_train_out is None or df_test_out is None:
            per_set_scores.append(0.0)
            continue

        feature_cols = [c for c in df_train_out.columns if c not in (id_column, target_column)]

        # Deduplicate column names across sets by adding suffix
        rename_map = {}
        for col in feature_cols:
            if col in all_feature_cols:
                new_name = f"{col}_s{idx + 1}"
                rename_map[col] = new_name
            else:
                rename_map[col] = col

        df_train_renamed = df_train_out[[id_column] + feature_cols].rename(columns=rename_map)
        df_test_renamed = df_test_out[[id_column] + feature_cols].rename(columns=rename_map)

        combined_train = combined_train.merge(df_train_renamed, on=id_column, how="left")
        combined_test = combined_test.merge(df_test_renamed, on=id_column, how="left")
        all_feature_cols.extend(rename_map.values())

        # Score individual set for logging
        try:
            score = evaluate_features(df_train_out, feature_cols, target_column)
            per_set_scores.append(score)
        except Exception as e:
            state["errors_log"].append(f"FeatureEvaluator set {idx + 1} scoring: {e}")
            per_set_scores.append(0.0)

    # --- Phase 1b: Add fallback features to the candidate pool ---
    try:
        fb_train, fb_test = generate_fallback_features(
            state["df_train"], state["df_test"], state["extra_tables"], schema
        )
        fb_cols = [c for c in fb_train.columns if c not in (id_column, target_column)]
        if fb_cols:
            rename_fb = {}
            for col in fb_cols:
                if col in all_feature_cols:
                    rename_fb[col] = f"{col}_fb"
                else:
                    rename_fb[col] = col
            fb_train_r = fb_train[[id_column] + fb_cols].rename(columns=rename_fb)
            fb_test_r = fb_test[[id_column] + fb_cols].rename(columns=rename_fb)
            combined_train = combined_train.merge(fb_train_r, on=id_column, how="left")
            combined_test = combined_test.merge(fb_test_r, on=id_column, how="left")
            all_feature_cols.extend(rename_fb.values())
            print(f"  [Evaluator] Fallback features added: {list(rename_fb.values())}")
    except Exception as e:
        state["errors_log"].append(f"FeatureEvaluator fallback: {e}")

    # Fill NaN in combined features
    combined_train[all_feature_cols] = combined_train[all_feature_cols].fillna(0)
    combined_test[all_feature_cols] = combined_test[all_feature_cols].fillna(0)

    # --- Phase 2: Forward selection from all candidates ---
    if all_feature_cols:
        selected_cols = forward_select_features(
            combined_train, all_feature_cols, target_column, max_features=5
        )
    else:
        selected_cols = []
        state["errors_log"].append("CRITICAL: no valid features from any set")

    if not selected_cols:
        state["errors_log"].append("CRITICAL: forward selection returned 0 features")
        return {
            "cv_scores": per_set_scores,
            "best_set_idx": 0,
        }

    # --- Phase 3: Build final DataFrames with selected features ---
    final_train = combined_train[[id_column, target_column] + selected_cols].copy()
    final_test = combined_test[[id_column] + selected_cols].copy()

    final_score = evaluate_features(final_train, selected_cols, target_column)

    # Store the mixed result back as the "winning" set
    # We overwrite computed_train_dfs[0] and computed_test_dfs[0] with the mixed result
    state["computed_train_dfs"][0] = final_train
    state["computed_test_dfs"][0] = final_test

    print(f"  [Evaluator] Per-set ROC-AUC: {[f'{s:.4f}' for s in per_set_scores]}")
    print(f"  [Evaluator] All candidates ({len(all_feature_cols)}): {all_feature_cols}")
    print(f"  [Evaluator] Selected ({len(selected_cols)}): {selected_cols}")
    print(f"  [Evaluator] Final ROC-AUC: {final_score:.4f}")

    return {
        "cv_scores": per_set_scores + [final_score],
        "best_set_idx": 0,  # always 0 because we put mixed result there
    }
