"""CatBoost CV evaluation for feature quality."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_features(df_train: pd.DataFrame, feature_cols: list, target_col: str) -> float:
    X = df_train[feature_cols].fillna(0).values
    y = df_train[target_col].values

    model = CatBoostClassifier(verbose=0, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return float(np.mean(scores))


def forward_select_features(
    df_train: pd.DataFrame,
    candidate_cols: list,
    target_col: str,
    max_features: int = 5,
) -> list:
    """Greedy forward selection: add features one by one, keep only if ROC-AUC improves."""
    selected = []
    best_score = 0.0

    for _ in range(min(max_features, len(candidate_cols))):
        best_candidate = None
        best_candidate_score = best_score

        for col in candidate_cols:
            if col in selected:
                continue
            trial = selected + [col]
            try:
                score = evaluate_features(df_train, trial, target_col)
            except Exception:
                continue

            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = col

        if best_candidate is None:
            break

        selected.append(best_candidate)
        best_score = best_candidate_score

    return selected
