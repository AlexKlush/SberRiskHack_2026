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


def _fast_evaluate(df_sample: pd.DataFrame, feature_cols: list, target_col: str) -> float:
    """Fast evaluation on a sample with 3-fold CV for selection purposes."""
    X = df_sample[feature_cols].fillna(0).values
    y = df_sample[target_col].values

    model = CatBoostClassifier(verbose=0, random_state=42, iterations=100)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return float(np.mean(scores))


def forward_select_features(
    df_train: pd.DataFrame,
    candidate_cols: list,
    target_col: str,
    max_features: int = 5,
) -> list:
    """Greedy forward selection on a subsample, then final scoring on full data."""
    MAX_SAMPLE = 50_000
    if len(df_train) > MAX_SAMPLE:
        df_sample = df_train.sample(n=MAX_SAMPLE, random_state=42)
    else:
        df_sample = df_train

    # Phase 1: score each feature individually, keep top 10
    individual_scores = []
    for col in candidate_cols:
        try:
            score = _fast_evaluate(df_sample, [col], target_col)
            individual_scores.append((col, score))
        except Exception:
            individual_scores.append((col, 0.0))

    individual_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [col for col, _ in individual_scores[:10]]

    print(f"  [Scoring] Individual scores: {[(c, f'{s:.4f}') for c, s in individual_scores]}")
    print(f"  [Scoring] Top candidates for forward selection: {top_candidates}")

    # Phase 2: greedy forward selection
    selected = []
    best_score = 0.0

    for step in range(min(max_features, len(top_candidates))):
        best_candidate = None
        best_candidate_score = -1.0  # accept any feature, even if no improvement

        for col in top_candidates:
            if col in selected:
                continue
            trial = selected + [col]
            try:
                score = _fast_evaluate(df_sample, trial, target_col)
            except Exception:
                continue

            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = col

        if best_candidate is None:
            break

        selected.append(best_candidate)
        best_score = best_candidate_score
        print(f"  [Scoring] +{best_candidate} -> ROC-AUC {best_score:.4f}")

    # Phase 3: if < max_features selected, fill from top individual scorers
    if len(selected) < max_features:
        for col, _ in individual_scores:
            if col not in selected:
                selected.append(col)
            if len(selected) >= max_features:
                break

    return selected
