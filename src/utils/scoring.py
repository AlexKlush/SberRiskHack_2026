"""CatBoost CV evaluation for feature quality."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Match organizer scoring parameters exactly
CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 0,
    "thread_count": 1,
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
}


def evaluate_features(df_train: pd.DataFrame, feature_cols: list, target_col: str) -> float:
    X = df_train[feature_cols].fillna(0).values
    y = df_train[target_col].values

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return float(np.mean(scores))


def _fast_evaluate(df_sample: pd.DataFrame, feature_cols: list, target_col: str) -> float:
    """Fast evaluation on a sample with adaptive CV for selection purposes."""
    X = df_sample[feature_cols].fillna(0).values
    y = df_sample[target_col].values

    n_splits = 5 if len(df_sample) < 50_000 else 3
    fast_params = {**CATBOOST_PARAMS, "iterations": 100, "thread_count": -1}
    model = CatBoostClassifier(**fast_params)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

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

    # --- Pre-filter: rank by abs correlation with target, keep top 20 ---
    target_vals = df_sample[target_col].values.astype(float)
    corr_ranked = []
    for col in candidate_cols:
        vals = df_sample[col].fillna(0).values.astype(float)
        c = abs(np.corrcoef(vals, target_vals)[0, 1])
        corr_ranked.append((col, 0.0 if np.isnan(c) else c))
    corr_ranked.sort(key=lambda x: x[1], reverse=True)
    pre_filtered = [col for col, _ in corr_ranked[:20]]
    print(f"  [Scoring] Pre-filter: {len(candidate_cols)} -> {len(pre_filtered)} candidates (by correlation)")

    # --- Deduplicate: drop features >0.995 correlated with a higher-ranked one ---
    deduped = []
    for col in pre_filtered:
        vals = df_sample[col].fillna(0).values.astype(float)
        is_dup = False
        for existing in deduped:
            ev = df_sample[existing].fillna(0).values.astype(float)
            c = np.corrcoef(vals, ev)[0, 1]
            if not np.isnan(c) and abs(c) > 0.995:
                is_dup = True
                break
        if not is_dup:
            deduped.append(col)
    print(f"  [Scoring] After dedup: {len(deduped)} candidates")

    # --- Phase 1: CatBoost-score only deduped candidates ---
    individual_scores = []
    for col in deduped:
        try:
            score = _fast_evaluate(df_sample, [col], target_col)
            individual_scores.append((col, score))
        except Exception:
            individual_scores.append((col, 0.0))

    individual_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [col for col, _ in individual_scores[:7]]

    print(f"  [Scoring] Individual scores: {[(c, f'{s:.4f}') for c, s in individual_scores]}")
    print(f"  [Scoring] Top candidates for forward selection: {top_candidates}")

    # --- Phase 2: greedy forward selection ---
    # A feature is accepted unless it degrades the combined score by more than DEGRADE_THRESHOLD.
    DEGRADE_THRESHOLD = 0.001
    selected = []
    best_score = 0.0

    for step in range(min(max_features, len(top_candidates))):
        best_candidate = None
        best_candidate_score = -1.0

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

        # First feature always accepted; subsequent accepted unless they degrade score
        if selected and best_candidate_score < best_score - DEGRADE_THRESHOLD:
            print(f"  [Scoring] Skip {best_candidate} (score {best_candidate_score:.4f} degrades {best_score:.4f} by >{DEGRADE_THRESHOLD})")
            break

        selected.append(best_candidate)
        best_score = max(best_score, best_candidate_score)
        print(f"  [Scoring] +{best_candidate} -> ROC-AUC {best_candidate_score:.4f} (best so far: {best_score:.4f})")

    # Ensure at least 1 feature
    if not selected and individual_scores:
        selected = [individual_scores[0][0]]
        print(f"  [Scoring] Fallback: {selected[0]}")

    return selected
