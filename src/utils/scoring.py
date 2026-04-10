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
