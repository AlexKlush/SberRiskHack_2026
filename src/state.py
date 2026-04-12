"""Shared agent state definition."""
from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    schema_info: dict
    df_train: Optional[Any]
    df_test: Optional[Any]
    extra_tables: dict
    # FeatureEngineer produces candidate features
    candidate_features_train: Optional[Any]  # DataFrame: id + target + candidates
    candidate_features_test: Optional[Any]   # DataFrame: id + candidates
    candidate_names: list                     # candidate column names
    selected_features: list                   # final top-5 feature names
    cv_score: float
    errors_log: list
