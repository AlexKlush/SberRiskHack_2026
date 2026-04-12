"""Общее состояние агентов."""
from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    schema_info: dict
    df_train: Optional[Any]
    df_test: Optional[Any]
    extra_tables: dict
    # FeatureEngineer формирует кандидатов
    candidate_features_train: Optional[Any]  # DataFrame: id + target + кандидаты
    candidate_features_test: Optional[Any]   # DataFrame: id + кандидаты
    candidate_names: list                     # имена кандидатов
    selected_features: list                   # итоговые фичи (до 5)
    cv_score: float
    errors_log: list
