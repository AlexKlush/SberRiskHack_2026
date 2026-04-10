"""Shared agent state definition."""
from typing import TypedDict, Optional, Any

import pandas as pd


class AgentState(TypedDict):
    schema_info: dict
    df_train: Optional[Any]
    df_test: Optional[Any]
    extra_tables: dict
    feature_ideas: list
    generated_code: list
    computed_train_dfs: list
    computed_test_dfs: list
    cv_scores: list
    best_set_idx: int
    errors_log: list
