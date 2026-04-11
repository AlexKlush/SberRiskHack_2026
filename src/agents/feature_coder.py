"""FeatureCoder agent — generates and executes feature engineering code."""
import json

import numpy as np
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState

SYSTEM_PROMPT = """\
<role>
Ты — Python-разработчик, специалист по pandas. Пишешь чистый, \
воспроизводимый код для вычисления признаков на табличных данных. \
Держи код минимальным: только то, что нужно.
</role>
<constraints>
ОБЯЗАТЕЛЬНО:
- Вычисляй агрегации ТОЛЬКО на df_train (или на доп. таблицах), применяй к df_test через merge/map
- extra_tables — это dict[str, pd.DataFrame], доступ: extra_tables["table_name"]
- Обрабатывай NaN: fillna медианой train или 0
- Итоговые df_train_out и df_test_out должны содержать ТОЛЬКО нужные колонки
- Каждый признак должен быть числовым (int или float)
ЗАПРЕЩЕНО:
- Использовать target_column при вычислении признаков
- Вызывать .fit() на test данных
- Использовать pd.read_csv или открывать файлы — работай только с переданными df
- Использовать библиотеки кроме pandas и numpy
</constraints>"""

USER_PROMPT_TEMPLATE = """\
<context>
<schema>
id_column: {id_column}
target_column: {target_column}
train_test_columns: {columns_with_dtypes}
</schema>
<extra_tables>
{extra_tables_info}
</extra_tables>
<feature_ideas>
{feature_ideas_json}
</feature_ideas>
{test_warning}
</context>
<task>
Напиши ПОЛНУЮ Python-функцию generate_features со следующей сигнатурой.
Включай в ответ строку def и строку return — это обязательно.
def generate_features(df_train, df_test, extra_tables=None):
    import pandas as pd
    import numpy as np
    # extra_tables — dict[str, pd.DataFrame], например extra_tables["users"]
    # реализуй признаки из feature_ideas
    # ...
    return df_train_out, df_test_out
Требования к возвращаемым df:
- df_train_out колонки: [{id_column}, {target_column}, feature_1, ..., feature_N]
- df_test_out колонки:  [{id_column}, feature_1, ..., feature_N]
- Имена признаков должны совпадать в обоих df
- Все признаки — числовые, без NaN
ВАЖНО: вычисляй признаки одинаково для train и test. Агрегации считай по доп. таблицам
(они одинаковы для train и test) или по df_train, затем merge к обоим.
</task>
<output_format>
Верни ПОЛНУЮ функцию — включая строку def и return.
Без markdown-обёртки ```python```, без пояснений.
Первая строка ответа должна быть:
def generate_features(df_train, df_test, extra_tables=None):
</output_format>"""

FIX_PROMPT_TEMPLATE = """\
<error>
При выполнении кода возникла ошибка:
{error_message}
</error>
<previous_code>
{previous_code}
</previous_code>
<task>
Исправь ТОЛЬКО ошибку. Не меняй логику вычисления признаков.
Верни исправленную ПОЛНУЮ функцию (включая def-строку и return).
</task>"""


def _build_extra_tables_info(schema: dict) -> str:
    extra_schema = schema.get("extra_tables_schema", {})
    if not extra_schema:
        return "Нет дополнительных таблиц."

    parts = []
    for tname, tinfo in extra_schema.items():
        # Only show tables that will be in sandbox (<=100K rows or pre-aggregated)
        if tinfo["shape"][0] > 100_000:
            continue
        cols = tinfo["columns"]
        join_keys = tinfo["join_keys"]
        shape = tinfo["shape"]
        col_desc = ", ".join(f"{c} ({dtype})" for c, dtype in cols.items())
        parts.append(
            f"extra_tables[\"{tname}\"] [{shape[0]} rows x {shape[1]} cols]:\n"
            f"  Колонки: {col_desc}\n"
            f"  Join keys (общие с train/test): {join_keys}"
        )
    if not parts:
        return "Нет дополнительных таблиц доступных в sandbox."
    return "\n".join(parts)


def _extract_code(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def _execute_code(code_str: str, state: AgentState):
    # For heavy tables (>100K rows), only pass pre-aggregated versions to sandbox
    safe_tables = {}
    for k, v in state["extra_tables"].items():
        if len(v) <= 100_000:
            safe_tables[k] = v.copy()
        # pre-aggregated tables (e.g. "order_items_by_product_id") are always small
    sandbox = {
        "df_train": state["df_train"].copy(),
        "df_test": state["df_test"].copy(),
        "extra_tables": safe_tables,
    }
    full_code = code_str.strip() + "\n\nresult = generate_features(df_train, df_test, extra_tables)"
    exec(full_code, sandbox)
    return sandbox["result"]


def run(state: AgentState) -> dict:
    schema = state["schema_info"]
    llm = GigaChat(
        model="GigaChat-2-Max",
        temperature=0.1,
        max_tokens=2000,
        verify_ssl_certs=False,
        profanity_check=False,
        scope="GIGACHAT_API_CORP",
        timeout=120,
    )

    columns_with_dtypes = json.dumps(schema["column_dtypes"], ensure_ascii=False)
    extra_tables_info = _build_extra_tables_info(schema)
    test_warning = ""
    if not schema.get("test_has_features", True):
        test_warning = (
            "ВНИМАНИЕ: test.csv содержит только ID-колонку. "
            "В df_test_out кроме id_column ставь признаки как NaN "
            "(df_test_out[feat] = np.nan) — они будут заполнены fillna(0) в OutputWriter."
        )

    generated_code = []
    computed_train_dfs = []
    computed_test_dfs = []

    for set_idx, ideas in enumerate(state["feature_ideas"]):
        if not ideas:
            generated_code.append(None)
            computed_train_dfs.append(None)
            computed_test_dfs.append(None)
            continue

        feature_ideas_json = json.dumps(ideas, ensure_ascii=False, indent=2)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            id_column=schema["id_column"],
            target_column=schema["target_column"],
            columns_with_dtypes=columns_with_dtypes,
            extra_tables_info=extra_tables_info,
            feature_ideas_json=feature_ideas_json,
            test_warning=test_warning,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        code_str = None
        df_train_out = None
        df_test_out = None

        for attempt in range(2):
            try:
                if attempt == 0:
                    response = llm.invoke(messages)
                    code_str = _extract_code(response.content)
                else:
                    fix_prompt = FIX_PROMPT_TEMPLATE.format(
                        error_message=str(last_error),
                        previous_code=code_str,
                    )
                    fix_messages = [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=fix_prompt),
                    ]
                    response = llm.invoke(fix_messages)
                    code_str = _extract_code(response.content)

                df_train_out, df_test_out = _execute_code(code_str, state)
                break
            except Exception as e:
                last_error = e
                state["errors_log"].append(f"FeatureCoder set {set_idx + 1} attempt {attempt + 1}: {e}")

        generated_code.append(code_str)
        computed_train_dfs.append(df_train_out)
        computed_test_dfs.append(df_test_out)

    return {
        "generated_code": generated_code,
        "computed_train_dfs": computed_train_dfs,
        "computed_test_dfs": computed_test_dfs,
    }
