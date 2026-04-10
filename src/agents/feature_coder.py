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
- Вычисляй агрегации ТОЛЬКО на df_train, применяй к df_test через merge/map
- Обрабатывай NaN: fillna медианой train или -1
- Итоговые df_train_out и df_test_out должны содержать только нужные колонки
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
available_columns: {columns_with_dtypes}
</schema>
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
    # реализуй признаки из feature_ideas
    # ...
    return df_train_out, df_test_out
Требования к возвращаемым df:
- df_train_out колонки: [{id_column}, {target_column}, feature_1, ..., feature_N]
- df_test_out колонки:  [{id_column}, feature_1, ..., feature_N]
- Имена признаков должны совпадать в обоих df
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


def _extract_code(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def _execute_code(code_str: str, state: AgentState):
    sandbox = {
        "df_train": state["df_train"].copy(),
        "df_test": state["df_test"].copy(),
        "extra_tables": state["extra_tables"],
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
        timeout=120,
    )

    columns_with_dtypes = json.dumps(schema["column_dtypes"], ensure_ascii=False)
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
