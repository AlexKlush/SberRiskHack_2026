"""FeatureEngineer agent — LLM picks operations from a fixed menu, then
an automatic pool of candidates is added.  No exec(), no sandbox."""
import json
from itertools import combinations

import numpy as np
import pandas as pd
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState
from src.utils.operations import execute_operation

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
<role>
Ты — senior feature engineer для бинарной классификации (CatBoost).
Ты НЕ пишешь код. Ты выбираешь операции из фиксированного меню.
</role>
<rules>
1. Используй ТОЛЬКО столбцы и таблицы из предоставленной схемы.
2. НЕ используй target_column и id_column как входные столбцы операций.
3. АКТИВНО используй дополнительные таблицы — AGG и COUNT самые сильные.
4. Верни ТОЛЬКО валидный JSON-массив, без markdown-обёртки и пояснений.
5. Выбирай разнообразные операции — не повторяй одно и то же.
</rules>"""

USER_PROMPT_TEMPLATE = """\
<context>
<readme>{readme_text}</readme>
<schema>
target_column: {target_column}
id_column: {id_column}
train_columns (кроме id и target): {columns_with_dtypes}
train_shape: {train_shape}
</schema>
<extra_tables>
{extra_tables_info}
</extra_tables>
<data_stats>
{basic_stats}
</data_stats>
</context>

<operations_menu>
Выбери 10-15 операций из этого меню:

1. FREQ_ENCODE — частотное кодирование столбца
   {{"op": "FREQ_ENCODE", "column": "col_name"}}

2. TARGET_ENCODE — сглаженное среднее таргета по группе
   {{"op": "TARGET_ENCODE", "column": "col_name"}}

3. AGG — агрегация столбца из доп. таблицы (func: mean/std/sum/max/min/count/nunique/median)
   {{"op": "AGG", "table": "table_name", "key": "join_key", "column": "col_name", "func": "mean"}}

4. COUNT — количество строк в доп. таблице по ключу
   {{"op": "COUNT", "table": "table_name", "key": "join_key"}}

5. INTERACTION — арифметическое взаимодействие (op_type: mul/div/add/sub)
   {{"op": "INTERACTION", "col1": "col_a", "op_type": "div", "col2": "col_b"}}

6. RANK — перцентильный ранг числовой колонки
   {{"op": "RANK", "column": "col_name"}}

7. IS_NULL — индикатор пропуска (1 если NaN, 0 иначе)
   {{"op": "IS_NULL", "column": "col_name"}}

8. LABEL_ENCODE — числовое кодирование категориальной переменной
   {{"op": "LABEL_ENCODE", "column": "col_name"}}

9. DIRECT_NUMERIC — прямое подключение числовой колонки из 1-к-1 доп. таблицы
   {{"op": "DIRECT_NUMERIC", "table": "table_name", "key": "join_key", "column": "col_name"}}

10. RATIO_TO_GROUP — отношение значения к среднему группы из доп. таблицы
    {{"op": "RATIO_TO_GROUP", "column": "train_col", "table": "table_name", "key": "join_key", "ref_column": "col_name"}}

11. EXTRA_FREQ_ENCODE — частотное кодирование категориальной колонки из доп. таблицы
    {{"op": "EXTRA_FREQ_ENCODE", "table": "table_name", "key": "join_key", "column": "cat_col"}}

12. EXTRA_TARGET_ENCODE — target-mean кодирование категориальной колонки из доп. таблицы
    {{"op": "EXTRA_TARGET_ENCODE", "table": "table_name", "key": "join_key", "column": "cat_col"}}

13. EXTRA_LABEL_ENCODE — числовое кодирование категориальной колонки из доп. таблицы
    {{"op": "EXTRA_LABEL_ENCODE", "table": "table_name", "key": "join_key", "column": "cat_col"}}

14. CROSS_AGG — агрегация по составному ключу из доп. таблицы
    {{"op": "CROSS_AGG", "table": "table_name", "keys": ["key1", "key2"], "column": "col_name", "func": "mean"}}
</operations_menu>

<task>
Выбери 10-15 самых полезных операций для предсказания "{target_column}".
Думай: какие признаки будут информативны для бинарной классификации?
ВАЖНО: если есть дополнительные таблицы — AGG, COUNT и DIRECT_NUMERIC дают сильнейшие фичи.
Верни ТОЛЬКО JSON-массив операций.
</task>"""


ROUND2_PROMPT_TEMPLATE = """\
<контекст>
target_column: {target_column}
id_column: {id_column}
train_shape: {train_shape}
</контекст>

<результаты_раунда_1>
Лучшие фичи (ROC-AUC индивидуально):
{top_features}

Слабые фичи:
{weak_features}
</результаты_раунда_1>

<доступные_таблицы>
{extra_tables_info}
</доступные_таблицы>

<operations_menu>
{operations_menu}
</operations_menu>

<задание>
Проанализируй результаты первого раунда. Предложи 5-10 НОВЫХ операций, которые дополнят сильные фичи.
Фокусируйся на:
1. AGG с другими функциями (std, max, min, nunique) для таблиц, давших сильные фичи
2. INTERACTION между столбцами, связанными с сильными фичами
3. Новые комбинации таблиц и ключей, которые ещё не были использованы
4. CROSS_AGG если в таблице есть несколько ключей, совпадающих со столбцами train
НЕ повторяй операции из первого раунда.
Верни ТОЛЬКО JSON-массив операций.
</задание>"""

OPERATIONS_MENU_TEXT = """\
1. FREQ_ENCODE: {{"op": "FREQ_ENCODE", "column": "col"}}
2. TARGET_ENCODE: {{"op": "TARGET_ENCODE", "column": "col"}}
3. AGG: {{"op": "AGG", "table": "t", "key": "k", "column": "c", "func": "mean|std|sum|max|min|count|nunique|median"}}
4. COUNT: {{"op": "COUNT", "table": "t", "key": "k"}}
5. INTERACTION: {{"op": "INTERACTION", "col1": "a", "op_type": "mul|div|add|sub", "col2": "b"}}
6. RANK: {{"op": "RANK", "column": "col"}}
7. IS_NULL: {{"op": "IS_NULL", "column": "col"}}
8. LABEL_ENCODE: {{"op": "LABEL_ENCODE", "column": "col"}}
9. DIRECT_NUMERIC: {{"op": "DIRECT_NUMERIC", "table": "t", "key": "k", "column": "c"}}
10. RATIO_TO_GROUP: {{"op": "RATIO_TO_GROUP", "column": "c", "table": "t", "key": "k", "ref_column": "rc"}}
11. EXTRA_FREQ_ENCODE: {{"op": "EXTRA_FREQ_ENCODE", "table": "t", "key": "k", "column": "c"}}
12. EXTRA_TARGET_ENCODE: {{"op": "EXTRA_TARGET_ENCODE", "table": "t", "key": "k", "column": "c"}}
13. EXTRA_LABEL_ENCODE: {{"op": "EXTRA_LABEL_ENCODE", "table": "t", "key": "k", "column": "c"}}
14. CROSS_AGG: {{"op": "CROSS_AGG", "table": "t", "keys": ["k1", "k2"], "column": "c", "func": "mean"}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_llm_json(text: str) -> list:
    """Parse LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    result = json.loads(text)
    return result if isinstance(result, list) else []

def _build_extra_tables_info(schema: dict) -> str:
    extra_schema = schema.get("extra_tables_schema", {})
    if not extra_schema:
        return "Нет дополнительных таблиц."
    parts = []
    for tname, tinfo in extra_schema.items():
        if tinfo["shape"][0] > 500_000:
            continue
        cols = tinfo["columns"]
        join_keys = tinfo["join_keys"]
        shape = tinfo["shape"]
        col_desc = ", ".join(f"{c} ({dtype})" for c, dtype in cols.items())
        parts.append(
            f"Таблица '{tname}' [{shape[0]} rows x {shape[1]} cols]:\n"
            f"  Колонки: {col_desc}\n"
            f"  Join keys: {join_keys}"
        )
    return "\n".join(parts) if parts else "Нет доступных дополнительных таблиц."


def _generate_auto_pool(schema, df_train, df_test, extra_tables):
    """Build a deterministic set of candidate operations regardless of LLM."""
    ops = []
    id_col = schema["id_column"]
    target_col = schema["target_column"]
    feature_cols = [c for c in df_train.columns if c not in (id_col, target_col)]

    # --- From train columns ---
    for col in feature_cols:
        nunique = df_train[col].nunique()
        n = len(df_train)
        # Frequency encoding for low/mid cardinality
        if nunique < n * 0.5:
            ops.append({"op": "FREQ_ENCODE", "column": col})
        # Target encoding for medium cardinality
        if 2 < nunique < n * 0.3:
            ops.append({"op": "TARGET_ENCODE", "column": col})
        # Null indicator
        if df_train[col].isna().sum() > 0:
            ops.append({"op": "IS_NULL", "column": col})
        # Label encode for categoricals
        if not pd.api.types.is_numeric_dtype(df_train[col]):
            ops.append({"op": "LABEL_ENCODE", "column": col})

    # --- INTERACTION between numeric train features (capped to avoid explosion) ---
    numeric_features = [c for c in feature_cols
                        if pd.api.types.is_numeric_dtype(df_train[c])]
    if len(numeric_features) >= 2:
        pairs = list(combinations(numeric_features[:6], 2))  # max 15 pairs
        for c1, c2 in pairs[:10]:  # max 10 interactions total
            ops.append({"op": "INTERACTION", "col1": c1,
                        "op_type": "mul", "col2": c2})

    # --- From extra tables ---
    extra_schema = schema.get("extra_tables_schema", {})
    for tname, tinfo in extra_schema.items():
        if tinfo["shape"][0] > 500_000:
            continue
        join_keys = tinfo["join_keys"]
        all_cols = tinfo["columns"]

        numeric_cols = [c for c, d in all_cols.items()
                        if ("int" in d or "float" in d)
                        and c not in join_keys and c != target_col
                        and not c.startswith("Unnamed")]
        cat_cols = [c for c, d in all_cols.items()
                    if ("object" in d or "category" in d)
                    and c not in join_keys]

        for key in join_keys[:2]:
            if key not in df_train.columns:
                continue
            # Count
            ops.append({"op": "COUNT", "table": tname, "key": key})
            # Numeric aggregations (top 5 columns, mean only — LLM round 2 adds more)
            for col in numeric_cols[:5]:
                ops.append({"op": "AGG", "table": tname, "key": key,
                            "column": col, "func": "mean"})
            # Nunique for categoricals in many-to-1 tables
            for col in cat_cols[:2]:
                ops.append({"op": "AGG", "table": tname, "key": key,
                            "column": col, "func": "nunique"})

            # Check if 1-to-1 table
            tdf = extra_tables.get(tname)
            is_one_to_one = (tdf is not None and key in tdf.columns
                             and tdf[key].nunique() == len(tdf))

            if is_one_to_one:
                # Direct numeric columns
                for col in numeric_cols[:6]:
                    ops.append({"op": "DIRECT_NUMERIC", "table": tname,
                                "key": key, "column": col})
                # Categorical columns: freq, target, label encoding
                for col in cat_cols[:8]:
                    ops.append({"op": "EXTRA_FREQ_ENCODE", "table": tname,
                                "key": key, "column": col})
                    ops.append({"op": "EXTRA_TARGET_ENCODE", "table": tname,
                                "key": key, "column": col})
                    ops.append({"op": "EXTRA_LABEL_ENCODE", "table": tname,
                                "key": key, "column": col})

    return ops


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def run(state: AgentState) -> dict:
    schema = state["schema_info"]
    id_col = schema["id_column"]
    target_col = schema["target_column"]

    # --- Phase 1: LLM suggests operations ---
    llm_ops = []
    try:
        llm = GigaChat(
            model="GigaChat-2-Max",
            temperature=0.3,
            max_tokens=2000,
            verify_ssl_certs=False,
            profanity_check=False,
            scope="GIGACHAT_API_CORP",
            timeout=120,
        )
        extra_tables_info = _build_extra_tables_info(schema)
        basic_stats = json.dumps(schema.get("basic_stats", {}),
                                 ensure_ascii=False, indent=2)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            readme_text=schema["readme_text"][:3000],  # truncate long readmes
            target_column=target_col,
            id_column=id_col,
            columns_with_dtypes=json.dumps(schema["column_dtypes"],
                                           ensure_ascii=False),
            train_shape=schema["train_shape"],
            extra_tables_info=extra_tables_info,
            basic_stats=basic_stats,
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        llm_ops = _parse_llm_json(response.content)
        print(f"  [FeatureEngineer] LLM suggested {len(llm_ops)} operations")
    except Exception as e:
        state["errors_log"].append(f"FeatureEngineer LLM: {e}")
        print(f"  [FeatureEngineer] LLM failed: {e}, using auto-pool only")

    # --- Phase 2: automatic candidate pool ---
    auto_ops = _generate_auto_pool(
        schema, state["df_train"], state["df_test"], state["extra_tables"]
    )
    print(f"  [FeatureEngineer] Auto-pool: {len(auto_ops)} operations")

    # --- Phase 3: deduplicate & execute ---
    all_ops = llm_ops + auto_ops
    seen = set()
    unique_ops = []
    for op in all_ops:
        sig = json.dumps(op, sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            unique_ops.append(op)
    print(f"  [FeatureEngineer] Unique operations to execute: {len(unique_ops)}")

    train_out = state["df_train"][[id_col, target_col]].copy()
    test_out = state["df_test"][[id_col]].copy()
    candidate_names = []

    for op in unique_ops:
        result = execute_operation(
            op, state["df_train"], state["df_test"],
            state["extra_tables"], target_col,
        )
        if result is None:
            continue
        name, tr_vals, te_vals = result
        if name in candidate_names or name in (id_col, target_col):
            continue
        # Skip zero-variance features
        if np.std(tr_vals) < 1e-12:
            continue
        train_out[name] = tr_vals
        test_out[name] = te_vals
        candidate_names.append(name)

    print(f"  [FeatureEngineer] Round 1 candidates: {len(candidate_names)}")

    # --- Phase 4: Multi-turn LLM — rank round 1 by correlation, ask for improvements ---
    if candidate_names and len(candidate_names) >= 3:
        try:
            # Fast ranking by abs correlation with target (instant, no CatBoost)
            target_data = state["df_train"][target_col].values.astype(float)
            corr_scores = []
            for col in candidate_names:
                vals = train_out[col].fillna(0).values.astype(float)
                c = abs(np.corrcoef(vals, target_data)[0, 1])
                corr_scores.append((col, 0.0 if np.isnan(c) else round(c, 4)))
            corr_scores.sort(key=lambda x: x[1], reverse=True)
            top_5 = corr_scores[:5]
            weak_5 = corr_scores[-5:]

            top_str = "\n".join(f"  - {n}: corr {s:.4f}" for n, s in top_5)
            weak_str = "\n".join(f"  - {n}: corr {s:.4f}" for n, s in weak_5)

            print(f"  [FeatureEngineer] Round 1 top (corr): {[(n, f'{s:.4f}') for n, s in top_5]}")

            round2_prompt = ROUND2_PROMPT_TEMPLATE.format(
                target_column=target_col,
                id_column=id_col,
                train_shape=schema["train_shape"],
                top_features=top_str,
                weak_features=weak_str,
                extra_tables_info=_build_extra_tables_info(schema),
                operations_menu=OPERATIONS_MENU_TEXT,
            )

            llm2 = GigaChat(
                model="GigaChat-2-Max",
                temperature=0.4,
                max_tokens=2000,
                verify_ssl_certs=False,
                profanity_check=False,
                scope="GIGACHAT_API_CORP",
                timeout=120,
            )
            messages2 = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=round2_prompt),
            ]
            response2 = llm2.invoke(messages2)
            llm_ops_2 = _parse_llm_json(response2.content)
            print(f"  [FeatureEngineer] LLM round 2 suggested {len(llm_ops_2)} operations")

            # Execute round 2 operations
            for op in llm_ops_2:
                sig = json.dumps(op, sort_keys=True)
                if sig in seen:
                    continue
                seen.add(sig)
                result = execute_operation(
                    op, state["df_train"], state["df_test"],
                    state["extra_tables"], target_col,
                )
                if result is None:
                    continue
                name, tr_vals, te_vals = result
                if name in candidate_names or name in (id_col, target_col):
                    continue
                if np.std(tr_vals) < 1e-12:
                    continue
                train_out[name] = tr_vals
                test_out[name] = te_vals
                candidate_names.append(name)

            print(f"  [FeatureEngineer] Total candidates after round 2: {len(candidate_names)}")
        except Exception as e:
            state["errors_log"].append(f"FeatureEngineer LLM round 2: {e}")
            print(f"  [FeatureEngineer] LLM round 2 failed: {e}")

    if candidate_names:
        print(f"  [FeatureEngineer] Final names: {candidate_names}")

    return {
        "candidate_features_train": train_out,
        "candidate_features_test": test_out,
        "candidate_names": candidate_names,
    }
