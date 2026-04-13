"""Агент FeatureEngineer — LLM выбирает операции из меню,
затем формируется автоматический пул кандидатов. Без exec(), без sandbox."""
import json

import numpy as np
import pandas as pd
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState
from src.utils.operations import execute_operation

# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
<role>
Ты — главный feature engineer для бинарной классификации (CatBoost).
Ты НЕ пишешь код. Ты выбираешь операции из фиксированного меню.
Именно от ТВОИХ решений зависит качество модели.
</role>
<context>
Автоматика делает ТОЛЬКО базовое кодирование столбцов train (FREQ_ENCODE, TARGET_ENCODE, LABEL_ENCODE, IS_NULL).
ВСЁ остальное — ТВОЯ ответственность. Без твоих операций модель будет слабой.
</context>
<rules>
1. Используй ТОЛЬКО столбцы и таблицы из предоставленной схемы.
2. НЕ используй target_column и id_column как входные столбцы операций.
3. ОБЯЗАТЕЛЬНО покрой ВСЕ дополнительные таблицы — не пропускай ни одной.
4. Для каждой доп. таблицы используй РАЗНЫЕ AGG-функции (mean, std, max, min, nunique, count) — не только mean.
5. Для таблиц с пометкой "1-к-1" используй DIRECT_NUMERIC и EXTRA_*_ENCODE.
6. Для таблиц "много-к-1" используй AGG, COUNT, CROSS_AGG.
7. INTERACTION: обязательно включи div и sub (отношения и разности важнее произведений).
8. Верни ТОЛЬКО валидный JSON-массив, без markdown-обёртки и пояснений.
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
<null_percentages>
{null_percentages}
</null_percentages>
<sample_rows>
{sample_rows}
</sample_rows>
</context>

<auto_pool_info>
Автоматика УЖЕ создаёт: FREQ_ENCODE, TARGET_ENCODE, LABEL_ENCODE, IS_NULL для каждого столбца train.
НЕ дублируй эти операции. Твоя задача — всё остальное:
- AGG, COUNT, CROSS_AGG для дополнительных таблиц (разные функции!)
- DIRECT_NUMERIC, EXTRA_*_ENCODE для таблиц 1-к-1
- INTERACTION (особенно div и sub) между числовыми столбцами
- RANK, RATIO_TO_GROUP для нормализации
</auto_pool_info>

<operations_menu>
1. AGG — агрегация столбца из доп. таблицы (func: mean/std/sum/max/min/count/nunique/median)
   {{"op": "AGG", "table": "table_name", "key": "join_key", "column": "col_name", "func": "mean"}}

2. COUNT — количество строк в доп. таблице по ключу
   {{"op": "COUNT", "table": "table_name", "key": "join_key"}}

3. INTERACTION — арифметическое взаимодействие (op_type: mul/div/add/sub)
   {{"op": "INTERACTION", "col1": "col_a", "op_type": "div", "col2": "col_b"}}

4. RANK — перцентильный ранг числовой колонки
   {{"op": "RANK", "column": "col_name"}}

5. DIRECT_NUMERIC — прямое подключение числовой колонки из 1-к-1 доп. таблицы
   {{"op": "DIRECT_NUMERIC", "table": "table_name", "key": "join_key", "column": "col_name"}}

6. RATIO_TO_GROUP — отношение значения к среднему группы из доп. таблицы
   {{"op": "RATIO_TO_GROUP", "column": "train_col", "table": "table_name", "key": "join_key", "ref_column": "col_name"}}

7. EXTRA_FREQ_ENCODE — частотное кодирование категориальной колонки из доп. таблицы
   {{"op": "EXTRA_FREQ_ENCODE", "table": "table_name", "key": "join_key", "column": "cat_col"}}

8. EXTRA_TARGET_ENCODE — target-mean кодирование категориальной колонки из доп. таблицы
   {{"op": "EXTRA_TARGET_ENCODE", "table": "table_name", "key": "join_key", "column": "cat_col"}}

9. EXTRA_LABEL_ENCODE — числовое кодирование категориальной колонки из доп. таблицы
   {{"op": "EXTRA_LABEL_ENCODE", "table": "table_name", "key": "join_key", "column": "cat_col"}}

10. CROSS_AGG — агрегация по составному ключу из доп. таблицы
    {{"op": "CROSS_AGG", "table": "table_name", "keys": ["key1", "key2"], "column": "col_name", "func": "mean"}}

Также доступны (автоматика их уже делает для train, но ты можешь использовать осознанно):
11. FREQ_ENCODE: {{"op": "FREQ_ENCODE", "column": "col_name"}}
12. TARGET_ENCODE: {{"op": "TARGET_ENCODE", "column": "col_name"}}
13. IS_NULL: {{"op": "IS_NULL", "column": "col_name"}}
14. LABEL_ENCODE: {{"op": "LABEL_ENCODE", "column": "col_name"}}
</operations_menu>

<task>
Предложи 15-25 операций для предсказания "{target_column}".
Стратегия:
1. Покрой ВСЕ дополнительные таблицы (AGG с разными func, COUNT, DIRECT_NUMERIC).
2. Для каждой числовой колонки доп. таблицы — минимум 2 AGG-функции (например mean + std).
3. Добавь 3-5 INTERACTION (приоритет: div и sub между связанными столбцами).
4. Добавь RANK для 1-2 самых важных числовых столбцов.
5. Для 1-к-1 таблиц — DIRECT_NUMERIC для числовых, EXTRA_TARGET_ENCODE для категорий.
Верни ТОЛЬКО JSON-массив операций.
</task>"""


ROUND2_PROMPT_TEMPLATE = """\
<контекст>
target_column: {target_column}
id_column: {id_column}
train_shape: {train_shape}
</контекст>

<результаты_раунда_1>
Лучшие фичи по корреляции с таргетом:
{top_features}

Слабые фичи:
{weak_features}
</результаты_раунда_1>

<уже_выполненные_операции>
{executed_ops_summary}
</уже_выполненные_операции>

<доступные_таблицы>
{extra_tables_info}
</доступные_таблицы>

<operations_menu>
{operations_menu}
</operations_menu>

<задание>
Проанализируй результаты раунда 1. Предложи 5-10 НОВЫХ операций (НЕ дублируя уже выполненные).
Стратегия:
1. Посмотри какие AGG-функции уже использованы — добавь ДРУГИЕ (std, max, min) для тех же таблиц.
2. Добавь INTERACTION (div, sub) между столбцами из сильных фичей.
3. Попробуй CROSS_AGG с составными ключами.
4. RATIO_TO_GROUP для числовых столбцов, нормализованных по группе.
5. Если какие-то доп. таблицы НЕ покрыты — покрой их.
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
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _parse_llm_json(text: str) -> list:
    """Парсим ответ LLM, убираем markdown-обёртку если есть."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    result = json.loads(text)
    return result if isinstance(result, list) else []

def _build_extra_tables_info(schema: dict, extra_tables: dict = None) -> str:
    """Формируем подробное описание доп. таблиц для промпта.
    Включает nunique, пометку 1-к-1 и примеры строк."""
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

        # Описание колонок с nunique
        col_lines = []
        tdf = extra_tables.get(tname) if extra_tables else None
        for c, dtype in cols.items():
            nuniq = ""
            if tdf is not None and c in tdf.columns:
                nuniq = f", nunique={tdf[c].nunique()}"
            col_lines.append(f"    {c} ({dtype}{nuniq})")

        # Определяем 1-к-1 связь
        one_to_one_keys = []
        if tdf is not None:
            for key in join_keys:
                if key in tdf.columns and tdf[key].nunique() == len(tdf):
                    one_to_one_keys.append(key)

        relation = ""
        if one_to_one_keys:
            relation = f"\n  Связь: 1-к-1 по {one_to_one_keys} → используй DIRECT_NUMERIC"
        else:
            relation = "\n  Связь: много-к-1 → используй AGG/COUNT"

        # Примеры строк
        sample = ""
        if tdf is not None:
            sample = f"\n  Примеры (3 строки):\n{tdf.head(3).to_string(index=False)}"

        parts.append(
            f"Таблица '{tname}' [{shape[0]} rows x {shape[1]} cols]:\n"
            f"  Join keys: {join_keys}{relation}\n"
            f"  Колонки:\n" + "\n".join(col_lines) + sample
        )
    return "\n\n".join(parts) if parts else "Нет доступных дополнительных таблиц."


def _generate_auto_pool(schema, df_train, df_test, extra_tables):
    """Детерминированный набор кандидатов независимо от LLM."""
    ops = []
    id_col = schema["id_column"]
    target_col = schema["target_column"]
    feature_cols = [c for c in df_train.columns if c not in (id_col, target_col)]

    # Фичи из столбцов train
    for col in feature_cols:
        nunique = df_train[col].nunique()
        n = len(df_train)
        # Частотное кодирование для низкой/средней кардинальности
        if nunique < n * 0.5:
            ops.append({"op": "FREQ_ENCODE", "column": col})
        # Таргет-кодирование для средней кардинальности
        if 2 < nunique < n * 0.3:
            ops.append({"op": "TARGET_ENCODE", "column": col})
        # Индикатор пропусков
        if df_train[col].isna().sum() > 0:
            ops.append({"op": "IS_NULL", "column": col})
        # Label-кодирование для категорий
        if not pd.api.types.is_numeric_dtype(df_train[col]):
            ops.append({"op": "LABEL_ENCODE", "column": col})

    # Базовые фичи из дополнительных таблиц (LLM добавит std, div, CROSS_AGG сверху)
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
            ops.append({"op": "COUNT", "table": tname, "key": key})
            for col in numeric_cols[:5]:
                ops.append({"op": "AGG", "table": tname, "key": key,
                            "column": col, "func": "mean"})
            for col in cat_cols[:2]:
                ops.append({"op": "AGG", "table": tname, "key": key,
                            "column": col, "func": "nunique"})

            tdf = extra_tables.get(tname)
            is_one_to_one = (tdf is not None and key in tdf.columns
                             and tdf[key].nunique() == len(tdf))

            if is_one_to_one:
                for col in numeric_cols[:6]:
                    ops.append({"op": "DIRECT_NUMERIC", "table": tname,
                                "key": key, "column": col})
                for col in cat_cols[:8]:
                    ops.append({"op": "EXTRA_FREQ_ENCODE", "table": tname,
                                "key": key, "column": col})
                    ops.append({"op": "EXTRA_TARGET_ENCODE", "table": tname,
                                "key": key, "column": col})
                    ops.append({"op": "EXTRA_LABEL_ENCODE", "table": tname,
                                "key": key, "column": col})

    return ops


# ---------------------------------------------------------------------------
# Точка входа агента
# ---------------------------------------------------------------------------

def run(state: AgentState) -> dict:
    schema = state["schema_info"]
    id_col = schema["id_column"]
    target_col = schema["target_column"]

    # Робастное определение ID-колонки в test (может отличаться от train)
    if id_col not in state["df_test"].columns:
        test_id_col = state["df_test"].columns[0]
        print(f"  [FeatureEngineer] WARNING: '{id_col}' not in test, using '{test_id_col}'")
    else:
        test_id_col = id_col

    # --- Фаза 1: LLM предлагает операции ---
    llm_ops = []
    try:
        llm = GigaChat(
            model="GigaChat-2-Max",
            temperature=0.3,
            max_tokens=7000,
            verify_ssl_certs=False,
            profanity_check=False,
            scope="GIGACHAT_API_CORP",
            timeout=120,
        )
        extra_tables_info = _build_extra_tables_info(schema, state["extra_tables"])
        basic_stats = json.dumps(schema.get("basic_stats", {}),
                                 ensure_ascii=False, indent=2)
        # Процент пропусков и примеры строк для контекста LLM
        null_pct = json.dumps(schema.get("null_percentages", {}),
                              ensure_ascii=False, indent=2)
        sample_rows_data = schema.get("sample_rows", "")
        if isinstance(sample_rows_data, (dict, list)):
            sample_rows_data = json.dumps(sample_rows_data, ensure_ascii=False, indent=2)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            readme_text=schema["readme_text"][:3000],
            target_column=target_col,
            id_column=id_col,
            columns_with_dtypes=json.dumps(schema["column_dtypes"],
                                           ensure_ascii=False),
            train_shape=schema["train_shape"],
            extra_tables_info=extra_tables_info,
            basic_stats=basic_stats,
            null_percentages=null_pct,
            sample_rows=str(sample_rows_data)[:2000],
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

    # --- Фаза 2: автоматический пул кандидатов ---
    auto_ops = _generate_auto_pool(
        schema, state["df_train"], state["df_test"], state["extra_tables"]
    )
    print(f"  [FeatureEngineer] Auto-pool: {len(auto_ops)} operations")

    # --- Фаза 3: дедупликация и выполнение ---
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
    test_out = state["df_test"][[test_id_col]].copy()
    if test_id_col != id_col:
        test_out = test_out.rename(columns={test_id_col: id_col})
    candidate_names = []

    # Passthrough: исходные числовые столбцы train как кандидаты
    feature_cols = [c for c in state["df_train"].columns if c not in (id_col, target_col)]
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(state["df_train"][col]) and col in state["df_test"].columns:
            try:
                tr_vals = state["df_train"][col].fillna(0).values.astype(float)
                te_vals = state["df_test"][col].fillna(0).values.astype(float)
            except (ValueError, TypeError):
                continue
            if np.std(tr_vals) < 1e-12:
                continue
            name = f"raw_{col}"
            train_out[name] = tr_vals
            test_out[name] = te_vals
            candidate_names.append(name)
    print(f"  [FeatureEngineer] Raw numeric passthrough: {len(candidate_names)} features")

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
        # Пропускаем нечисловые и фичи с нулевой дисперсией
        try:
            tr_arr = np.asarray(tr_vals, dtype=float)
            te_arr = np.asarray(te_vals, dtype=float)
        except (ValueError, TypeError):
            continue
        if np.std(np.nan_to_num(tr_arr)) < 1e-12:
            continue
        train_out[name] = tr_arr
        test_out[name] = te_arr
        candidate_names.append(name)

    print(f"  [FeatureEngineer] Round 1 candidates: {len(candidate_names)}")

    # --- Фаза 4: второй раунд LLM — ранжируем по корреляции, просим улучшения ---
    if candidate_names and len(candidate_names) >= 3:
        try:
            # Мгновенное ранжирование по корреляции с таргетом (без CatBoost)
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

            # Формируем список уже выполненных операций для LLM
            executed_ops_lines = []
            for op in unique_ops:
                executed_ops_lines.append(json.dumps(op, ensure_ascii=False))
            executed_ops_str = "\n".join(executed_ops_lines[:50])

            round2_prompt = ROUND2_PROMPT_TEMPLATE.format(
                target_column=target_col,
                id_column=id_col,
                train_shape=schema["train_shape"],
                top_features=top_str,
                weak_features=weak_str,
                executed_ops_summary=executed_ops_str,
                extra_tables_info=_build_extra_tables_info(schema, state["extra_tables"]),
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
                try:
                    tr_arr = np.asarray(tr_vals, dtype=float)
                    te_arr = np.asarray(te_vals, dtype=float)
                except (ValueError, TypeError):
                    continue
                if np.std(np.nan_to_num(tr_arr)) < 1e-12:
                    continue
                train_out[name] = tr_arr
                test_out[name] = te_arr
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
