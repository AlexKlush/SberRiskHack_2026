"""FeatureIdeator agent — generates feature ideas via GigaChat."""
import json

from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState

SYSTEM_PROMPT = """\
<role>
Ты — senior feature engineer, специалист по созданию признаков для задач \
бинарной классификации на табличных данных. Ты работаешь с моделями бустинга \
(CatBoost). Твои признаки должны быть информативными, вычислимыми и \
интерпретируемыми.
</role>
<operating_principles>
1. Используй ТОЛЬКО столбцы из предоставленной схемы (train/test И дополнительные таблицы)
2. Не используй target_column как входной признак (data leakage)
3. Не используй id_column как признак
4. Все признаки должны вычисляться идентично на train и test
5. АКТИВНО используй дополнительные таблицы — джойни их к train/test по join_keys
6. Агрегации (mean, count, std, nunique) по дополнительным таблицам — самые сильные признаки
7. Верни ТОЛЬКО валидный JSON, без markdown-обёртки и пояснений
</operating_principles>"""

USER_PROMPT_TEMPLATE = """\
<context>
<readme>{readme_text}</readme>
<schema>
target_column: {target_column}
id_column: {id_column}
train_test_columns: {columns_with_dtypes}
train_shape: {train_shape}
</schema>
<data_stats>
{basic_stats}
</data_stats>
<sample_rows>
{sample_rows}
</sample_rows>
<extra_tables>
{extra_tables_info}
</extra_tables>
<forbidden_feature_names>
Следующие имена уже заняты существующими колонками — НЕ используй их как имена признаков:
{reserved_names}
</forbidden_feature_names>
<round>{round_number} из 2</round>
</context>
<task>
Придумай ровно 5 идей признаков для данного датасета.
ВАЖНО: используй дополнительные таблицы! Джойни их к train/test по указанным join_keys.
Лучшие признаки — агрегации из доп. таблиц: count, mean, std, nunique, ratio.
{round2_hint}
</task>
<feature_categories>
Рассмотри следующие типы (выбери наиболее подходящие для данных):
- AGGREGATIONS: группировка по ключам из доп. таблиц (mean, std, count, nunique)
- INTERACTIONS: произведения, отношения, разности числовых столбцов
- DATE_FEATURES: день недели, месяц, дни от референсной точки (если есть даты)
- RANK_FEATURES: ранги значений внутри групп
- STATISTICAL: z-score, отклонение от медианы группы, квантильные бины
- FREQUENCY_ENCODING: частота встречаемости категориальных значений
- CROSS_TABLE: признаки, объединяющие данные из нескольких таблиц через merge/join
</feature_categories>
<thinking>
Перед ответом коротко обдумай:
- Какие доп. таблицы можно заджойнить к train/test и какие агрегации построить?
- Какие взаимодействия между столбцами из разных таблиц могут быть важны?
- Какие count/mean/std агрегации по ключевым группировкам будут информативны?
</thinking>
<few_shot_example>
Хорошая идея (с доп. таблицей):
{{"name": "user_avg_basket_size", "description": "Среднее количество товаров в корзине пользователя из таблицы users", "columns_used": ["user_id"], "extra_tables_used": ["users"], "category": "CROSS_TABLE", "hypothesis": "Пользователи с большой корзиной чаще покупают повторно"}}
Хорошая идея (агрегация):
{{"name": "product_reorder_rate", "description": "Доля повторных покупок товара из order_items", "columns_used": ["product_id"], "extra_tables_used": ["order_items"], "category": "AGGREGATIONS", "hypothesis": "Товары с высокой долей повторных покупок чаще покупаются снова"}}
Плохая идея (не делай так):
{{"name": "raw_id", "description": "Использовать ID напрямую", "columns_used": ["id"]}}
</few_shot_example>
<output_schema>
Верни ТОЛЬКО JSON-массив ровно из 5 объектов:
[
  {{
    "name": "snake_case_max_30_chars",
    "description": "что вычисляется (1 предложение)",
    "columns_used": ["col1", "col2"],
    "extra_tables_used": ["table_name"],
    "category": "AGGREGATIONS|INTERACTIONS|CROSS_TABLE|FREQUENCY_ENCODING|RANK_FEATURES|STATISTICAL",
    "hypothesis": "почему полезен для таргета (1 предложение)"
  }}
]
</output_schema>"""


def _build_extra_tables_info(schema: dict) -> str:
    extra_schema = schema.get("extra_tables_schema", {})
    if not extra_schema:
        return "Нет дополнительных таблиц."

    parts = []
    for tname, tinfo in extra_schema.items():
        cols = tinfo["columns"]
        join_keys = tinfo["join_keys"]
        shape = tinfo["shape"]
        col_desc = ", ".join(f"{c} ({dtype})" for c, dtype in cols.items())
        parts.append(
            f"Таблица '{tname}' [{shape[0]} rows x {shape[1]} cols]:\n"
            f"  Колонки: {col_desc}\n"
            f"  Join keys (общие с train/test): {join_keys}"
        )
    return "\n".join(parts)


def run(state: AgentState) -> dict:
    schema = state["schema_info"]
    llm = GigaChat(
        model="GigaChat-2-Max",
        temperature=0.7,
        max_tokens=1500,
        verify_ssl_certs=False,
        profanity_check=False,
        scope="GIGACHAT_API_CORP",
        timeout=120,
    )

    columns_with_dtypes = json.dumps(schema["column_dtypes"], ensure_ascii=False)
    extra_tables_info = _build_extra_tables_info(schema)
    feature_ideas = []

    for round_number in (1, 2):
        round1_names = []
        if round_number == 2 and feature_ideas:
            round1_names = [idea["name"] for idea in feature_ideas[0]]

        round2_hint = ""
        if round_number == 2:
            round2_hint = f"В раунде 2 придумай ДРУГИЕ признаки, не повторяй: {round1_names}"

        basic_stats = json.dumps(schema.get("basic_stats", {}), ensure_ascii=False, indent=2)
        sample_rows = json.dumps(schema.get("sample_rows", []), ensure_ascii=False, indent=2)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            readme_text=schema["readme_text"],
            target_column=schema["target_column"],
            id_column=schema["id_column"],
            columns_with_dtypes=columns_with_dtypes,
            train_shape=schema["train_shape"],
            basic_stats=basic_stats,
            sample_rows=sample_rows,
            extra_tables_info=extra_tables_info,
            reserved_names=", ".join(schema["reserved_names"]),
            round_number=round_number,
            round2_hint=round2_hint,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = llm.invoke(messages)
            text = response.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            ideas = json.loads(text)
            feature_ideas.append(ideas)
        except Exception as e:
            state["errors_log"].append(f"FeatureIdeator round {round_number}: {e}")
            feature_ideas.append([])

    return {"feature_ideas": feature_ideas}
