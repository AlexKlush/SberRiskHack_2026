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
1. Используй ТОЛЬКО столбцы из предоставленной схемы
2. Не используй target_column как входной признак (data leakage)
3. Не используй id_column как признак
4. Все признаки должны вычисляться идентично на train и test
5. Верни ТОЛЬКО валидный JSON, без markdown-обёртки и пояснений
</operating_principles>"""

USER_PROMPT_TEMPLATE = """\
<context>
<readme>{readme_text}</readme>
<schema>
target_column: {target_column}
id_column: {id_column}
available_columns: {columns_with_dtypes}
train_shape: {train_shape}
</schema>
<forbidden_feature_names>
Следующие имена уже заняты существующими колонками — НЕ используй их как имена признаков:
{reserved_names}
</forbidden_feature_names>
<round>{round_number} из 2</round>
</context>
<task>
Придумай ровно 5 идей признаков для данного датасета.
{round2_hint}
</task>
<feature_categories>
Рассмотри следующие типы (выбери наиболее подходящие для данных):
- AGGREGATIONS: группировка по категориальным столбцам (mean, std, count)
- INTERACTIONS: произведения, отношения, разности числовых столбцов
- DATE_FEATURES: день недели, месяц, дни от референсной точки (если есть даты)
- RANK_FEATURES: ранги значений внутри групп
- STATISTICAL: z-score, отклонение от медианы группы, квантильные бины
- MISSING_FLAGS: флаги пропусков как отдельные признаки
</feature_categories>
<thinking>
Перед ответом коротко обдумай:
- Какие столбцы выглядят наиболее предиктивными?
- Какие взаимодействия между столбцами могут быть важны?
- Несут ли пропуски сами по себе информацию?
</thinking>
<few_shot_example>
Хорошая идея:
{{"name": "amount_to_merchant_avg", "description": "Отношение суммы транзакции к средней сумме по merchant_id", "columns_used": ["amount", "merchant_id"], "category": "AGGREGATIONS", "hypothesis": "Аномально высокие суммы для мерчанта могут указывать на фрод"}}
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
    "category": "AGGREGATIONS|INTERACTIONS|DATE_FEATURES|RANK_FEATURES|STATISTICAL|MISSING_FLAGS",
    "hypothesis": "почему полезен для таргета (1 предложение)"
  }}
]
</output_schema>"""


def run(state: AgentState) -> dict:
    schema = state["schema_info"]
    llm = GigaChat(
        model="GigaChat-2-Max",
        temperature=0.7,
        max_tokens=1500,
        verify_ssl_certs=False,
        timeout=120,
    )

    columns_with_dtypes = json.dumps(schema["column_dtypes"], ensure_ascii=False)
    feature_ideas = []

    for round_number in (1, 2):
        round1_names = []
        if round_number == 2 and feature_ideas:
            round1_names = [idea["name"] for idea in feature_ideas[0]]

        round2_hint = ""
        if round_number == 2:
            round2_hint = f"В раунде 2 придумай ДРУГИЕ признаки, не повторяй: {round1_names}"

        user_prompt = USER_PROMPT_TEMPLATE.format(
            readme_text=schema["readme_text"],
            target_column=schema["target_column"],
            id_column=schema["id_column"],
            columns_with_dtypes=columns_with_dtypes,
            train_shape=schema["train_shape"],
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
            # Strip markdown code fences if present
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
