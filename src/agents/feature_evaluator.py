"""FeatureEvaluator agent — scores feature sets and picks the winner."""
import json

from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import AgentState
from src.utils.scoring import evaluate_features

DECISION_PROMPT_TEMPLATE = """\
<role>
Ты — ML-инженер. Выбери лучший набор признаков строго по правилам.
</role>
<cv_results>
Набор 1 ROC-AUC: {score1}
Набор 2 ROC-AUC: {score2}
Признаки набора 1: {names1}
Признаки набора 2: {names2}
</cv_results>
<decision_rules>
1. Если |score1 - score2| > 0.005 → выбирай набор с большим score
2. Если |score1 - score2| <= 0.005 → выбирай набор 1 (tie-breaking)
3. Если оба score < 0.51 → выбирай лучший из двух, но добавь warning
</decision_rules>
<output_schema>
Верни ТОЛЬКО валидный JSON без markdown-обёртки:
{{"winner": 1, "reason": "одно предложение", "warning": null}}
</output_schema>"""


def run(state: AgentState) -> dict:
    schema = state["schema_info"]
    id_column = schema["id_column"]
    target_column = schema["target_column"]

    cv_scores = []
    feature_names_per_set = []

    for idx, df_train_out in enumerate(state["computed_train_dfs"]):
        if df_train_out is None:
            cv_scores.append(0.0)
            feature_names_per_set.append([])
            continue

        feature_cols = [c for c in df_train_out.columns if c not in (id_column, target_column)]
        feature_names_per_set.append(feature_cols)

        try:
            score = evaluate_features(df_train_out, feature_cols, target_column)
            cv_scores.append(score)
        except Exception as e:
            state["errors_log"].append(f"FeatureEvaluator set {idx + 1}: {e}")
            cv_scores.append(0.0)

    if all(s == 0.0 for s in cv_scores):
        state["errors_log"].append("CRITICAL: all feature sets failed evaluation")
        return {"cv_scores": cv_scores, "best_set_idx": 0}

    # Format scores for prompt (handle None-equivalent 0.0 for failed sets)
    score1_str = f"{cv_scores[0]:.4f}" if cv_scores[0] > 0.0 else "N/A (failed)"
    score2_str = f"{cv_scores[1]:.4f}" if len(cv_scores) > 1 and cv_scores[1] > 0.0 else "N/A (failed)"
    names1 = feature_names_per_set[0] if feature_names_per_set else []
    names2 = feature_names_per_set[1] if len(feature_names_per_set) > 1 else []

    llm = GigaChat(
        model="GigaChat-2-Max",
        temperature=0.0,
        max_tokens=200,
        verify_ssl_certs=False,
        timeout=120,
    )

    decision_prompt = DECISION_PROMPT_TEMPLATE.format(
        score1=score1_str,
        score2=score2_str,
        names1=names1,
        names2=names2,
    )

    messages = [
        SystemMessage(content="Ты — автономный ИИ-агент для feature engineering."),
        HumanMessage(content=decision_prompt),
    ]

    try:
        response = llm.invoke(messages)
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        result = json.loads(text)
        best_set_idx = result["winner"] - 1
    except Exception as e:
        state["errors_log"].append(f"FeatureEvaluator decision parse: {e}")
        best_set_idx = int(cv_scores.index(max(cv_scores)))

    return {"cv_scores": cv_scores, "best_set_idx": best_set_idx}
