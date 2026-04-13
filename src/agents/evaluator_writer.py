"""Агент EvaluatorWriter — отбирает лучшие фичи, оценивает, сохраняет результат."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.state import AgentState
from src.utils.scoring import evaluate_features, forward_select_features

OUTPUT_DIR = Path("output")


def run(state: AgentState) -> dict:
    schema = state["schema_info"]
    id_col = schema["id_column"]
    target_col = schema["target_column"]

    # Робастное определение ID-колонки в test
    if id_col not in state["df_test"].columns:
        test_id_col = state["df_test"].columns[0]
    else:
        test_id_col = id_col

    train_df = state["candidate_features_train"]
    test_df = state["candidate_features_test"]
    candidates = list(state["candidate_names"])

    # Если кандидатов нет — ОШИБКА (не сохраняем мусор)
    if not candidates or train_df is None or test_df is None:
        raise ValueError("CRITICAL: no candidate features generated. Refusing to save empty output.")

    # Заполняем оставшиеся NaN и inf
    train_df[candidates] = train_df[candidates].fillna(0)
    test_df[candidates] = test_df[candidates].fillna(0)
    for c in candidates:
        train_df[c] = np.nan_to_num(train_df[c], nan=0.0, posinf=0.0, neginf=0.0)
        test_df[c] = np.nan_to_num(test_df[c], nan=0.0, posinf=0.0, neginf=0.0)

    # Отбираем лучшие фичи (до 5)
    selected = forward_select_features(
        train_df, candidates, target_col, max_features=5,
    )

    # Фоллбэк: если forward selection ничего не выбрал — берём первые 5
    if not selected:
        state["errors_log"].append("Forward selection returned 0 features, taking first 5")
        selected = candidates[:5]

    # Финальная 5-fold оценка
    final_score = evaluate_features(train_df, selected, target_col)

    # ===== ПОРОГ КАЧЕСТВА: не сохраняем мусор =====
    MIN_ROC_AUC = 0.55
    if final_score < MIN_ROC_AUC:
        raise ValueError(
            f"ROC-AUC {final_score:.4f} < {MIN_ROC_AUC}. "
            f"Результат ненадёжный, output НЕ сохранён. "
            f"Features: {selected}"
        )

    # Формируем результат
    train_out = train_df[[id_col, target_col] + selected].copy()
    test_id = id_col if id_col in test_df.columns else test_df.columns[0]
    test_out = test_df[[test_id] + selected].copy()
    if test_id != id_col:
        test_out = test_out.rename(columns={test_id: id_col})
    train_out[selected] = train_out[selected].fillna(0)
    test_out[selected] = test_out[selected].fillna(0)

    # ===== ВАЛИДАЦИЯ OUTPUT перед сохранением =====
    _validate_output(train_out, test_out, state["df_test"], id_col, target_col, selected)

    _save(train_out, test_out, selected, final_score)

    return {"selected_features": selected, "cv_score": final_score}


def _validate_output(train_out, test_out, df_test_original, id_col, target_col, features):
    """Финальная проверка output перед записью. Падает при проблемах."""
    errors = []

    # 1. Количество фичей 1-5
    if not (1 <= len(features) <= 5):
        errors.append(f"Количество фичей {len(features)}, нужно 1-5")

    # 2. Нет NaN в фичах
    for c in features:
        if train_out[c].isna().any():
            errors.append(f"NaN в train фиче '{c}'")
        if test_out[c].isna().any():
            errors.append(f"NaN в test фиче '{c}'")

    # 3. Нет дублей ID
    if train_out[id_col].duplicated().any():
        errors.append(f"Дублирующиеся ID в train output")
    if test_out[id_col].duplicated().any():
        errors.append(f"Дублирующиеся ID в test output")

    # 4. Длина test output == длина исходного test
    if len(test_out) != len(df_test_original):
        errors.append(
            f"Длина test output ({len(test_out)}) != исходный test ({len(df_test_original)})"
        )

    # 5. ID значения в test output должны совпадать с исходным test
    if id_col in df_test_original.columns:
        original_ids = set(df_test_original[id_col].values)
        output_ids = set(test_out[id_col].values)
        if original_ids != output_ids:
            missing = len(original_ids - output_ids)
            extra = len(output_ids - original_ids)
            errors.append(
                f"ID в test output не совпадают с исходным test: "
                f"{missing} пропущено, {extra} лишних"
            )

    # 6. Названия фичей совпадают в train и test output
    train_features = list(train_out.columns[2:])  # skip id, target
    test_features = list(test_out.columns[1:])     # skip id
    if train_features != test_features:
        errors.append(
            f"Фичи не совпадают: train={train_features}, test={test_features}"
        )

    if errors:
        msg = "OUTPUT VALIDATION FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        print(f"  [EvaluatorWriter] {msg}")
        raise ValueError(msg)

    print(f"  [EvaluatorWriter] Output validation OK: {len(features)} features, "
          f"{len(test_out)} test rows, IDs match")


def _save(train_out, test_out, features, score):
    """Сохраняем CSV и выводим итог."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_out.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print(f"\n{'=' * 50}")
    print(f"  ROC-AUC:      {score:.4f}")
    print(f"  Features ({len(features)}): {features}")
    print(f"  Train shape:  {train_out.shape}")
    print(f"  Test shape:   {test_out.shape}")
    print(f"  Saved to:     {OUTPUT_DIR / 'train.csv'}, {OUTPUT_DIR / 'test.csv'}")
    print(f"{'=' * 50}\n")
