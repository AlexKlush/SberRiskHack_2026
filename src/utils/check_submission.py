"""Скрипт проверки сабмита перед отправкой (13 проверок)."""
import sys
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).resolve().parent.parent.parent
    passed = 0
    failed = 0

    def check(condition: bool, msg: str):
        nonlocal passed, failed
        if condition:
            print(f"  [OK] {msg}")
            passed += 1
        else:
            print(f"  [FAIL] {msg}")
            failed += 1

    env_path = root / ".env"
    check(env_path.exists(), ".env file exists")

    if env_path.exists():
        env_text = env_path.read_text(encoding="utf-8")
        check("GIGACHAT_CREDENTIALS" in env_text and "GIGACHAT_SCOPE" in env_text,
              ".env contains GIGACHAT_CREDENTIALS and GIGACHAT_SCOPE")
    else:
        check(False, ".env contains GIGACHAT_CREDENTIALS and GIGACHAT_SCOPE")

    train_path = root / "output" / "train.csv"
    test_path = root / "output" / "test.csv"

    check(train_path.exists(), "output/train.csv exists")
    check(test_path.exists(), "output/test.csv exists")

    df_train = None
    df_test = None

    if train_path.exists():
        try:
            df_train = pd.read_csv(train_path)
            check(True, "output/train.csv is readable as DataFrame")
        except Exception:
            check(False, "output/train.csv is readable as DataFrame")
    else:
        check(False, "output/train.csv is readable as DataFrame")

    if test_path.exists():
        try:
            df_test = pd.read_csv(test_path)
            check(True, "output/test.csv is readable as DataFrame")
        except Exception:
            check(False, "output/test.csv is readable as DataFrame")
    else:
        check(False, "output/test.csv is readable as DataFrame")

    if df_train is not None and df_test is not None:
        train_cols = list(df_train.columns)
        test_cols = list(df_test.columns)

        has_id_target = len(train_cols) >= 2
        check(has_id_target, "train.csv: первый столбец — ID, второй — target, затем 1-5 фичей")

        has_id = len(test_cols) >= 1
        check(has_id, "test.csv: первый столбец — ID, затем 1-5 фичей (без target)")

        if has_id_target and has_id:
            train_features = train_cols[2:]
            test_features = test_cols[1:]
            check(train_features == test_features,
                  "Названия фичей совпадают в train и test")

            no_nan_train = not df_train[train_features].isna().any().any() if train_features else True
            no_nan_test = not df_test[test_features].isna().any().any() if test_features else True
            check(no_nan_train and no_nan_test, "Нет NaN в столбцах фичей")

            check(not df_train[train_cols[0]].duplicated().any(), "Нет дублей ID в train")
            check(not df_test[test_cols[0]].duplicated().any(), "Нет дублей ID в test")

            n_features = len(train_features)
            check(1 <= n_features <= 5, f"Количество фичей ({n_features}) от 1 до 5")
        else:
            for _ in range(5):
                check(False, "(пропущено — структура столбцов некорректна)")
    else:
        for _ in range(7):
            check(False, "(пропущено — CSV не читается)")

    print()
    if failed == 0:
        print("Все проверки пройдены! Готово к отправке.")
        sys.exit(0)
    else:
        print(f"{failed} проверок не пройдено.")
        sys.exit(1)


if __name__ == "__main__":
    main()
