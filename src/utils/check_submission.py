"""Pre-submit validation script (13 checks)."""
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
            print(f"  \u2713 {msg}")
            passed += 1
        else:
            print(f"  \u2717 FAIL: {msg}")
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
        check(has_id_target, "train.csv: first column is ID, second is target, then 1-5 features")

        has_id = len(test_cols) >= 1
        check(has_id, "test.csv: first column is ID, then 1-5 features (no target)")

        if has_id_target and has_id:
            train_features = train_cols[2:]
            test_features = test_cols[1:]
            check(train_features == test_features,
                  "Feature names match in train and test")

            no_nan_train = not df_train[train_features].isna().any().any() if train_features else True
            no_nan_test = not df_test[test_features].isna().any().any() if test_features else True
            check(no_nan_train and no_nan_test, "No NaN in feature columns")

            check(not df_train[train_cols[0]].duplicated().any(), "No duplicate IDs in train")
            check(not df_test[test_cols[0]].duplicated().any(), "No duplicate IDs in test")

            n_features = len(train_features)
            check(1 <= n_features <= 5, f"Number of features ({n_features}) is between 1 and 5")
        else:
            for _ in range(5):
                check(False, "(skipped — column structure invalid)")
    else:
        for _ in range(7):
            check(False, "(skipped — CSV not readable)")

    print()
    if failed == 0:
        print("All checks passed! Ready to submit.")
        sys.exit(0)
    else:
        print(f"{failed} checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
