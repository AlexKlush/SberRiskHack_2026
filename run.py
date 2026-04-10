"""Entry point for the feature-agent pipeline."""
import sys
import time
import traceback
import os
import signal

from dotenv import load_dotenv

load_dotenv()
os.makedirs("output", exist_ok=True)

MAX_SECONDS = 570


def _timeout(signum, frame):
    raise TimeoutError(f"Agent exceeded {MAX_SECONDS}s limit")


if sys.platform != "win32":
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(MAX_SECONDS)

from src.graph import build_graph
from src.state import AgentState


def main():
    start = time.time()
    graph = build_graph()
    initial_state: AgentState = {
        "schema_info": {},
        "df_train": None,
        "df_test": None,
        "extra_tables": {},
        "feature_ideas": [],
        "generated_code": [],
        "computed_train_dfs": [],
        "computed_test_dfs": [],
        "cv_scores": [],
        "best_set_idx": 0,
        "errors_log": [],
    }
    graph.invoke(initial_state)
    if sys.platform != "win32":
        signal.alarm(0)
    print(f" \u2713 Done in {time.time()-start:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
