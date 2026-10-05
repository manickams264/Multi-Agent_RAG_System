"""Run Task 5 evaluation from the command line (more reliable than a long notebook cell)."""

from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT
from src.generation.chains import run_baseline
from src.graph.runner import run_rag
from tests.evaluation.metrics import run_eval_suite
from tests.evaluation.questions import SAMPLE_QUESTIONS


def main() -> None:
    questions = SAMPLE_QUESTIONS
    assert len(questions) >= 20
    traces = PROJECT_ROOT / "data" / "traces"
    traces.mkdir(parents=True, exist_ok=True)

    print(f"Running LangGraph eval on {len(questions)} questions...")
    df_graph = run_eval_suite(
        questions,
        lambda q: run_rag(q),
        "langgraph",
        pause_seconds=4.0,
        checkpoint_path=traces / "eval_langgraph_checkpoint.csv",
    )

    print(f"\nRunning baseline eval on {len(questions)} questions...")
    df_base = run_eval_suite(
        questions,
        lambda q: run_baseline(q),
        "baseline",
        pause_seconds=3.0,
        checkpoint_path=traces / "eval_baseline_checkpoint.csv",
    )

    df = pd.concat([df_graph, df_base], ignore_index=True)
    out = traces / "eval_results.csv"
    df.to_csv(out, index=False)
    fails = int((df["error"].astype(str).str.len() > 0).sum())
    print(f"\nDone. Saved {out}")
    print(f"Rows={len(df)}  failures={fails}")


if __name__ == "__main__":
    main()
