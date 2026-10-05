"""
Fixed evaluation script -- resolves the 3 issues causing the notebook to hang/fail.

Run with:
    .venv\Scripts\python scripts\run_evaluation_fast.py
"""

from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT
from src.generation.chains import run_baseline
from src.graph.runner import run_rag
from src.indexing.vector_db import load_vector_store
from tests.evaluation.metrics import run_eval_suite
from tests.evaluation.questions import SAMPLE_QUESTIONS


def main() -> None:
    questions = SAMPLE_QUESTIONS
    assert len(questions) >= 20, f"Need >=20 questions, got {len(questions)}"

    traces = PROJECT_ROOT / "data" / "traces"
    traces.mkdir(parents=True, exist_ok=True)

    # FIX 1: Pre-load the vector store ONCE.
    # HuggingFace all-MiniLM-L6-v2 takes ~30-120s to load on first call.
    # After this call, load_vector_store() returns the cached Chroma instance instantly.
    print("Loading vector store (one-time warm-up)...")
    _vs = load_vector_store()
    print("Vector store ready.")

    print(f"\nRunning LangGraph eval on {len(questions)} questions...")
    df_graph = run_eval_suite(
        questions,
        lambda q: run_rag(q, vector_store=_vs),
        "langgraph",
        pause_seconds=6.0,                   # FIX 2: more headroom for Gemini free-tier
        checkpoint_path=traces / "eval_langgraph_checkpoint.csv",
    )

    print(f"\nRunning baseline eval on {len(questions)} questions...")
    df_base = run_eval_suite(
        questions,
        lambda q: run_baseline(q, vector_store=_vs),
        "baseline",
        pause_seconds=5.0,
        checkpoint_path=traces / "eval_baseline_checkpoint.csv",
    )

    df = pd.concat([df_graph, df_base], ignore_index=True)
    out = traces / "eval_results.csv"
    df.to_csv(out, index=False)

    # FIX 3: fillna so NaN cells do not count as failures
    fails = int((df["error"].fillna("").astype(str).str.len() > 0).sum())
    print(f"\nDone. Saved {out}")
    print(f"Rows={len(df)}  failures={fails}")


if __name__ == "__main__":
    main()
