"""Evaluation helpers for Task 5 (baseline vs LangGraph comparison)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ACCURACY_LABELS = (
    "correct",
    "partially_correct",
    "incorrect",
    "hallucinated",
    "unable_to_answer",
)


def empty_eval_row(question: str) -> dict[str, Any]:
    return {
        "question": question,
        "system": "",
        "answer": "",
        "accuracy": "",
        "retry_triggered": False,
        "retrieval_retries": 0,
        "generation_retries": 0,
        "hallucination_flagged": False,
        "latency_seconds": 0.0,
        "error": "",
    }


def run_eval_suite(
    questions: list[str],
    run_fn: Callable[[str], dict[str, Any]],
    system_name: str,
    pause_seconds: float = 3.0,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Run questions through a system callable with:
    - per-question error handling (one failure does not stop the suite)
    - resume capability from checkpoint CSV (skips questions already completed)
    - pause between questions (helps Gemini free-tier rate limits)
    - immediate stdout flushing (live visibility in Jupyter/terminal)
    - CSV checkpoint saved after each question
    """
    rows: list[dict[str, Any]] = []
    completed_questions: set[str] = set()

    # Load existing checkpoint if available
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.exists():
            try:
                existing_df = pd.read_csv(path)
                # Keep rows matching system_name that have a non-empty answer or failure recorded
                matched = existing_df[existing_df["system"] == system_name]
                for _, row in matched.iterrows():
                    q_str = str(row["question"])
                    completed_questions.add(q_str)
                    rows.append(row.to_dict())
                if completed_questions:
                    print(
                        f"[{system_name}] Resuming from checkpoint '{path.name}'. "
                        f"Already completed {len(completed_questions)} / {len(questions)} questions.",
                        flush=True,
                    )
            except Exception as e:
                print(f"[{system_name}] Warning loading checkpoint ({e}). Starting fresh.", flush=True)

    total = len(questions)

    for i, q in enumerate(questions, start=1):
        if q in completed_questions:
            print(f"[{system_name}] ({i}/{total}) [SKIP - Already Completed] {q[:60]}...", flush=True)
            continue

        print(f"[{system_name}] ({i}/{total}) Running: {q[:70]}...", flush=True)
        t0 = time.perf_counter()
        try:
            out = run_fn(q)
            latency = out.get("latency_seconds")
            if latency is None:
                latency = round(time.perf_counter() - t0, 3)

            retrieval_retries = int(out.get("retrieval_retry_count") or 0)
            generation_retries = int(out.get("generation_retry_count") or 0)
            ans = out.get("final_answer") or out.get("answer", "")
            rows.append(
                {
                    "question": q,
                    "system": system_name,
                    "answer": ans,
                    "accuracy": "",
                    "retry_triggered": (retrieval_retries + generation_retries) > 0,
                    "retrieval_retries": retrieval_retries,
                    "generation_retries": generation_retries,
                    "hallucination_flagged": bool(out.get("is_hallucinated", False)),
                    "latency_seconds": latency,
                    "error": "",
                }
            )
            print(
                f"  -> OK ({latency:.1f}s) | retries: r={retrieval_retries}, g={generation_retries}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - keep suite running
            latency = round(time.perf_counter() - t0, 3)
            rows.append(
                {
                    "question": q,
                    "system": system_name,
                    "answer": "",
                    "accuracy": "unable_to_answer",
                    "retry_triggered": False,
                    "retrieval_retries": 0,
                    "generation_retries": 0,
                    "hallucination_flagged": False,
                    "latency_seconds": latency,
                    "error": str(exc)[:500],
                }
            )
            print(f"  -> FAIL ({latency:.1f}s) | {exc}", flush=True)

        completed_questions.add(q)

        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(path, index=False)

        if i < total and pause_seconds > 0:
            time.sleep(pause_seconds)

    final_df = pd.DataFrame(rows)
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(path, index=False)
        print(f"[{system_name}] Checkpoint updated at: {path}", flush=True)

    return final_df


def summarize_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate correctness and hallucination rate by system (after labeling)."""
    if df.empty or "accuracy" not in df.columns:
        return pd.DataFrame()

    labeled = df[df["accuracy"].astype(str).str.len() > 0].copy()
    if labeled.empty:
        return pd.DataFrame(
            {
                "note": [
                    "Fill the accuracy column, then re-run summarize_comparison."
                ]
            }
        )

    summary = (
        labeled.groupby("system")
        .agg(
            n=("question", "count"),
            correct_rate=("accuracy", lambda s: (s == "correct").mean()),
            hallucination_rate=(
                "accuracy",
                lambda s: (s == "hallucinated").mean(),
            ),
            flagged_rate=("hallucination_flagged", "mean"),
            avg_latency=("latency_seconds", "mean"),
            retry_rate=("retry_triggered", "mean"),
        )
        .reset_index()
    )
    return summary
