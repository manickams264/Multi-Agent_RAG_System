"""Run the compiled graph, save state traces, and expose a simple API helper."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma

from src.config import Settings, get_settings
from src.graph.builder import build_graph
from src.graph.state import GraphState, initial_state


def run_rag(
    query: str,
    vector_store: Chroma | None = None,
    settings: Settings | None = None,
    use_multi_query: bool = True,
    thread_id: str | None = None,
    save_trace: bool = False,
    trace_name: str | None = None,
) -> dict[str, Any]:
    """Execute the full multi-agent graph for one query."""
    settings = settings or get_settings()
    thread_id = thread_id or str(uuid.uuid4())
    compiled = build_graph(
        vector_store=vector_store,
        settings=settings,
        use_multi_query=use_multi_query,
    )

    state: GraphState = initial_state(
        query,
        max_retrieval_retries=settings.max_retrieval_retries,
        max_generation_retries=settings.max_generation_retries,
    )

    t0 = time.perf_counter()
    final_state = compiled.invoke(
        state,
        config={"configurable": {"thread_id": thread_id}},
    )
    latency = time.perf_counter() - t0

    result = {
        "query": query,
        "final_answer": final_state.get("final_answer")
        or final_state.get("answer_draft", ""),
        "query_type": final_state.get("query_type"),
        "entities": final_state.get("entities"),
        "rewritten_queries": final_state.get("rewritten_queries"),
        "avg_relevance": final_state.get("avg_relevance"),
        "retrieval_retry_count": final_state.get("retrieval_retry_count", 0),
        "generation_retry_count": final_state.get("generation_retry_count", 0),
        "is_hallucinated": final_state.get("is_hallucinated", False),
        "unsupported_claims": final_state.get("unsupported_claims", []),
        "hallucination_notes": final_state.get("hallucination_notes", ""),
        "transition_log": final_state.get("transition_log", []),
        "latency_seconds": round(latency, 3),
        "thread_id": thread_id,
        "system": "langgraph",
        "full_state": dict(final_state),
    }

    if save_trace:
        path = save_state_trace(result, settings=settings, name=trace_name)
        result["trace_path"] = str(path)

    return result


def save_state_trace(
    result: dict[str, Any],
    settings: Settings | None = None,
    name: str | None = None,
) -> Path:
    """Persist a full state trace JSON (Task 4 submission artifact)."""
    settings = settings or get_settings()
    settings.traces_path.mkdir(parents=True, exist_ok=True)
    safe = (name or "state_trace").replace(" ", "_")
    path = settings.traces_path / f"{safe}.json"

    # Make JSON-serializable (drop non-serializable if any)
    payload = {
        "query": result.get("query"),
        "final_answer": result.get("final_answer"),
        "latency_seconds": result.get("latency_seconds"),
        "retrieval_retry_count": result.get("retrieval_retry_count"),
        "generation_retry_count": result.get("generation_retry_count"),
        "is_hallucinated": result.get("is_hallucinated"),
        "transition_log": result.get("transition_log", []),
        "full_state": result.get("full_state", {}),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
