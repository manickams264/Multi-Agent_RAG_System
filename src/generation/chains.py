"""Baseline retrieve → generate chain (no graph logic) for Task 5 comparison."""

from __future__ import annotations

import time
from typing import Any

from langchain_chroma import Chroma

from src.config import Settings, get_settings
from src.generation.llm import get_llm
from src.generation.prompts import BASELINE_ANSWER_PROMPT
from src.graph.utils import message_to_text
from src.retrieval.retriever import retrieve_with_scores


def run_baseline(
    query: str,
    vector_store: Chroma | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Simple RAG: embed query → top-k retrieve → generate. No loops/gates."""
    settings = settings or get_settings()
    llm = get_llm(settings)
    t0 = time.perf_counter()

    chunks = retrieve_with_scores(query, vector_store=vector_store, settings=settings)
    context = "\n\n".join(
        f"[{c.get('source', 'unknown')}] {c['content']}" for c in chunks
    )
    prompt = BASELINE_ANSWER_PROMPT.format(query=query, context=context or "(empty)")
    answer = message_to_text(llm.invoke(prompt).content)
    latency = time.perf_counter() - t0

    return {
        "query": query,
        "answer": answer,
        "chunks": chunks,
        "latency_seconds": round(latency, 3),
        "system": "baseline",
    }
