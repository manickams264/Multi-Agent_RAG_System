"""Task 1 — Typed shared graph state. All nodes read/write only this object."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict


QueryType = Literal["factual", "multi-hop", "comparative", "unanswerable"]


class RetrievedChunk(TypedDict, total=False):
    content: str
    metadata: dict[str, Any]
    score: float
    source: str


class RelevanceGrade(TypedDict, total=False):
    chunk_index: int
    score: float
    reason: str


class GraphState(TypedDict, total=False):
    """Shared state passed between every LangGraph node."""

    # Input
    original_query: str

    # Query Analyser
    query_type: QueryType
    entities: list[str]
    analysis_rationale: str

    # Query Rewriter
    rewritten_queries: list[str]

    # Retriever
    retrieved_chunks: list[RetrievedChunk]

    # Relevance Grader
    relevance_grades: list[RelevanceGrade]
    avg_relevance: float
    retrieval_quality_ok: bool

    # Answer Generator
    answer_draft: str
    final_answer: str

    # Hallucination Checker
    is_hallucinated: bool
    unsupported_claims: list[str]
    hallucination_notes: str

    # Retry / control
    retrieval_retry_count: int
    generation_retry_count: int
    max_retrieval_retries: int
    max_generation_retries: int

    # Observability (Task 4)
    # Annotated with operator.add so each node can append without overwriting
    transition_log: Annotated[list[dict[str, Any]], operator.add]


def initial_state(
    query: str,
    max_retrieval_retries: int = 2,
    max_generation_retries: int = 2,
) -> GraphState:
    """Create a clean starting state for one run."""
    return GraphState(
        original_query=query,
        query_type="factual",
        entities=[],
        analysis_rationale="",
        rewritten_queries=[],
        retrieved_chunks=[],
        relevance_grades=[],
        avg_relevance=0.0,
        retrieval_quality_ok=False,
        answer_draft="",
        final_answer="",
        is_hallucinated=False,
        unsupported_claims=[],
        hallucination_notes="",
        retrieval_retry_count=0,
        generation_retry_count=0,
        max_retrieval_retries=max_retrieval_retries,
        max_generation_retries=max_generation_retries,
        transition_log=[],
    )
