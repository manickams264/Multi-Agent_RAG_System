"""Conditional routing functions for LangGraph edges."""

from __future__ import annotations

from typing import Literal

from src.graph.state import GraphState


def route_after_relevance(
    state: GraphState,
) -> Literal["retriever", "answer_generator"]:
    """
    Loop back to Retriever if quality is low and retries remain;
    otherwise proceed to Answer Generator.
    """
    if state.get("retrieval_quality_ok", True):
        return "answer_generator"
    return "retriever"


def route_after_hallucination(
    state: GraphState,
) -> Literal["answer_generator", "end"]:
    """
    Regenerate if hallucinations flagged and retries remain;
    otherwise finish with final_answer.
    """
    if state.get("final_answer"):
        return "end"
    if state.get("is_hallucinated") and not state.get("final_answer"):
        return "answer_generator"
    return "end"
