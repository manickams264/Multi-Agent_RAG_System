"""Unit tests for graph state shape and routing (no LLM required)."""

from __future__ import annotations

from src.graph.edges import route_after_hallucination, route_after_relevance
from src.graph.state import initial_state


def test_initial_state_has_required_fields():
    state = initial_state("What is RAG?")
    assert state["original_query"] == "What is RAG?"
    assert "rewritten_queries" in state
    assert "retrieved_chunks" in state
    assert "relevance_grades" in state
    assert "answer_draft" in state
    assert "is_hallucinated" in state
    assert state["retrieval_retry_count"] == 0
    assert state["generation_retry_count"] == 0


def test_route_after_relevance_loops_when_quality_low():
    state = initial_state("q")
    state["retrieval_quality_ok"] = False
    assert route_after_relevance(state) == "retriever"


def test_route_after_relevance_proceeds_when_ok():
    state = initial_state("q")
    state["retrieval_quality_ok"] = True
    assert route_after_relevance(state) == "answer_generator"


def test_route_after_hallucination_regenerates():
    state = initial_state("q")
    state["is_hallucinated"] = True
    state["final_answer"] = ""
    assert route_after_hallucination(state) == "answer_generator"


def test_route_after_hallucination_ends_when_clean():
    state = initial_state("q")
    state["is_hallucinated"] = False
    state["final_answer"] = "Grounded answer"
    assert route_after_hallucination(state) == "end"
