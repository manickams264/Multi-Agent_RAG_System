"""Retrieval helper tests that do not require a live vector store API key."""

from __future__ import annotations

from src.retrieval.reranker import simple_score_rerank


def test_simple_score_rerank_orders_by_score():
    chunks = [
        {"content": "a", "score": 0.2},
        {"content": "b", "score": 0.9},
        {"content": "c", "score": 0.5},
    ]
    ranked = simple_score_rerank(chunks, top_n=2)
    assert [c["content"] for c in ranked] == ["b", "c"]
