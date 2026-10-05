"""Optional reranker — score/filter chunks after initial retrieval."""

from __future__ import annotations

from typing import Any


def simple_score_rerank(
    chunks: list[dict[str, Any]], top_n: int | None = None
) -> list[dict[str, Any]]:
    """Rerank by existing similarity score (placeholder for a cross-encoder)."""
    ranked = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
    if top_n is not None:
        return ranked[:top_n]
    return ranked
