"""Retriever configuration and search helpers."""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import Settings, get_settings
from src.indexing.vector_db import load_vector_store


def get_retriever(vector_store: Chroma | None = None, k: int | None = None):
    settings = get_settings()
    store = vector_store or load_vector_store(settings=settings)
    return store.as_retriever(search_kwargs={"k": k or settings.top_k})


def retrieve_with_scores(
    query: str,
    vector_store: Chroma | None = None,
    k: int | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Semantic search returning chunk text, metadata, and similarity score."""
    settings = settings or get_settings()
    store = vector_store or load_vector_store(settings=settings)
    results = store.similarity_search_with_relevance_scores(
        query, k=k or settings.top_k
    )
    chunks: list[dict[str, Any]] = []
    for doc, score in results:
        chunks.append(
            {
                "content": doc.page_content,
                "metadata": dict(doc.metadata),
                "score": float(score),
                "source": doc.metadata.get("source", "unknown"),
            }
        )
    return chunks


def retrieve_multi_query(
    queries: list[str],
    vector_store: Chroma | None = None,
    k: int | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """
    Multi-query retrieval (Task 3 enhancement):
    retrieve for each rewritten query, then merge with deduplication.
    """
    settings = settings or get_settings()
    store = vector_store or load_vector_store(settings=settings)
    k = k or settings.top_k

    merged: dict[str, dict[str, Any]] = {}
    for q in queries:
        for chunk in retrieve_with_scores(q, vector_store=store, k=k, settings=settings):
            # Dedupe key: content fingerprint
            key = chunk["content"].strip()[:500]
            existing = merged.get(key)
            if existing is None or chunk["score"] > existing["score"]:
                merged[key] = chunk

    # Highest score first, keep top_k * 2 to give grader more options
    ranked = sorted(merged.values(), key=lambda c: c["score"], reverse=True)
    return ranked[: max(k * 2, k)]


def docs_from_chunks(chunks: list[dict[str, Any]]) -> list[Document]:
    return [
        Document(page_content=c["content"], metadata=c.get("metadata", {}))
        for c in chunks
    ]
