"""Chroma vector store helpers — build, load, and query."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import Settings, get_settings
from src.indexing.embeddings import get_embeddings

COLLECTION_NAME = "rag_corpus"

# Module-level singleton cache — avoids reloading HuggingFace weights and
# reconnecting to Chroma on every run_rag() call during evaluation loops.
_VECTOR_STORE_CACHE: Optional[Chroma] = None


def build_vector_store(
    chunks: list[Document],
    persist_directory: Path | None = None,
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
) -> Chroma:
    """Create (or recreate) a persistent Chroma store from document chunks."""
    settings = settings or get_settings()
    persist_directory = persist_directory or settings.vector_store_path
    embeddings = embeddings or get_embeddings(settings)
    persist_directory.mkdir(parents=True, exist_ok=True)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=COLLECTION_NAME,
    )
    return store


def load_vector_store(
    persist_directory: Path | None = None,
    embeddings: Embeddings | None = None,
    settings: Settings | None = None,
) -> Chroma:
    """Load an existing persistent Chroma store.

    Results are cached at the module level so the HuggingFace embedding model
    and Chroma connection are only initialised once per Python process,
    which is the dominant latency source during evaluation loops.
    """
    global _VECTOR_STORE_CACHE

    # Return cached instance if available and no custom args override it
    if _VECTOR_STORE_CACHE is not None and persist_directory is None and embeddings is None:
        return _VECTOR_STORE_CACHE

    settings = settings or get_settings()
    persist_directory = persist_directory or settings.vector_store_path
    embeddings = embeddings or get_embeddings(settings)

    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_directory}. "
            "Run the ingestion/indexing pipeline first."
        )

    store = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    # Cache when using default args (the common evaluation path)
    if persist_directory == get_settings().vector_store_path:
        _VECTOR_STORE_CACHE = store

    return store


def clear_vector_store_cache() -> None:
    """Invalidate the module-level cache (call after re-indexing)."""
    global _VECTOR_STORE_CACHE
    _VECTOR_STORE_CACHE = None
