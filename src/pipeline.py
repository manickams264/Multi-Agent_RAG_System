"""Ingest raw documents → chunk → build persistent vector store."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import Settings, get_settings
from src.indexing.vector_db import build_vector_store
from src.ingestion.loaders import load_documents
from src.ingestion.splitters import split_documents


def ingest_and_index(settings: Settings | None = None) -> dict:
    """Full indexing pipeline. Returns summary stats."""
    settings = settings or get_settings()
    raw_dir = settings.raw_path
    processed_dir = settings.processed_path
    processed_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(raw_dir)
    if not documents:
        raise FileNotFoundError(
            f"No documents found in {raw_dir}. "
            "Add PDF/TXT/MD/HTML files before indexing."
        )

    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # Persist a lightweight processed dump for inspection
    dump_path = processed_dir / "chunks_preview.json"
    preview = [
        {
            "chunk_id": c.metadata.get("chunk_id"),
            "source": c.metadata.get("source"),
            "content": c.page_content[:500],
        }
        for c in chunks[:50]
    ]
    dump_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")

    store = build_vector_store(chunks, settings=settings)
    from src.indexing.vector_db import clear_vector_store_cache
    clear_vector_store_cache()

    return {
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "vector_store": str(settings.vector_store_path),
        "processed_preview": str(dump_path),
        "collection_count": store._collection.count()
        if hasattr(store, "_collection")
        else len(chunks),
    }


if __name__ == "__main__":
    summary = ingest_and_index()
    print(json.dumps(summary, indent=2))
