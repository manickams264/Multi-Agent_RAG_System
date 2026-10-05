"""Embedding model factory (local by default; Gemini or OpenAI optional)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from src.config import Settings, get_settings


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()

    # Free local embeddings -- recommended when using Gemini free tier
    if settings.use_local_embeddings:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.local_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    provider = (settings.llm_provider or "gemini").lower()

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for Gemini embeddings.")
        return GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.google_api_key,
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_api_base,
    )
