"""Central configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLMProvider = Literal["gemini", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider: gemini (default) or openai
    llm_provider: LLMProvider = Field(default="gemini", alias="LLM_PROVIDER")

    # Google Gemini
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_chat_model: str = Field(
        default="gemini-3.5-flash-lite", alias="GEMINI_CHAT_MODEL"
    )
    gemini_fallback_models: str = Field(
        default="gemini-3.8-flash,gemini-2.5-flash",
        alias="GEMINI_FALLBACK_MODELS",
    )
    gemini_embedding_model: str = Field(
        default="models/text-embedding-004", alias="GEMINI_EMBEDDING_MODEL"
    )

   

    # Embeddings — local by default (free, no API key)
    use_local_embeddings: bool = Field(default=True, alias="USE_LOCAL_EMBEDDINGS")
    local_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="LOCAL_EMBEDDING_MODEL",
    )

    # RAG / graph
    top_k: int = Field(default=5, alias="TOP_K")
    relevance_threshold: float = Field(default=0.5, alias="RELEVANCE_THRESHOLD")
    max_retrieval_retries: int = Field(default=2, alias="MAX_RETRIEVAL_RETRIES")
    max_generation_retries: int = Field(default=2, alias="MAX_GENERATION_RETRIES")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    # If false, relevance grader uses vector scores only (faster, fewer API calls)
    use_llm_relevance_grader: bool = Field(
        default=True, alias="USE_LLM_RELEVANCE_GRADER"
    )
    # Max chunks to LLM-grade when use_llm_relevance_grader is true
    relevance_grade_top_n: int = Field(default=2, alias="RELEVANCE_GRADE_TOP_N")

    # Paths
    data_raw_dir: str = Field(default="data/raw", alias="DATA_RAW_DIR")
    data_processed_dir: str = Field(default="data/processed", alias="DATA_PROCESSED_DIR")
    vector_store_dir: str = Field(default="data/vector_store", alias="VECTOR_STORE_DIR")
    traces_dir: str = Field(default="data/traces", alias="TRACES_DIR")
    checkpoint_db: str = Field(
        default="data/traces/checkpoints.sqlite", alias="CHECKPOINT_DB"
    )

    @property
    def raw_path(self) -> Path:
        return PROJECT_ROOT / self.data_raw_dir

    @property
    def processed_path(self) -> Path:
        return PROJECT_ROOT / self.data_processed_dir

    @property
    def vector_store_path(self) -> Path:
        return PROJECT_ROOT / self.vector_store_dir

    @property
    def traces_path(self) -> Path:
        return PROJECT_ROOT / self.traces_dir

    @property
    def checkpoint_db_path(self) -> Path:
        return PROJECT_ROOT / self.checkpoint_db


def get_settings() -> Settings:
    return Settings()
