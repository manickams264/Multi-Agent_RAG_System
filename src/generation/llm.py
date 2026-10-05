"""LLM client factory — Gemini by default, OpenAI optional.

Includes retries for temporary Gemini free-tier overload (503 / 429)
and automatic fallback when a model ID is retired (404).
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from src.config import Settings, get_settings


def _is_transient_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "503",
        "429",
        "unavailable",
        "high demand",
        "resource_exhausted",
        "rate limit",
        "try again",
        "temporar",
    )
    return any(m in text for m in markers)


def _is_model_unavailable(exc: BaseException) -> bool:
    text = str(exc).lower()
    markers = (
        "404",
        "not_found",
        "no longer available",
        "not found",
        "googlemodelnotfounderror",
    )
    return any(m in text for m in markers)


class ResilientChatModel:
    """Retry transient failures and fall back across Gemini models."""

    def __init__(
        self,
        models: list[str],
        api_key: str,
        temperature: float = 0.0,
        max_attempts_per_model: int = 4,
        base_delay_seconds: float = 2.0,
    ) -> None:
        self.models = [m.strip() for m in models if m and m.strip()]
        self.api_key = api_key
        self.temperature = temperature
        self.max_attempts_per_model = max_attempts_per_model
        self.base_delay_seconds = base_delay_seconds
        self._clients: dict[str, BaseChatModel] = {}

    def _client_for(self, model: str) -> BaseChatModel:
        if model not in self._clients:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._clients[model] = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=self.api_key,
                temperature=self.temperature,
                max_retries=0,
            )
        return self._clients[model]

    def invoke(self, input: Any, config: Any | None = None, **kwargs: Any) -> AIMessage:
        last_error: BaseException | None = None
        for model in self.models:
            client = self._client_for(model)
            for attempt in range(1, self.max_attempts_per_model + 1):
                try:
                    return client.invoke(input, config=config, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if _is_model_unavailable(exc):
                        print(
                            f"[llm] Model {model} not available. Trying next fallback...",
                            flush=True,
                        )
                        break  # next model
                    if not _is_transient_error(exc):
                        raise
                    delay = self.base_delay_seconds * (2 ** (attempt - 1))
                    print(
                        f"[llm] {model} busy/unavailable "
                        f"(attempt {attempt}/{self.max_attempts_per_model}). "
                        f"Retrying in {delay:.0f}s...",
                        flush=True,
                    )
                    time.sleep(delay)
            else:
                print(
                    f"[llm] Giving up on {model}; trying next fallback model...",
                    flush=True,
                )

        raise RuntimeError(
            "All Gemini models failed (overloaded or unavailable). "
            "Wait 1–2 minutes and retry, or change GEMINI_CHAT_MODEL in .env "
            "(try gemini-3.5-flash-lite or gemini-3.8-flash)."
        ) from last_error


def get_llm(
    settings: Settings | None = None,
    temperature: float = 0.0,
) -> BaseChatModel | ResilientChatModel:
    settings = settings or get_settings()
    provider = (settings.llm_provider or "gemini").lower()

    if provider == "gemini":
        if not settings.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/apikey "
                "and add it to your .env file."
            )
        primary = settings.gemini_chat_model
        fallbacks = [
            m.strip()
            for m in (settings.gemini_fallback_models or "").split(",")
            if m.strip() and m.strip() != primary
        ]
        models = [primary, *fallbacks]
        return ResilientChatModel(
            models=models,
            api_key=settings.google_api_key,
            temperature=temperature,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return ChatOpenAI(
            model=settings.openai_chat_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER={settings.llm_provider!r}. Use 'gemini' or 'openai'."
    )


def ensure_ai_message(msg: Any) -> BaseMessage:
    return msg
