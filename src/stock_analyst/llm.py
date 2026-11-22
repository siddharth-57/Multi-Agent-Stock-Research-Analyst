"""
llm.py: The single place that creates the chat model.

Every agent calls get_llm() instead of building a model itself, so the
provider and model can be switched from .env without touching agent code.
Agents can request a thinking level: low for simple tool-and-summarize
work, the model default (high) for complex judgment.
"""
from functools import lru_cache
from typing import Literal

from langchain_core.language_models import BaseChatModel

from stock_analyst.config import settings

ThinkingLevel = Literal["minimal", "low", "medium", "high"]


@lru_cache  # one cached model per thinking level, reused across agents
def get_llm(thinking_level: ThinkingLevel | None = None) -> BaseChatModel:
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        # Imported here so this package is only needed when Gemini is selected
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is missing from .env")

        extra = {}
        if thinking_level is not None:
            # None = model default (high for Gemini 3); lower = faster and cheaper
            extra["thinking_level"] = thinking_level

        # No temperature set on purpose: Gemini 3 models work best at the default
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            max_retries=2,  # retry brief network errors and rate limits
            **extra,
        )

    # The Groq fallback is added in a later step
    raise ValueError(f"Unsupported LLM provider: {provider}")