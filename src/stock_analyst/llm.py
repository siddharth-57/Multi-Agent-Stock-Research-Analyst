"""
llm.py: The single place that creates the chat model.

Every agent calls get_llm() instead of building a model itself, so the
provider and model can be switched from .env without touching agent code.
Agents can request a thinking level. If the main model fails (e.g. a 503
overload), the same request automatically goes to a fallback model.
"""
from functools import lru_cache
from typing import Literal

from langchain_core.runnables import Runnable

from stock_analyst.config import settings

ThinkingLevel = Literal["minimal", "low", "medium", "high"]


def _gemini(model: str, thinking_level: ThinkingLevel | None, max_retries: int):
    """Build one Gemini chat model."""
    # Imported here so this package is only needed when Gemini is selected
    from langchain_google_genai import ChatGoogleGenerativeAI

    # None = model default (high for Gemini 3); lower = faster and cheaper
    extra = {"thinking_level": thinking_level} if thinking_level else {}

    # No temperature set on purpose: Gemini 3 models work best at the default
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        max_retries=max_retries,
        **extra,
    )


@lru_cache  # one cached model per thinking level, reused across agents
def get_llm(thinking_level: ThinkingLevel | None = None) -> Runnable:
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is missing from .env")

        # Few retries on the main model so a long outage fails over quickly
        primary = _gemini(settings.llm_model, thinking_level, max_retries=1)
        if not settings.llm_fallback_model:
            return primary

        fallback = _gemini(settings.llm_fallback_model, thinking_level, max_retries=2)
        # On any error from the main model, the same request is retried on the fallback.
        # bind_tools / with_structured_output still work: LangChain applies them to both.
        return primary.with_fallbacks([fallback])

    # The Groq fallback is added in phase 5
    raise ValueError(f"Unsupported LLM provider: {provider}")