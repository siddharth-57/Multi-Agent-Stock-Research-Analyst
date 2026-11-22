"""
test_llm.py: Unit tests for the LLM factory.

Checks that a fallback model is attached when one is configured, and that
the plain model is returned when it isn't. No API calls are made.
"""
import pytest
from langchain_core.runnables.fallbacks import RunnableWithFallbacks

from stock_analyst.config import settings
from stock_analyst.llm import get_llm


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", "fake-key")
    get_llm.cache_clear()   # don't reuse models cached by other tests
    yield
    get_llm.cache_clear()


def test_fallback_attached_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_model", "gemini-3.1-flash-lite")
    llm = get_llm("low")
    assert isinstance(llm, RunnableWithFallbacks)
    assert len(llm.fallbacks) == 1


def test_plain_model_when_no_fallback(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_model", None)
    assert not isinstance(get_llm("low"), RunnableWithFallbacks)