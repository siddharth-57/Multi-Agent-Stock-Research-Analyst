"""
test_news.py: Unit tests for the Finnhub news client.

Fakes Finnhub responses to check the relevance filter, duplicate removal,
the article limit, the rate-limit error, and that the key is sent in a header.
"""
import httpx
import pytest

from stock_analyst.data.news import FinnhubClient

# Newest first, like Finnhub returns them
ARTICLES = [
    {"headline": "Apple launches new iPhone", "summary": "", "source": "Reuters",
     "datetime": 1763700000, "url": "https://example.com/1"},
    # Same story, different case/spacing -> duplicate
    {"headline": "apple  launches new IPHONE", "summary": "", "source": "Yahoo",
     "datetime": 1763690000, "url": "https://example.com/2"},
    # Roundup that never mentions Apple -> filtered out
    {"headline": "5 stocks to watch", "summary": "Nvidia and Dell lead", "source": "Benzinga",
     "datetime": 1763680000, "url": "https://example.com/3"},
    # "Pineapple" contains "apple" but isn't the word Apple -> filtered out
    {"headline": "Pineapple prices soar", "summary": "", "source": "Yahoo",
     "datetime": 1763670000, "url": "https://example.com/4"},
    # Apple mentioned only in the summary -> kept
    {"headline": "Big Tech earnings preview", "summary": "Apple reports Thursday",
     "source": "CNBC", "datetime": 1763660000, "url": "https://example.com/5"},
]

requests_seen: list[httpx.Request] = []


def ok_handler(request: httpx.Request) -> httpx.Response:
    requests_seen.append(request)
    return httpx.Response(200, json=ARTICLES)


async def fetch(handler, **kwargs):
    """Run one news fetch against a fake Finnhub and always close the client."""
    client = FinnhubClient(transport=httpx.MockTransport(handler))
    try:
        return await client.get_company_news("AAPL", **kwargs)
    finally:
        await client.close()


async def test_keyword_filter_and_dedupe():
    articles = await fetch(ok_handler, keywords=["Apple", "AAPL"])
    assert [a.url for a in articles] == ["https://example.com/1", "https://example.com/5"]


async def test_without_keywords_only_dedupes():
    articles = await fetch(ok_handler)
    assert len(articles) == 4  # only the duplicate is removed


async def test_limit():
    assert len(await fetch(ok_handler, limit=1)) == 1


async def test_api_key_in_header_not_url():
    requests_seen.clear()
    await fetch(ok_handler)
    req = requests_seen[0]
    assert req.headers["X-Finnhub-Token"] == "test-key"
    assert "test-key" not in str(req.url)


async def test_rate_limit_raises_clear_error():
    with pytest.raises(RuntimeError, match="rate limit"):
        await fetch(lambda r: httpx.Response(429))