"""
news.py: Async client for Finnhub company news.

Fetches recent articles for a ticker, removes duplicate stories that were
published by several outlets, and returns the newest ones as NewsArticle
objects. The article count is capped to keep LLM prompts small.
"""
import re
from datetime import UTC, date, datetime, timedelta

import httpx

from stock_analyst.config import settings
from stock_analyst.data.models import NewsArticle

NEWS_URL = "https://finnhub.io/api/v1/company-news"


class FinnhubClient:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        # Key goes in a header, not the URL, so it never shows up in logs.
        # `transport` is only passed in tests, to fake Finnhub responses.
        self._http = httpx.AsyncClient(
            headers={"X-Finnhub-Token": settings.finnhub_api_key},
            timeout=20.0,
            transport=transport,
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def get_company_news(
        self,
        ticker: str,
        days: int = 30,
        limit: int = 15,
        keywords: list[str] | None = None,
    ) -> list[NewsArticle]:
        """Return up to `limit` unique articles from the last `days` days, newest first.

        If `keywords` is given (e.g. ["Apple", "AAPL"]), keep only articles whose
        headline or summary mentions one of them. This drops general market roundups.
        """
        today = date.today()
        params = {
            "symbol": ticker.upper(),
            "from": (today - timedelta(days=days)).isoformat(),
            "to": today.isoformat(),
        }
        resp = await self._http.get(NEWS_URL, params=params)

        # Free tier allows 60 calls/min; fail clearly (retries are added later)
        if resp.status_code == 429:
            raise RuntimeError("Finnhub rate limit hit, try again in a minute")
        resp.raise_for_status()

        # \b = word boundary, so ticker "F" won't match inside "Fund"
        pattern = (
            re.compile(r"\b(" + "|".join(map(re.escape, keywords)) + r")\b", re.IGNORECASE)
            if keywords
            else None
        )

        articles: list[NewsArticle] = []
        seen_headlines: set[str] = set()

        for item in resp.json():
            headline = item.get("headline", "")
            summary = item.get("summary", "")

            # Skip articles that don't actually mention the company
            if pattern and not pattern.search(f"{headline} {summary}"):
                continue

            # Normalize the headline so case/spacing differences still count as duplicates
            key = " ".join(headline.lower().split())
            if not key or key in seen_headlines:
                continue
            seen_headlines.add(key)

            articles.append(
                NewsArticle(
                    ticker=ticker.upper(),
                    headline=headline,
                    summary=summary,
                    source=item.get("source", "unknown"),
                    published_at=datetime.fromtimestamp(item["datetime"], tz=UTC),
                    url=item["url"],
                )
            )
            if len(articles) >= limit:
                break

        return articles