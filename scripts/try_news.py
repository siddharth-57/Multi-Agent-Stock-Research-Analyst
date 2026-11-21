"""
try_news.py: Manual check of the Finnhub news client.

Prints Apple's latest unique news headlines with date, source and link.
"""
import asyncio

from stock_analyst.data.news import FinnhubClient


async def main() -> None:
    client = FinnhubClient()
    try:
        articles = await client.get_company_news(
            "AAPL", days=7, limit=10, keywords=["Apple", "AAPL"]
        )
        print(f"Got {len(articles)} articles\n")
        for a in articles:
            print(f"[{a.published_at:%Y-%m-%d}] {a.source}: {a.headline}")
            print(f"    {a.url}")
    finally:
        await client.close()


asyncio.run(main())