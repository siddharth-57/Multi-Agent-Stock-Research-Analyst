"""
try_tools.py: Calls each specialist tool for Apple against the real APIs.

Also prints one tool's schema (the JSON description the LLM receives)
and shows a tool returning an error for a fake ticker.
"""
import asyncio
import json

from stock_analyst.agents.tools import make_tools
from stock_analyst.data.edgar import EdgarClient
from stock_analyst.data.news import FinnhubClient


async def main() -> None:
    edgar, news = EdgarClient(), FinnhubClient()
    try:
        tools = make_tools(edgar, news)
        get_financials, compute_ratios = tools["fundamentals"]
        (price_tool,) = tools["market"]
        (news_tool,) = tools["news"]

        # This is exactly what the model reads to decide how to call the tool
        print("Schema the LLM sees for get_financials:")
        print(json.dumps(get_financials.args, indent=2))

        print("\ncompute_ratios(AAPL):")
        print(json.dumps(await compute_ratios.ainvoke({"ticker": "AAPL"}), indent=2))

        print("\nget_financials(AAPL, revenue, 3 years):")
        result = await get_financials.ainvoke({"ticker": "AAPL", "metrics": ["revenue"], "years": 3})
        print(json.dumps(result, indent=2))

        news_result = await news_tool.ainvoke({"ticker": "AAPL", "company_name": "Apple", "days": 7})
        print(f"\nget_company_news: {len(news_result.get('articles', []))} articles")

        print("\nError handling (fake ticker):")
        print(await price_tool.ainvoke({"ticker": "NOTREAL123"}))
    finally:
        await edgar.close()
        await news.close()


asyncio.run(main())