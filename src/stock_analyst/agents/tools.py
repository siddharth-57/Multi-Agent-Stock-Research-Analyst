"""
tools.py: The tools specialists can call, grouped by specialist.

Wraps the data clients as LangChain tools. Each tool's docstring and typed
arguments become the schema the LLM sees. Tools return compact JSON with
sources, return errors as data (so the LLM can recover instead of crashing),
and do all math in code, so numbers never pass through the LLM.
"""

import logging
from typing import Literal

from langchain_core.tools import BaseTool, tool

from stock_analyst.data.edgar import EdgarClient
from stock_analyst.data.models import FinancialFact
from stock_analyst.data.news import FinnhubClient
from stock_analyst.data.prices import get_price_summary as fetch_price_summary

logger = logging.getLogger(__name__)


def _tool_error(name: str, e: Exception) -> dict:
    """Log the full traceback for developers; return a short message for the LLM."""
    logger.exception("%s failed", name)  # must be called inside an except block
    return {"error": f"{name} failed: {e}"}

# Listed explicitly so the LLM sees the allowed values in the tool schema
MetricName = Literal[
    "revenue", "net_income", "operating_income", "eps_diluted", "total_assets",
    "total_liabilities", "stockholders_equity", "operating_cash_flow", "cash",
]
Period = Literal["1mo", "3mo", "6mo", "1y", "2y", "5y"]

# The figures compute_ratios needs from SEC
RATIO_INPUTS = [
    "revenue", "net_income", "operating_income",
    "eps_diluted", "total_liabilities", "stockholders_equity",
]

def pct(x: float | None) -> float | None:
    """0.2762 -> 27.62, so the LLM can quote percentages without doing math."""
    return round(x * 100, 2) if x is not None else None


def calculate_ratios(facts: dict[str, list[FinancialFact]], price: float | None) -> dict:
    """Pure function: compute key ratios from the latest annual figures."""
    latest = {m: series[-1] for m, series in facts.items() if series}

    def ratio(top: str, bottom: str) -> float | None:
        # Only divide figures from the same fiscal year, and never by zero
        a, b = latest.get(top), latest.get(bottom)
        if a is None or b is None or not b.value or a.period_end != b.period_end:
            return None
        return round(a.value / b.value, 4)

    revenue = facts.get("revenue", [])
    growth = (
        round(revenue[-1].value / revenue[-2].value - 1, 4)
        if len(revenue) >= 2 and revenue[-2].value
        else None
    )

    # P/E is meaningless when earnings are zero or negative
    eps = latest.get("eps_diluted")
    pe = round(price / eps.value, 2) if price and eps and eps.value > 0 else None

    return {
        "fiscal_year_end": str(latest["revenue"].period_end) if "revenue" in latest else None,
        "net_margin_pct": pct(ratio("net_income", "revenue")),
        "operating_margin_pct": pct(ratio("operating_income", "revenue")),
        "revenue_growth_yoy_pct": pct(growth),
        "return_on_equity_pct": pct(ratio("net_income", "stockholders_equity")),
        "liabilities_to_equity": ratio("total_liabilities", "stockholders_equity"),
        "pe_ratio": pe,
        "note": "P/E = latest close / diluted EPS from the latest annual report (not trailing 12 months)",
        "sources": sorted({f.source_url for f in latest.values()}),
    }


def make_tools(edgar: EdgarClient, news: FinnhubClient) -> dict[str, list[BaseTool]]:
    """Build each specialist's tool set. Clients are passed in so tests can use fakes."""

    @tool
    async def get_financials(ticker: str, metrics: list[MetricName], years: int = 5) -> dict:
        """Get official annual figures from SEC 10-K filings for a US stock.
        Returns each metric's values by fiscal year end, with source links.
        `years` is how many recent fiscal years to return (1-10)."""
        try:
            years = min(max(years, 1), 10)  # keep the LLM's input in a sane range
            result = {}
            # Sequential on purpose: the first call caches the company's SEC data
            for metric in metrics:
                facts = await edgar.get_annual_metric(ticker, metric, years=years)
                result[metric] = [
                    {"fiscal_year_end": str(f.period_end), "value": f.value,
                     "unit": f.unit, "source": f.source_url}
                    for f in facts
                ]
            return {"ticker": ticker.upper(), "metrics": result}
        # Tool boundary: catch everything so the agent survives; traceback is logged
        except Exception as e:  # noqa: BLE001
            return _tool_error("get_financials", e)

    @tool
    async def compute_ratios(ticker: str) -> dict:
        """Compute key ratios for a US stock from its latest annual report: net margin,
        operating margin, revenue growth, return on equity, liabilities-to-equity and P/E.
        All math is done in code. Use this instead of calculating ratios yourself."""
        try:
            facts = {m: await edgar.get_annual_metric(ticker, m, years=2) for m in RATIO_INPUTS}
            price = (await fetch_price_summary(ticker, "1mo")).latest_close
            return {"ticker": ticker.upper(), **calculate_ratios(facts, price)}
        
        # Tool boundary: catch everything so the agent survives; traceback is logged
        except Exception as e:  # noqa: BLE001
            return _tool_error("compute_ratios", e)


    @tool
    async def get_price_summary(ticker: str, period: Period = "1y") -> dict:
        """Get stock price stats for a US ticker over a period: latest close, total
        return, annualized volatility, high/low and max drawdown.
        Fields ending in _pct are percentages (26.62 means +26.62%)."""
        try:
            data = (await fetch_price_summary(ticker, period)).model_dump(mode="json")
            # Convert decimals to percentages here, so the LLM never does the math
            for key in ("total_return", "annualized_volatility", "max_drawdown"):
                data[f"{key}_pct"] = pct(data.pop(key))
            return data
        # Tool boundary: catch everything so the agent survives; traceback is logged
        except Exception as e:  # noqa: BLE001
            return _tool_error("get_price_summary", e)
        

    @tool
    async def get_company_news(ticker: str, company_name: str, days: int = 30) -> dict:
        """Get recent news about a company, newest first, with duplicates and unrelated
        market roundups removed. `days` is how far back to look (1-90)."""
        try:
            articles = await news.get_company_news(
                ticker, days=min(max(days, 1), 90), limit=15,
                keywords=[company_name, ticker],
            )
            return {
                "ticker": ticker.upper(),
                "articles": [
                    {"date": f"{a.published_at:%Y-%m-%d}", "source": a.source,
                     "headline": a.headline, "summary": a.summary[:300], "url": a.url}
                    for a in articles
                ],
            }
            
        # Tool boundary: catch everything so the agent survives; traceback is logged
        except Exception as e:  # noqa: BLE001
            return _tool_error("get_company_news", e)


    # Each specialist only gets the tools it needs
    return {
        "fundamentals": [get_financials, compute_ratios],
        "market": [get_price_summary],
        "news": [get_company_news],
    }