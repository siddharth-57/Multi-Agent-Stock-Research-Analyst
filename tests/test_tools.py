"""
test_tools.py: Unit tests for the specialist tools.

Checks the ratio math with numbers easy to verify by hand, the edge cases
(negative EPS, figures from different years), that tool metric names match
the EDGAR client, and that tools return errors instead of raising.
"""
from datetime import date
from typing import get_args

import httpx

from stock_analyst.agents.tools import MetricName, calculate_ratios, make_tools
from stock_analyst.data.edgar import METRIC_CONCEPTS, EdgarClient
from stock_analyst.data.models import FinancialFact
from stock_analyst.data.news import FinnhubClient


def fact(metric: str, value: float, end: str = "2025-09-27") -> FinancialFact:
    """Build a FinancialFact with dummy metadata; only metric, value and date matter here."""
    return FinancialFact(
        metric=metric, concept="x", value=value, unit="USD", period_start=None,
        period_end=date.fromisoformat(end), form="10-K", filed=date(2025, 11, 1),
        accession="0000000000-25-000001", source_url=f"https://sec.example/{metric}",
    )


def test_metric_names_match_edgar():
    # If someone adds a metric to EDGAR but not the tool (or vice versa), this fails
    assert set(get_args(MetricName)) == set(METRIC_CONCEPTS)


def test_ratios_calculated_correctly():
    facts = {
        "revenue": [fact("revenue", 100, "2024-09-28"), fact("revenue", 120)],
        "net_income": [fact("net_income", 12)],
        "operating_income": [fact("operating_income", 18)],
        "eps_diluted": [fact("eps_diluted", 2)],
        "total_liabilities": [fact("total_liabilities", 90)],
        "stockholders_equity": [fact("stockholders_equity", 60)],
    }
    r = calculate_ratios(facts, price=40)

    assert r["net_margin_pct"] == 10.0          # 12 / 120
    assert r["operating_margin_pct"] == 15.0    # 18 / 120
    assert r["revenue_growth_yoy_pct"] == 20.0  # 120 / 100 - 1
    assert r["return_on_equity_pct"] == 20.0    # 12 / 60
    assert r["liabilities_to_equity"] == 1.5   # 90 / 60
    assert r["pe_ratio"] == 20                 # 40 / 2


def test_negative_eps_gives_no_pe():
    r = calculate_ratios({"eps_diluted": [fact("eps_diluted", -1.5)]}, price=40)
    assert r["pe_ratio"] is None


def test_figures_from_different_years_are_not_divided():
    facts = {
        "net_income": [fact("net_income", 12, "2024-09-28")],
        "revenue": [fact("revenue", 120)],  # newer year than net income
    }
    assert calculate_ratios(facts, price=None)["net_margin_pct"] is None


async def test_tool_returns_error_instead_of_raising():
    # Fake SEC returns an empty ticker list, so every lookup fails inside the tool
    edgar = EdgarClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    news = FinnhubClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
    get_financials = make_tools(edgar, news)["fundamentals"][0]

    result = await get_financials.ainvoke({"ticker": "FAKE", "metrics": ["revenue"]})
    await edgar.close()
    await news.close()

    assert "error" in result