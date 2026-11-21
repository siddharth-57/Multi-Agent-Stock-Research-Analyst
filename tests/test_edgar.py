"""
test_edgar.py: Unit tests for the SEC EDGAR client.

Fakes SEC responses to check ticker lookup, annual-figure filtering,
keeping restated values, source links, and the required User-Agent header.
"""
from datetime import date

import httpx
import pytest

from stock_analyst.data.edgar import EdgarClient

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

FACTS = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {
                    "USD": [
                        # FY2023 as first reported
                        {"start": "2022-09-25", "end": "2023-09-30", "val": 383000,
                         "form": "10-K", "filed": "2023-11-03", "accn": "0000320193-23-000106"},
                        # FY2023 restated in the next year's 10-K -> this one should win
                        {"start": "2022-09-25", "end": "2023-09-30", "val": 383285,
                         "form": "10-K", "filed": "2024-11-01", "accn": "0000320193-24-000123"},
                        # Quarterly figure inside a 10-K -> should be dropped
                        {"start": "2024-06-30", "end": "2024-09-28", "val": 94930,
                         "form": "10-K", "filed": "2024-11-01", "accn": "0000320193-24-000123"},
                        # Quarterly report (10-Q) -> should be dropped
                        {"start": "2024-03-31", "end": "2024-06-29", "val": 85777,
                         "form": "10-Q", "filed": "2024-08-02", "accn": "0000320193-24-000081"},
                        # FY2024
                        {"start": "2023-10-01", "end": "2024-09-28", "val": 391035,
                         "form": "10-K", "filed": "2024-11-01", "accn": "0000320193-24-000123"},
                    ]
                }
            }
        }
    }
}

requests_seen: list[httpx.Request] = []  # lets tests inspect what was sent


def fake_sec(request: httpx.Request) -> httpx.Response:
    """Pretend to be SEC: route by URL path."""
    requests_seen.append(request)
    if request.url.path.endswith("company_tickers.json"):
        return httpx.Response(200, json=TICKERS)
    if "companyfacts" in request.url.path:
        return httpx.Response(200, json=FACTS)
    return httpx.Response(404)


@pytest.fixture
async def client():
    requests_seen.clear()
    c = EdgarClient(transport=httpx.MockTransport(fake_sec))
    yield c
    await c.close()


async def test_annual_revenue_filters_and_keeps_restated(client):
    facts = await client.get_annual_metric("AAPL", "revenue")

    # Only the two annual values survive, oldest first, restated FY2023 wins
    assert [f.value for f in facts] == [383285, 391035]
    assert facts[0].filed == date(2024, 11, 1)
    # Source link points at the filing the number came from
    assert "000032019324000123" in facts[1].source_url


async def test_cik_is_zero_padded_and_user_agent_sent(client):
    await client.get_annual_metric("AAPL", "revenue")
    facts_request = next(r for r in requests_seen if "companyfacts" in r.url.path)

    assert facts_request.url.path.endswith("CIK0000320193.json")
    assert facts_request.headers["User-Agent"] == "Test User test@example.com"


async def test_unknown_ticker_raises(client):
    with pytest.raises(ValueError):
        await client.get_annual_metric("FAKE", "revenue")


async def test_metric_not_reported_returns_empty(client):
    # Our fake company has no NetIncomeLoss data
    assert await client.get_annual_metric("AAPL", "net_income") == []