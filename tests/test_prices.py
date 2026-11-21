"""
test_prices.py: Unit tests for the price client.

Replaces the Yahoo download with a small fake price series where the
correct return, drawdown and high/low are easy to work out by hand.
"""
import pandas as pd
import pytest

from stock_analyst.data import prices


def fake_history(ticker: str, period: str) -> pd.DataFrame:
    # Closes 100 -> 110 -> 99 -> 120: return +20%, worst drop 110 -> 99 = -10%
    index = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    return pd.DataFrame(
        {"Close": [100, 110, 99, 120], "High": [101, 112, 100, 121], "Low": [99, 108, 98, 118]},
        index=index,
    )


async def test_price_stats(monkeypatch):
    monkeypatch.setattr(prices, "_download_history", fake_history)
    s = await prices.get_price_summary("aapl")

    assert s.ticker == "AAPL"            # ticker is normalized to upper case
    assert s.latest_close == 120
    assert s.total_return == 0.2
    assert s.max_drawdown == -0.1
    assert s.period_high == 121 and s.period_low == 98
    assert s.annualized_volatility > 0


async def test_empty_data_raises(monkeypatch):
    # Fake yfinance returning no rows, like it does for an invalid ticker
    class EmptyTicker:
        def __init__(self, ticker): ...
        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(prices.yf, "Ticker", EmptyTicker)
    with pytest.raises(ValueError):
        await prices.get_price_summary("FAKE")