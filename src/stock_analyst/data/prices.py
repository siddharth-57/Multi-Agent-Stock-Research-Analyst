"""
prices.py: Stock price data via yfinance.

Downloads daily price history for a ticker and computes summary stats
(return, volatility, drawdown) in code, so the LLM never does the math.
yfinance is synchronous, so the download runs in a worker thread to keep
the async app responsive.
"""
import asyncio
import math

import pandas as pd
import yfinance as yf

from stock_analyst.data.models import PriceSummary

TRADING_DAYS_PER_YEAR = 252  # used to scale daily volatility to annual


def _download_history(ticker: str, period: str) -> pd.DataFrame:
    """Blocking call to Yahoo Finance. auto_adjust=True gives split/dividend-adjusted prices."""
    df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data for ticker: {ticker}")
    return df


async def get_price_summary(ticker: str, period: str = "1y") -> PriceSummary:
    """Fetch price history and return computed stats for the period."""
    # Run the blocking download in a thread so other tasks keep running
    df = await asyncio.to_thread(_download_history, ticker, period)

    closes = df["Close"]
    daily_returns = closes.pct_change().dropna()  # % change day to day

    # Drawdown = how far the price is below its highest point so far
    drawdown = closes / closes.cummax() - 1

    return PriceSummary(
        ticker=ticker.upper(),
        period=period,
        start_date=closes.index[0].date(),
        end_date=closes.index[-1].date(),
        latest_close=round(float(closes.iloc[-1]), 2),
        total_return=round(float(closes.iloc[-1] / closes.iloc[0] - 1), 4),
        annualized_volatility=round(
            float(daily_returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)), 4
        ),
        period_high=round(float(df["High"].max()), 2),   # intraday high
        period_low=round(float(df["Low"].min()), 2),     # intraday low
        max_drawdown=round(float(drawdown.min()), 4),
        source_url=f"https://finance.yahoo.com/quote/{ticker.upper()}/history",
    )