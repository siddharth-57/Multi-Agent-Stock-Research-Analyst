"""
models.py: Typed data models for financial data.

Every number carries its source (filing, accession number, URL) so agents
can cite it and the evaluation harness can verify it later.
"""
from datetime import date, datetime

from pydantic import BaseModel


class FinancialFact(BaseModel):
    """One reported value for one period, e.g. Apple's FY2024 revenue."""
    metric: str                 # our friendly name, e.g. "revenue"
    concept: str                # actual XBRL tag the company used
    value: float
    unit: str                   # "USD", "USD/shares", "shares"
    period_start: date | None   # None for balance-sheet items (a single point in time)
    period_end: date
    form: str                   # filing type, e.g. "10-K" (annual report)
    filed: date                 # date the filing was submitted to SEC
    accession: str              # unique filing ID
    source_url: str             # link to the filing on sec.gov

class PriceSummary(BaseModel):
    """Price stats for one ticker over a period, all computed in code."""
    ticker: str
    period: str                    # e.g. "1y", "5y"
    start_date: date
    end_date: date
    latest_close: float            # adjusted for splits and dividends
    total_return: float            # 0.12 means +12% over the period
    annualized_volatility: float   # std dev of daily returns, scaled to one year
    period_high: float
    period_low: float
    max_drawdown: float            # worst peak-to-bottom fall, e.g. -0.25 = -25%
    source_url: str

class NewsArticle(BaseModel):
    """One news article about a company, with its source for citation."""
    ticker: str
    headline: str
    summary: str          # short blurb from the publisher; can be empty
    source: str           # publisher name, e.g. "Reuters"
    published_at: datetime
    url: str