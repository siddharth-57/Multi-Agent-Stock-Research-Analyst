"""
edgar.py: Async client for the SEC EDGAR API.

Converts a ticker to its CIK (SEC's company ID), downloads all the
company's reported XBRL financial data, and returns clean annual figures
as FinancialFact objects. Free, official data; only a User-Agent is required.
"""
from datetime import date

import httpx

from stock_analyst.config import settings
from stock_analyst.data.models import FinancialFact

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{acc}-index.htm"

# Companies tag the same item differently, so each metric lists candidate
# XBRL tags in priority order. We use the first one the company reports.
METRIC_CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": ["StockholdersEquity"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
}


class EdgarClient:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        # `transport` is only passed in tests, to fake SEC responses without network calls
        self._http = httpx.AsyncClient(
            headers={"User-Agent": settings.sec_user_agent},
            timeout=30.0,
            transport=transport,
        )
        
        self._ticker_map: dict[str, int] | None = None   # ticker -> CIK, loaded once
        self._facts_cache: dict[int, dict] = {}           # CIK -> raw facts JSON

    async def close(self) -> None:
        await self._http.aclose()

    async def get_cik(self, ticker: str) -> int:
        """Look up a ticker's CIK. The full ticker list is downloaded once and cached."""
        if self._ticker_map is None:
            resp = await self._http.get(TICKERS_URL)
            resp.raise_for_status()
            # Shape: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
            self._ticker_map = {
                row["ticker"].upper(): row["cik_str"] for row in resp.json().values()
            }
        cik = self._ticker_map.get(ticker.upper())
        if cik is None:
            raise ValueError(f"Unknown ticker: {ticker}")
        return cik

    async def get_annual_metric(
        self, ticker: str, metric: str, years: int = 5
    ) -> list[FinancialFact]:
        """Return the last `years` annual values of a metric, oldest first."""
        if metric not in METRIC_CONCEPTS:
            raise ValueError(f"Unsupported metric: {metric}")

        cik = await self.get_cik(ticker)
        us_gaap = (await self._get_company_facts(cik)).get("facts", {}).get("us-gaap", {})

        # Try each candidate tag until one yields annual data
        for concept in METRIC_CONCEPTS[metric]:
            if concept in us_gaap:
                results = self._parse_annual(cik, metric, concept, us_gaap[concept])
                if results:
                    return results[-years:]
        return []  # company doesn't report this metric

    async def _get_company_facts(self, cik: int) -> dict:
        """Download all XBRL facts for a company (a few MB), cached per CIK."""
        if cik not in self._facts_cache:
            # The API expects the CIK zero-padded to 10 digits
            resp = await self._http.get(FACTS_URL.format(cik=f"{cik:010d}"))
            resp.raise_for_status()
            self._facts_cache[cik] = resp.json()
        return self._facts_cache[cik]

    @staticmethod
    def _parse_annual(
        cik: int, metric: str, concept: str, concept_data: dict
    ) -> list[FinancialFact]:
        """Filter raw facts down to one clean value per fiscal year."""
        by_period: dict[date, FinancialFact] = {}

        # "units" groups values by unit, e.g. {"USD": [...]} or {"USD/shares": [...]}
        for unit, entries in concept_data["units"].items():
            for e in entries:
                if e.get("form") != "10-K":   # annual reports only
                    continue

                start = date.fromisoformat(e["start"]) if "start" in e else None
                end = date.fromisoformat(e["end"])

                # 10-Ks can also contain quarterly figures; keep only ~1-year periods.
                # (We use dates, not the "fy" field, because "fy" is the year of
                # the filing, not of the period the number belongs to.)
                if start and not 350 <= (end - start).days <= 380:
                    continue

                # The same period reappears in later 10-Ks as a comparison column.
                # Keep the most recently filed version, which includes any restatements.
                filed = date.fromisoformat(e["filed"])
                existing = by_period.get(end)
                if existing and existing.filed >= filed:
                    continue

                acc = e["accn"]
                by_period[end] = FinancialFact(
                    metric=metric,
                    concept=concept,
                    value=e["val"],
                    unit=unit,
                    period_start=start,
                    period_end=end,
                    form=e["form"],
                    filed=filed,
                    accession=acc,
                    source_url=FILING_URL.format(
                        cik=cik, acc_nodash=acc.replace("-", ""), acc=acc
                    ),
                )

        return [by_period[k] for k in sorted(by_period)]  # oldest -> newest