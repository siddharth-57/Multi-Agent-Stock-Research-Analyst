"""
try_edgar.py: Manual check of the EDGAR client.

Prints Apple's last 5 years of revenue and net income with source links,
so we can compare them against Apple's actual annual reports.
"""
import asyncio

from stock_analyst.data.edgar import EdgarClient


async def main() -> None:
    client = EdgarClient()
    try:
        for metric in ["revenue", "net_income"]:
            facts = await client.get_annual_metric("AAPL", metric)
            print(f"\n{metric}:")
            for f in facts:
                print(f"  FY ending {f.period_end}: {f.value:,.0f} {f.unit}")
                print(f"    source: {f.source_url}")
    finally:
        await client.close()  # always release the HTTP connection


asyncio.run(main())