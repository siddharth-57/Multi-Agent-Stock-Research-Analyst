"""
try_prices.py: Manual check of the price client.

Prints 1-year price stats for Apple, then confirms that an invalid
ticker raises a clear error instead of returning garbage.
"""
import asyncio

from stock_analyst.data.prices import get_price_summary


async def main() -> None:
    summary = await get_price_summary("AAPL", "1y")
    # model_dump_json prints the whole model in readable JSON
    print(summary.model_dump_json(indent=2))

    # A fake ticker should fail loudly
    try:
        await get_price_summary("NOTAREALTICKER123")
    except ValueError as e:
        print(f"\nInvalid ticker handled: {e}")


asyncio.run(main())