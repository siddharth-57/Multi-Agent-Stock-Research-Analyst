"""
try_specialists.py: Runs the real planner, then the specialists it picks.

The chosen specialists run at the same time (the graph will do this for us
later). Prints each summary, its number of sources and its usage stats.
"""
import asyncio

from stock_analyst.agents.planner import make_planner
from stock_analyst.agents.specialists import make_specialists
from stock_analyst.agents.tools import make_tools
from stock_analyst.data.edgar import EdgarClient
from stock_analyst.data.news import FinnhubClient


async def main() -> None:
    edgar, news = EdgarClient(), FinnhubClient()
    try:
        planner = make_planner(edgar)
        specialists = make_specialists(make_tools(edgar, news))

        state = {"question": "Is Apple overvalued right now?"}
        state.update(await planner(state))
        if "plan" not in state:
            print("Planner failed:", state.get("errors"))
            return

        plan = state["plan"]
        print(f"Plan: {plan.ticker} -> {plan.specialists}\n")

        # Run the chosen specialists in parallel
        results = await asyncio.gather(*(specialists[n](state) for n in plan.specialists))
        for r in results:
            for f in r.get("findings", {}).values():
                print(f"=== {f.specialist} ===\n{f.summary}")
                print(f"sources: {len(f.sources)} | usage: {f.usage}\n")
            for err in r.get("errors", []):
                print(f"ERROR: {err}")
    finally:
        await edgar.close()
        await news.close()


asyncio.run(main())