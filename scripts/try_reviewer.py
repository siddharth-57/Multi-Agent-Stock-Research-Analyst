"""
try_reviewer.py: Runs planner -> specialists -> writer -> reviewer for real.

Prints the reviewer's verdict on the real report, then deliberately breaks
one key figure to show the code checks catching it.
"""
import asyncio

from stock_analyst.agents.planner import make_planner
from stock_analyst.agents.reviewer import make_reviewer
from stock_analyst.agents.specialists import make_specialists
from stock_analyst.agents.tools import make_tools
from stock_analyst.agents.writer import make_writer
from stock_analyst.data.edgar import EdgarClient
from stock_analyst.data.news import FinnhubClient


async def main() -> None:
    edgar, news = EdgarClient(), FinnhubClient()
    try:
        planner = make_planner(edgar)
        specialists = make_specialists(make_tools(edgar, news))
        writer, reviewer = make_writer(), make_reviewer()

        state = {"question": "Is Apple overvalued right now?", "findings": {}, "errors": []}
        state.update(await planner(state))
        if "plan" not in state:
            print("Planner failed:", state["errors"])
            return

        for r in await asyncio.gather(*(specialists[n](state) for n in state["plan"].specialists)):
            state["findings"].update(r.get("findings", {}))
            state["errors"] += r.get("errors", [])

        state.update(await writer(state))
        if "report" not in state:
            print("Writer failed:", state["errors"])
            return

        review = (await reviewer(state))["review"]
        print(f"Real report -> approved={review.approved}")
        for issue in review.issues:
            print(" -", issue)

        # Break one figure on purpose: the code checks must catch it
        tampered = state["report"].model_copy(deep=True)
        tampered.key_figures[0].value = "99.99"
        bad = (await reviewer({**state, "report": tampered}))["review"]
        print(f"\nTampered report -> approved={bad.approved}")
        for issue in bad.issues:
            print(" -", issue)
    finally:
        await edgar.close()
        await news.close()


asyncio.run(main())