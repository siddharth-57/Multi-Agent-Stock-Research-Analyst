"""
try_writer.py: Runs planner -> specialists (in parallel) -> writer for real.

Prints the final markdown report and saves it to report.md so it can be
opened in an editor or markdown preview.
"""
import asyncio
from pathlib import Path

from stock_analyst.agents.planner import make_planner
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
        writer = make_writer()

        state = {"question": "Is Apple overvalued right now?", "findings": {}, "errors": []}
        state.update(await planner(state))
        if "plan" not in state:
            print("Planner failed:", state["errors"])
            return

        # Merge specialist outputs by hand (the graph's reducers will do this later)
        for r in await asyncio.gather(*(specialists[n](state) for n in state["plan"].specialists)):
            state["findings"].update(r.get("findings", {}))
            state["errors"] += r.get("errors", [])

        out = await writer(state)
        if "draft" in out:
            print(out["draft"])
            Path("report.md").write_text(out["draft"])
            print("\n(saved to report.md)")
        print("errors:", out.get("errors", []))
    finally:
        await edgar.close()
        await news.close()


asyncio.run(main())