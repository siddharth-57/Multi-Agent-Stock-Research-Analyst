"""
try_planner.py: Runs the real planner on a few sample questions.

Shows which ticker and specialists the LLM picks for each kind of question,
plus how a fake company is handled.
"""
import asyncio

from stock_analyst.agents.planner import make_planner
from stock_analyst.data.edgar import EdgarClient

QUESTIONS = [
    "Is Apple overvalued right now?",             # expect fundamentals + market
    "Why did Nvidia stock move this week?",       # expect market + news
    "Give me a full analysis of Microsoft",       # expect all three
    "What do you think of Zorblax Quantum Corp?", # fake company, expect an error
]


async def main() -> None:
    edgar = EdgarClient()
    planner = make_planner(edgar)
    try:
        for q in QUESTIONS:
            result = await planner({"question": q})
            print(f"\nQ: {q}")
            if "plan" in result:
                p = result["plan"]
                print(f"  ticker={p.ticker}  specialists={p.specialists}")
                print(f"  focus: {p.focus}")
            else:
                print(f"  error: {result['errors']}")
    finally:
        await edgar.close()


asyncio.run(main())