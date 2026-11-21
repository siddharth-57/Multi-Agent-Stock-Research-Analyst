"""
try_llm.py: Smoke test for the LLM setup.

1) Sends a plain question to confirm the API key and model work.
2) Gives the model a price tool and checks that it chooses to call it.
   The model only *proposes* the call here; we don't run the tool yet.
"""
import asyncio

from langchain_core.tools import tool

from stock_analyst.data.prices import get_price_summary
from stock_analyst.llm import get_llm


@tool
async def price_summary(ticker: str, period: str = "1y") -> dict:
    """Get price stats (return, volatility, drawdown, high/low) for a US stock
    ticker over a period like '1y' or '5y'."""
    # The docstring above is what the model reads to decide when to use this tool
    return (await get_price_summary(ticker, period)).model_dump(mode="json")


async def main() -> None:
    llm = get_llm()

    # 1) Plain call: checks the key and model name
    reply = await llm.ainvoke("In one sentence, what does a 10-K filing contain?")
    print("Plain answer:\n ", reply.text, "\n")

    # 2) Tool calling: does the model decide it needs the price tool?
    llm_with_tools = llm.bind_tools([price_summary])
    reply = await llm_with_tools.ainvoke("How has Apple stock performed over the last year?")
    print("Tool calls the model wants to make:")
    for call in reply.tool_calls:
        print(" ", call["name"], call["args"])


asyncio.run(main())