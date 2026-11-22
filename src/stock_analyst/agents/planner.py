"""
planner.py: The planner agent, the first node in the graph.

Reads the user's question and returns a structured Plan: which ticker,
which specialists are needed, and what to focus on. The LLM's ticker is
then checked against SEC's official list, so a made-up ticker stops the
run early instead of producing a report about nothing.
"""
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from stock_analyst.agents.state import Plan, ResearchState
from stock_analyst.data.edgar import EdgarClient
from stock_analyst.llm import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the planning step of a US stock research system.
Read the user's question and decide what research is needed.

Specialists available:
- fundamentals: official financials from SEC filings (revenue, profit, margins, balance sheet)
- market: stock price performance, volatility, drawdowns
- news: recent company news and events

Rules:
- Choose only the specialists the question actually needs. Fewer is better.
- Valuation questions (cheap, expensive, overvalued) need fundamentals and market.
- Questions about recent price moves need market and news.
- A general "analyze this company" question needs all three.
- Use the company's main US-listed ticker."""


def make_planner(edgar: EdgarClient, llm: BaseChatModel | None = None):
    """Build the planner node. Clients are passed in so tests can use fakes."""
    # Forces the LLM to return a validated Plan object instead of free text
    # Low thinking: picking specialists is a simple classification task
    planner_llm = (llm or get_llm(thinking_level="low")).with_structured_output(Plan)

    async def planner(state: ResearchState) -> dict:
        try:
            plan = await planner_llm.ainvoke(
                [SystemMessage(SYSTEM_PROMPT), HumanMessage(state["question"])]
            )
        # LLM outage or rate limit: record it so the graph can stop cleanly
        except Exception as e:
            logger.exception("planner failed")
            return {"errors": [f"planner: {e}"]}

        # Structured output can come back empty if the model fails to follow the schema
        if plan is None:
            return {"errors": ["planner: model returned no plan"]}

        plan.ticker = plan.ticker.upper().strip()

        # Trust but verify: never research a ticker SEC doesn't know
        try:
            await edgar.get_cik(plan.ticker)
        except ValueError:
            return {"errors": [f"planner: unknown ticker '{plan.ticker}'"]}

        return {"plan": plan}

    return planner