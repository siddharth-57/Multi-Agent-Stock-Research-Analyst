"""
specialists.py: The three specialist agents (fundamentals, market, news).

All three share one tool loop and differ only in their prompt and tools.
Each returns SpecialistFindings with its written summary, the raw tool
results (the real numbers), source links found by code, and usage stats.
A failing specialist records an error instead of stopping the whole run.
"""
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from stock_analyst.agents.loop import run_tool_loop
from stock_analyst.agents.state import ResearchState, SpecialistFindings, SpecialistName
from stock_analyst.llm import get_llm

logger = logging.getLogger(__name__)

# Rules every specialist follows, so reports stay grounded in tool data
COMMON_RULES = """
Rules:
- Use only numbers that appear in your tool results. Never estimate or calculate them yourself.
- State the fiscal year or date for every figure.
- If a tool returns an error, say what data is missing instead of guessing.
- Keep your summary under 200 words.
- Cover only your own area. Other specialists handle the rest, so never report
  data outside your area as missing."""

PROMPTS: dict[SpecialistName, str] = {
    "fundamentals": (
        "You are a fundamentals analyst. Assess the company's financial health and valuation "
        "using official SEC data. Use compute_ratios for any ratio." + COMMON_RULES
    ),
    "market": (
        "You are a market analyst. Describe the stock's price performance, volatility and "
        "drawdowns. Choose the period(s) that fit the question." + COMMON_RULES
    ),
    "news": (
        "You are a news analyst. Summarize the recent events that matter for the stock. "
        "Separate reported facts from opinion pieces, and give dates." + COMMON_RULES
    ),
}

SOURCE_KEYS = {"source", "sources", "url", "source_url"}


def collect_sources(obj) -> set[str]:
    """Walk tool results and pull out every source link, however deeply nested."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in SOURCE_KEYS and isinstance(value, str):
                found.add(value)
            elif key in SOURCE_KEYS and isinstance(value, list):
                found.update(v for v in value if isinstance(v, str))
            else:
                found |= collect_sources(value)  # look deeper
    elif isinstance(obj, list):
        for item in obj:
            found |= collect_sources(item)
    return found


def make_specialist(name: SpecialistName, tools: list[BaseTool], llm: BaseChatModel | None = None):
    """Build one specialist node. The LLM is passed in so tests can use a fake."""
    # Low thinking: specialists mainly pick tools and summarize their results
    model = llm or get_llm(thinking_level="low")

    async def specialist(state: ResearchState) -> dict:
        plan = state["plan"]
        messages = [
            SystemMessage(PROMPTS[name]),
            HumanMessage(
                f"Ticker: {plan.ticker}\nCompany: {plan.company_name}\n"
                f"User's question: {state['question']}\nFocus: {plan.focus}"
            ),
        ]
        try:
            result = await run_tool_loop(model, tools, messages)
        # One specialist failing must not stop the others; traceback is logged
        except Exception as e:
            logger.exception("%s specialist failed", name)
            return {"errors": [f"{name}: {e}"]}

        findings = SpecialistFindings(
            specialist=name,
            summary=result.final_text,
            data={"tool_calls": result.tool_calls},  # raw numbers kept for writer + evaluation
            sources=sorted(collect_sources(result.tool_calls)),
            usage=result.model_dump(
                include={"rounds", "input_tokens", "output_tokens", "latency_s", "hit_limit","models_used"}
            ),
        )
        return {"findings": {name: findings}}

    return specialist


def make_specialists(tools_by_specialist: dict[str, list[BaseTool]], llm: BaseChatModel | None = None):
    """Build all specialist nodes, each with only its own tools."""
    return {name: make_specialist(name, tools, llm) for name, tools in tools_by_specialist.items()}