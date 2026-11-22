"""
test_specialists.py: Unit tests for the specialist nodes.

Checks that a specialist turns a tool run into SpecialistFindings (summary,
raw data, sources found by code, usage), and that a crash becomes an error entry.
"""
from langchain_core.tools import tool

from stock_analyst.agents.specialists import make_specialist
from stock_analyst.agents.state import Plan
from tests.fakes import ScriptedLLM, answer, ask

PLAN = Plan(ticker="AAPL", company_name="Apple", specialists=["fundamentals"], focus="test")


@tool
def lookup(ticker: str) -> dict:
    """Fake data tool with sources at different nesting levels."""
    return {"value": 42, "source": "https://sec.example/filing",
            "nested": [{"url": "https://news.example/a"}]}


async def test_specialist_builds_findings():
    llm = ScriptedLLM([ask("lookup", {"ticker": "AAPL"}), answer("Revenue was 42 in FY2026.")])
    node = make_specialist("fundamentals", [lookup], llm=llm)
    out = await node({"question": "q", "plan": PLAN})

    f = out["findings"]["fundamentals"]
    assert f.summary == "Revenue was 42 in FY2026."
    assert f.sources == ["https://news.example/a", "https://sec.example/filing"]
    assert f.data["tool_calls"][0]["result"]["value"] == 42
    assert f.usage["rounds"] == 1


async def test_specialist_crash_becomes_error():
    class BrokenLLM:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            raise RuntimeError("API down")

    node = make_specialist("news", [], llm=BrokenLLM())
    out = await node({"question": "q", "plan": PLAN})
    assert out == {"errors": ["news: API down"]}