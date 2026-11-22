"""
test_planner.py: Unit tests for the planner node.

Uses a fake LLM (returns a fixed Plan) and a fake SEC ticker list to test
the logic around the LLM: ticker cleanup, ticker verification, and
handling an empty model response.
"""
import httpx

from stock_analyst.agents.planner import make_planner
from stock_analyst.agents.state import Plan
from stock_analyst.data.edgar import EdgarClient

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}


class FakeLLM:
    """Stands in for the real model: always returns the plan it was given."""
    def __init__(self, plan):
        self.plan = plan

    def with_structured_output(self, schema):
        return self  # the planner calls this once at build time

    async def ainvoke(self, messages):
        return self.plan


def fake_edgar() -> EdgarClient:
    return EdgarClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=TICKERS)))


def make_plan(ticker: str) -> Plan:
    return Plan(ticker=ticker, company_name="Apple", specialists=["market"], focus="test")


async def test_valid_ticker_is_cleaned_and_returned():
    edgar = fake_edgar()
    planner = make_planner(edgar, llm=FakeLLM(make_plan(" aapl ")))
    result = await planner({"question": "How is Apple doing?"})
    await edgar.close()

    assert result["plan"].ticker == "AAPL"   # trimmed and upper-cased


async def test_unknown_ticker_stops_with_error():
    edgar = fake_edgar()
    planner = make_planner(edgar, llm=FakeLLM(make_plan("ZZZZ")))
    result = await planner({"question": "Tell me about ZZZZ"})
    await edgar.close()

    assert "plan" not in result
    assert "unknown ticker" in result["errors"][0]


async def test_empty_model_output_is_an_error():
    edgar = fake_edgar()
    planner = make_planner(edgar, llm=FakeLLM(None))
    result = await planner({"question": "anything"})
    await edgar.close()

    assert result["errors"] == ["planner: model returned no plan"]