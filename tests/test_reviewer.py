"""
test_reviewer.py: Unit tests for the reviewer node.

Checks that approval needs both layers to pass, that code-check problems
block an LLM approval, and that an LLM failure falls back to code checks.
"""
from stock_analyst.agents.reviewer import make_reviewer
from stock_analyst.agents.state import Review
from tests.fakes import FixedOutputLLM
from tests.test_checks import FINDINGS, report_with

GOOD = report_with(("Net margin", "27.6%", "S1"))
BAD = report_with(("Net margin", "99.9%", "S1"))


def state_with(report):
    return {"question": "q", "findings": FINDINGS, "report": report, "draft": "# draft"}


async def test_approved_when_both_layers_pass():
    reviewer = make_reviewer(llm=FixedOutputLLM(Review(approved=True)))
    out = await reviewer(state_with(GOOD))
    assert out["review"].approved and out["review"].issues == []


async def test_code_issue_blocks_llm_approval():
    reviewer = make_reviewer(llm=FixedOutputLLM(Review(approved=True)))
    out = await reviewer(state_with(BAD))
    assert not out["review"].approved
    assert "99.9" in out["review"].issues[0]


async def test_llm_failure_falls_back_to_code_checks():
    class BrokenLLM:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            raise RuntimeError("API down")

    out = await make_reviewer(llm=BrokenLLM())(state_with(GOOD))
    assert out["review"].approved                  # code checks passed
    assert "LLM check skipped" in out["errors"][0]  # but we say the LLM layer didn't run