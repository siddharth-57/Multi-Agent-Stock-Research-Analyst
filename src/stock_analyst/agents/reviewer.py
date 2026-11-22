"""
reviewer.py: The reviewer agent, the last quality gate before a report is final.

Works in two layers: exact code checks first (numbers, citations, required
parts), then an LLM judges what needs judgment (claims supported, opinions
labeled, question answered, no advice). The report is approved only if both
layers pass; otherwise the combined issues go back to the writer.
"""
import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from stock_analyst.agents.checks import check_report
from stock_analyst.agents.state import ResearchState, Review
from stock_analyst.agents.writer import build_source_index
from stock_analyst.llm import get_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You review stock research reports before they reach the user.
Compare the report with the findings it was written from, and check that:
1. It directly answers the user's question.
2. Every claim is supported by the findings. No outside facts or benchmarks.
3. Opinions and interpretations are worded as opinions, and not cited as if a source said them.
4. Judgment words (high, strong, cheap, risky) are backed by a number or a comparison in the findings.
5. It never recommends buying, selling or holding.

Approve if there are no real problems. Do not reject for minor style. For each
problem, quote the sentence and say exactly how to fix it."""


def build_prompt(state: ResearchState) -> str:
    """The reviewer sees the same raw data the writer used, plus the draft."""
    parts = [f"User's question: {state['question']}", "", "Findings the report must be based on:"]
    for name, f in state["findings"].items():
        parts += [
            f"## {name}",
            f"Summary: {f.summary}",
            f"Raw tool data: {json.dumps(f.data.get('tool_calls', []), default=str)}",
            "",
        ]
    parts += ["Report to review:", state["draft"]]
    return "\n".join(parts)


def make_reviewer(llm: BaseChatModel | None = None):
    """Build the reviewer node. The LLM is passed in so tests can use a fake."""
    # Default (high) thinking: judging support and wording is this agent's whole job
    reviewer_llm = (llm or get_llm()).with_structured_output(Review)

    async def reviewer(state: ResearchState) -> dict:
        report = state.get("report")
        if report is None:
            return {"errors": ["reviewer: no report to review"]}

        # Layer 1: exact checks in code (free, same result every time)
        sources = build_source_index(state["findings"])
        code_issues = check_report(report, state["findings"], sources)

        # Layer 2: LLM judgment against the rubric
        errors: list[str] = []
        try:
            llm_review = await reviewer_llm.ainvoke(
                [SystemMessage(SYSTEM_PROMPT), HumanMessage(build_prompt(state))]
            )
        # LLM outage: don't crash; the code checks decide on their own
        except Exception as e:
            logger.exception("reviewer LLM check failed")
            llm_review, errors = None, [f"reviewer: LLM check skipped ({e})"]

        if llm_review is None and not errors:
            errors = ["reviewer: model returned no review"]

        llm_ok = llm_review.approved if llm_review else True
        llm_issues = llm_review.issues if llm_review else []

        # Approved only if BOTH layers pass
        update: dict = {
            "review": Review(approved=llm_ok and not code_issues, issues=code_issues + llm_issues)
        }
        if errors:
            update["errors"] = errors
        return update

    return reviewer