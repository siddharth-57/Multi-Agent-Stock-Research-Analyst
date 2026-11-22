"""
writer.py: The writer agent, which turns specialist findings into one report.

Code first gives every source URL an ID (S1, S2, ...). The LLM fills a
structured Report citing only those IDs, using the raw tool data for exact
numbers. Code then renders the markdown, adds the real URLs and the
disclaimer, and flags any citation to a source ID that doesn't exist.
"""
import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from stock_analyst.agents.state import Report, ResearchState, SpecialistFindings
from stock_analyst.llm import get_llm

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This report was generated automatically from public data for research purposes "
    "and is not investment advice."
)

SYSTEM_PROMPT = """You are the writer for a US stock research system. Combine the
specialists' findings into one clear report that answers the user's question.

Rules:
- Use only facts and numbers from the findings below. Do not add outside facts,
  benchmarks or historical comparisons that are not in the findings.
- Copy every number exactly as it appears in the raw tool data.
- Every key figure needs a source_id from the source list. Cite claims in the text like [S1].
- Cite only source IDs from the list. Never write URLs yourself.
- Give a balanced assessment. Never recommend buying, selling or holding.
- List any missing data in data_gaps.
- When the data has a "display" value, use it in the text instead of the raw number."""

CITATION = re.compile(r"\[(S\d+)\]")  # matches [S1], [S12], ...


def build_source_index(findings: dict[str, SpecialistFindings]) -> dict[str, str]:
    """Give every unique source URL a stable ID: S1, S2, ... (sorted so runs match)."""
    urls = sorted({url for f in findings.values() for url in f.sources})
    return {f"S{i}": url for i, url in enumerate(urls, start=1)}


def cited_ids(report: Report) -> set[str]:
    """Every source ID the report uses, in its text and its key figures."""
    text = " ".join([report.summary, *(s.body for s in report.sections), *report.risks])
    return set(CITATION.findall(text)) | {k.source_id for k in report.key_figures}


def render_markdown(report: Report, sources: dict[str, str]) -> str:
    """Turn the structured report into markdown, with only the sources actually cited."""
    lines = [f"# {report.title}", "", report.summary, ""]

    if report.key_figures:
        lines += ["## Key figures", "", "| Metric | Value | Source |", "|---|---|---|"]
        lines += [f"| {k.label} | {k.value} | [{k.source_id}] |" for k in report.key_figures]
        lines.append("")

    for section in report.sections:
        lines += [f"## {section.heading}", "", section.body, ""]

    if report.risks:
        lines += ["## Risks", "", *(f"- {r}" for r in report.risks), ""]
    if report.data_gaps:
        lines += ["## Data gaps", "", *(f"- {g}" for g in report.data_gaps), ""]

    # Only list sources that exist and were cited, in numeric order (S2 before S10)
    used = sorted(cited_ids(report) & sources.keys(), key=lambda sid: int(sid[1:]))
    lines += ["## Sources", "", *(f"- [{sid}] {sources[sid]}" for sid in used)]
    lines += ["", f"*{DISCLAIMER}*"]  # added by code, so it can't be left out
    return "\n".join(lines)


def build_prompt(state: ResearchState, sources: dict[str, str]) -> str:
    """Everything the writer needs: question, source list, findings, and any review feedback."""
    plan = state["plan"]
    parts = [
        f"User's question: {state['question']}",
        f"Focus: {plan.focus}",
        "",
        "Sources (cite only these IDs):",
        *(f"{sid}: {url}" for sid, url in sources.items()),
        "",
        "Findings:",
    ]
    for name, f in state["findings"].items():
        parts += [
            f"## {name}",
            f"Summary: {f.summary}",
            # Raw tool results hold the exact numbers the writer must copy
            f"Raw tool data: {json.dumps(f.data.get('tool_calls', []), default=str)}",
            "",
        ]
    if state.get("errors"):
        parts += ["Missing data or errors:", *(f"- {e}" for e in state["errors"]), ""]

    # On a revision, show the last draft and exactly what the reviewer wants fixed
    review = state.get("review")
    if review and not review.approved:
        parts += [
            "Your previous draft:", state.get("draft", ""), "",
            "Fix these issues from the reviewer:", *(f"- {i}" for i in review.issues),
        ]
    return "\n".join(parts)


def make_writer(llm: BaseChatModel | None = None):
    """Build the writer node. The LLM is passed in so tests can use a fake."""
    # Medium thinking: combining several findings needs more reasoning than one tool summary
    writer_llm = (llm or get_llm(thinking_level="medium")).with_structured_output(Report)

    async def writer(state: ResearchState) -> dict:
        findings = state.get("findings", {})
        if not findings:
            return {"errors": ["writer: no findings to write from"]}

        sources = build_source_index(findings)
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(build_prompt(state, sources))]
        try:
            report = await writer_llm.ainvoke(messages)
        # LLM outage or bad output: record it instead of crashing the run
        except Exception as e:
            logger.exception("writer failed")
            return {"errors": [f"writer: {e}"]}

        if report is None:
            return {"errors": ["writer: model returned no report"]}

        update = {
            "report": report,
            "draft": render_markdown(report, sources),
            "revision_count": state.get("revision_count", 0) + 1,  # number of drafts so far
        }
        # Citations to IDs that don't exist are flagged for the reviewer
        unknown = cited_ids(report) - sources.keys()
        if unknown:
            update["errors"] = [f"writer: cited unknown sources {sorted(unknown)}"]
        return update

    return writer