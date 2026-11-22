"""
test_writer.py: Unit tests for the writer agent.

Checks source numbering, citation detection, markdown rendering (only cited
sources, disclaimer always present), and the writer node's handling of
unknown source IDs and missing findings. Uses a fake LLM.
"""
from stock_analyst.agents.state import (
    KeyFigure,
    Plan,
    Report,
    ReportSection,
    SpecialistFindings,
)
from stock_analyst.agents.writer import (
    DISCLAIMER,
    build_source_index,
    cited_ids,
    make_writer,
    render_markdown,
)
from tests.fakes import FixedOutputLLM

PLAN = Plan(ticker="AAPL", company_name="Apple", specialists=["fundamentals"], focus="test")
FINDINGS = {
    "fundamentals": SpecialistFindings(
        specialist="fundamentals", summary="ok",
        sources=["https://sec.example/b", "https://sec.example/a"],
    )
}


def make_report(source_id: str = "S1") -> Report:
    return Report(
        title="Apple check",
        summary=f"Margins are strong [{source_id}].",
        key_figures=[KeyFigure(label="Net margin (FY2026)", value="27.6%", source_id=source_id)],
        sections=[ReportSection(heading="Profitability", body="High margins [S1].")],
        risks=["Valuation is high."],
    )


def test_source_ids_are_sorted_and_numbered():
    assert build_source_index(FINDINGS) == {"S1": "https://sec.example/a", "S2": "https://sec.example/b"}


def test_cited_ids_found_in_text_and_figures():
    assert cited_ids(make_report("S2")) == {"S1", "S2"}


def test_render_lists_only_cited_sources_and_disclaimer():
    md = render_markdown(make_report(), build_source_index(FINDINGS))
    assert "[S1] https://sec.example/a" in md
    assert "https://sec.example/b" not in md   # S2 was never cited
    assert DISCLAIMER in md


async def test_writer_builds_draft():
    writer = make_writer(llm=FixedOutputLLM(make_report()))
    out = await writer({"question": "q", "plan": PLAN, "findings": FINDINGS})

    assert out["draft"].startswith("# Apple check")
    assert out["revision_count"] == 1
    assert "errors" not in out


async def test_unknown_source_id_is_flagged():
    writer = make_writer(llm=FixedOutputLLM(make_report("S9")))
    out = await writer({"question": "q", "plan": PLAN, "findings": FINDINGS})
    assert out["errors"] == ["writer: cited unknown sources ['S9']"]


async def test_no_findings_is_an_error():
    writer = make_writer(llm=FixedOutputLLM(make_report()))
    out = await writer({"question": "q", "plan": PLAN})
    assert out == {"errors": ["writer: no findings to write from"]}