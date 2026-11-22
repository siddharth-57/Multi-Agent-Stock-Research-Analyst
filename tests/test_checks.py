"""
test_checks.py: Unit tests for the exact report checks.

Uses small fake findings with two sources (an SEC filing and a price page)
to check that correct figures pass, and that wrong numbers, numbers cited
to the wrong source, and unknown source IDs are all flagged.
"""
from stock_analyst.agents.checks import check_report
from stock_analyst.agents.state import (
    KeyFigure,
    Report,
    ReportSection,
    SpecialistFindings,
)
from stock_analyst.agents.writer import build_source_index

SEC = "https://sec.example/filing"
YAHOO = "https://yahoo.example/aapl"

FINDINGS = {
    "fundamentals": SpecialistFindings(
        specialist="fundamentals", summary="", sources=[SEC, YAHOO],
        data={"tool_calls": [
            {"name": "compute_ratios", "args": {"ticker": "AAPL"},
             "result": {"net_margin_pct": 27.6, "pe_ratio": 38.15, "sources": [SEC, YAHOO]}},
            {"name": "get_financials", "args": {"ticker": "AAPL"},
             "result": {"metrics": {"revenue": [
                 {"value": 478834000000.0, "display": "$478.83B", "source": SEC}]}}},
        ]},
    ),
    "market": SpecialistFindings(
        specialist="market", summary="", sources=[YAHOO],
        data={"tool_calls": [
            {"name": "get_price_summary", "args": {"ticker": "AAPL"},
             "result": {"latest_close": 342.56, "source_url": YAHOO}},
        ]},
    ),
}
SOURCES = build_source_index(FINDINGS)  # S1 = SEC, S2 = YAHOO (sorted)


def report_with(*figures: tuple[str, str, str]) -> Report:
    """Build a minimal valid report with the given (label, value, source_id) figures."""
    return Report(
        title="t", summary="s",
        key_figures=[KeyFigure(label=label, value=value, source_id=sid) for label, value, sid in figures],
        sections=[ReportSection(heading="h", body="b")], risks=[],
    )


def test_correct_figures_pass():
    report = report_with(
        ("Net margin", "27.60%", "S1"),     # 27.60 matches 27.6
        ("Revenue", "$478.83B", "S1"),      # matches the display value
        ("Latest close", "$342.56", "S2"),
    )
    assert check_report(report, FINDINGS, SOURCES) == []


def test_wrong_number_is_flagged():
    issues = check_report(report_with(("Net margin", "28.1%", "S1")), FINDINGS, SOURCES)
    assert len(issues) == 1 and "28.1" in issues[0]


def test_right_number_wrong_source_is_flagged():
    # 342.56 is real, but it comes from the price page (S2), not the SEC filing (S1)
    issues = check_report(report_with(("Latest close", "342.56", "S1")), FINDINGS, SOURCES)
    assert len(issues) == 1


def test_unknown_source_id_is_flagged():
    issues = check_report(report_with(("Something", "1", "S9")), FINDINGS, SOURCES)
    assert any("don't exist" in i for i in issues)