"""
checks.py: Exact, code-based checks on a report (no LLM involved).

Verifies what can be verified precisely: every cited source ID exists, the
report has its required parts, and every key figure's number really appears
in the tool data behind the source it cites. Used by the reviewer now and
by the evaluation harness later.
"""
import math
import re

from stock_analyst.agents.specialists import collect_sources
from stock_analyst.agents.state import Report, SpecialistFindings
from stock_analyst.agents.writer import cited_ids

NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")  # matches 27.6, -13.8, 1,234.5


def numbers_in(text: str) -> list[float]:
    """Pull every number out of a string, e.g. '$478.83B' -> [478.83]."""
    return [float(n.replace(",", "")) for n in NUMBER.findall(text)]


def collect_numbers(obj) -> list[float]:
    """Every number in a tool result: numeric values plus numbers in 'display' strings.
    Other strings (URLs, dates) are skipped so their digits can't cause false matches."""
    found: list[float] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "display" and isinstance(value, str):
                found += numbers_in(value)
            else:
                found += collect_numbers(value)
    elif isinstance(obj, list):
        for item in obj:
            found += collect_numbers(item)
    elif isinstance(obj, int | float) and not isinstance(obj, bool):  # bool is a subclass of int
        found.append(float(obj))
    return found


def numbers_by_source(findings: dict[str, SpecialistFindings]) -> dict[str, list[float]]:
    """For each source URL, all numbers from the tool calls that came from that source."""
    index: dict[str, list[float]] = {}
    for f in findings.values():
        for call in f.data.get("tool_calls", []):
            numbers = collect_numbers(call["result"])
            for url in collect_sources(call):
                index.setdefault(url, []).extend(numbers)
    return index


def matches(value: float, candidates: list[float]) -> bool:
    # Tiny tolerance: 27.6 matches 27.60, but 478.8 does not match 478.83
    return any(math.isclose(value, c, rel_tol=1e-9, abs_tol=0.006) for c in candidates)


def check_report(
    report: Report, findings: dict[str, SpecialistFindings], sources: dict[str, str]
) -> list[str]:
    """Return a list of problems; an empty list means every exact check passed."""
    issues: list[str] = []

    unknown = cited_ids(report) - sources.keys()
    if unknown:
        issues.append(f"Cites source IDs that don't exist: {sorted(unknown)}")
    if not report.key_figures:
        issues.append("Report has no key figures.")
    if not report.sections:
        issues.append("Report has no sections.")

    by_source = numbers_by_source(findings)
    for fig in report.key_figures:
        url = sources.get(fig.source_id)
        if url is None:
            continue  # already reported above as an unknown ID
        for n in numbers_in(fig.value):
            if not matches(n, by_source.get(url, [])):
                issues.append(
                    f"Key figure '{fig.label}' = '{fig.value}': {n} does not appear "
                    f"in the data from [{fig.source_id}]. Copy the exact value or fix the citation."
                )
    return issues