"""
state.py: Shared state and data models for the agent graph.

ResearchState is the shared record that flows through the graph: each agent
reads what it needs and returns only the fields it changes. The Pydantic
models define what the LLM must output (plan, findings, review), so
malformed LLM responses are caught by validation.
"""
import operator
from typing import Annotated, Literal, Required, TypedDict

from pydantic import BaseModel, Field

# The only specialists that exist; the planner can't invent new ones
SpecialistName = Literal["fundamentals", "market", "news"]


class Plan(BaseModel):
    """Planner output: what to research. Field descriptions are shown to the LLM."""
    ticker: str = Field(description="US stock ticker in upper case, e.g. AAPL")
    company_name: str = Field(description="Company's common name, e.g. Apple")
    specialists: list[SpecialistName] = Field(
        description="Only the specialists needed to answer the question"
    )
    focus: str = Field(description="One sentence on what the user wants to know")


class SpecialistFindings(BaseModel):
    """What one specialist found, with raw data kept for citations and evaluation."""
    specialist: SpecialistName
    summary: str                                      # short written takeaway
    data: dict = Field(default_factory=dict)          # raw tool outputs (the real numbers)
    sources: list[str] = Field(default_factory=list)  # URLs backing the findings
    usage: dict = Field(default_factory=dict)         # rounds, tokens, latency (for evaluation)    


class Review(BaseModel):
    """Reviewer output: pass/fail plus the specific problems to fix."""
    approved: bool
    issues: list[str] = Field(default_factory=list)


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Reducer: each parallel specialist adds its own key instead of overwriting others."""
    return {**(left or {}), **(right or {})}


class ResearchState(TypedDict, total=False):
    question: Required[str]   # the user's question, the only required input
    plan: Plan                # set by the planner

    # Annotated[type, reducer] tells LangGraph how to combine parallel updates
    findings: Annotated[dict[str, SpecialistFindings], merge_dicts]
    errors: Annotated[list[str], operator.add]   # lists from parallel nodes are concatenated

    draft: str                # set by the writer
    review: Review            # set by the reviewer
    revision_count: int       # caps the writer/reviewer loop