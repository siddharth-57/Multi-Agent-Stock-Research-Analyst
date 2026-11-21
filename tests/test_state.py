"""
test_state.py: Unit tests for the graph state.

Builds a tiny graph with two nodes running in parallel to prove the reducers
merge their updates, and checks that invalid planner output is rejected.
"""
import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from stock_analyst.agents.state import Plan, ResearchState, SpecialistFindings


def test_parallel_nodes_merge_findings_and_errors():
    # Two fake specialists that write to the same state keys at the same time
    def fundamentals(state):
        return {"findings": {"fundamentals": SpecialistFindings(specialist="fundamentals", summary="ok")}}

    def news(state):
        return {
            "findings": {"news": SpecialistFindings(specialist="news", summary="ok")},
            "errors": ["news API slow"],
        }

    graph = StateGraph(ResearchState)
    graph.add_node("fundamentals", fundamentals)
    graph.add_node("news", news)
    # Both edges leave START, so both nodes run in the same step (parallel)
    graph.add_edge(START, "fundamentals")
    graph.add_edge(START, "news")
    graph.add_edge("fundamentals", END)
    graph.add_edge("news", END)

    result = graph.compile().invoke({"question": "test", "errors": []})

    assert set(result["findings"]) == {"fundamentals", "news"}  # nothing was overwritten
    assert result["errors"] == ["news API slow"]


def test_plan_rejects_unknown_specialist():
    # Pydantic blocks the LLM from inventing a specialist that doesn't exist
    with pytest.raises(ValidationError):
        Plan(ticker="AAPL", company_name="Apple", specialists=["astrology"], focus="x")