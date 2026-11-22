"""
test_loop.py: Unit tests for the tool-calling loop.

Checks the normal flow (tool call, then answer), token counting,
unknown tool names, and that the round cap stops a model that never stops.
"""
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from stock_analyst.agents.loop import run_tool_loop
from tests.fakes import ScriptedLLM, answer, ask


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def test_runs_tool_then_answers():
    llm = ScriptedLLM([ask("add", {"a": 1, "b": 2}), answer("The sum is 3")])
    messages = [HumanMessage("what is 1+2?")]
    r = await run_tool_loop(llm, [add], messages)

    assert r.final_text == "The sum is 3"
    assert r.tool_calls == [{"name": "add", "args": {"a": 1, "b": 2}, "result": 3}]
    assert r.rounds == 1 and not r.hit_limit
    assert (r.input_tokens, r.output_tokens) == (20, 10)  # 2 LLM calls x (10 in, 5 out)
    # History: question -> tool request -> tool result -> answer
    assert isinstance(messages[2], ToolMessage)


async def test_unknown_tool_gets_error_reply():
    llm = ScriptedLLM([ask("delete_everything", {}), answer("ok")])
    r = await run_tool_loop(llm, [add], [HumanMessage("hi")])
    assert r.tool_calls[0]["result"] == {"error": "unknown tool 'delete_everything'"}


async def test_round_cap_stops_endless_tool_calls():
    llm = ScriptedLLM([ask("add", {"a": 1, "b": 1})])  # this model never stops asking
    r = await run_tool_loop(llm, [add], [HumanMessage("loop forever")], max_rounds=2)

    assert r.hit_limit and r.rounds == 2
    assert len(r.tool_calls) == 2   # only real tool runs are logged
    assert llm.calls == 4           # first call + 2 rounds + 1 final call after the limit