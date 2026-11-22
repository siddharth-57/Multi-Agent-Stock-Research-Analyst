"""
loop.py: Our own tool-calling agent loop, shared by all specialists.

Asks the LLM, runs any tools it requests (in parallel), sends the results
back, and repeats until the LLM answers or the round limit is hit. Records
every tool call plus token usage and latency for the evaluation harness.
"""
import asyncio
import json
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

LIMIT_REPLY = {"error": "Tool limit reached. Answer now using the results you already have."}


class LoopResult(BaseModel):
    """Everything one agent run produced, plus what it cost."""
    final_text: str = ""
    tool_calls: list[dict] = Field(default_factory=list)  # name, args, result of each call
    rounds: int = 0            # rounds in which tools were actually run
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    hit_limit: bool = False    # True if the round cap forced the answer


async def run_tool_loop(
    llm: BaseChatModel,
    tools: list[BaseTool],
    messages: list[BaseMessage],
    max_rounds: int = 4,
) -> LoopResult:
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}
    result = LoopResult()
    start = time.perf_counter()

    async def call_llm() -> AIMessage:
        response = await llm_with_tools.ainvoke(messages)
        # usage_metadata is LangChain's standard token count; some models omit it
        usage = response.usage_metadata or {}
        result.input_tokens += usage.get("input_tokens", 0)
        result.output_tokens += usage.get("output_tokens", 0)
        messages.append(response)  # the model needs the full history next round
        return response

    async def run_tool(call: dict):
        tool = tools_by_name.get(call["name"])
        if tool is None:  # the model asked for a tool it doesn't have
            return {"error": f"unknown tool '{call['name']}'"}
        return await tool.ainvoke(call["args"])

    response = await call_llm()
    while response.tool_calls:
        over_budget = result.rounds >= max_rounds

        if over_budget:
            # Every tool call must get a reply, so answer each one with the limit message
            result.hit_limit = True
            outputs = [LIMIT_REPLY] * len(response.tool_calls)
        else:
            result.rounds += 1
            # Independent tool calls run at the same time
            outputs = await asyncio.gather(*(run_tool(c) for c in response.tool_calls))
            for call, output in zip(response.tool_calls, outputs, strict=True):
                result.tool_calls.append({"name": call["name"], "args": call["args"], "result": output})

        for call, output in zip(response.tool_calls, outputs, strict=True):
            # tool_call_id links each result to the request it answers
            messages.append(ToolMessage(content=json.dumps(output, default=str), tool_call_id=call["id"]))

        response = await call_llm()
        if over_budget:
            break  # one final answer after the limit, never more tool rounds

    result.final_text = response.text
    result.latency_s = round(time.perf_counter() - start, 2)
    return result