"""
fakes.py: Test helpers shared across test files.

ScriptedLLM replays a fixed list of replies, so agent logic can be tested
without calling a real model. ask() builds a tool-call reply, answer() a text reply.
"""
from langchain_core.messages import AIMessage

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def ask(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """A model reply that requests one tool call."""
    return AIMessage(
        content="", usage_metadata=USAGE,
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def answer(text: str) -> AIMessage:
    """A model reply with a final text answer."""
    return AIMessage(content=text, usage_metadata=USAGE)


class ScriptedLLM:
    """Fake model: returns replies in order, repeating the last one forever."""
    def __init__(self, replies: list[AIMessage]):
        self.replies = replies
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply

class FixedOutputLLM:
    """Fake model for structured output: always returns the object it was given."""
    def __init__(self, output):
        self.output = output

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        return self.output