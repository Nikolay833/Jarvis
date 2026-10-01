import asyncio

import pytest

from jarvis.agent import CLAUDE_GUARD_MSG, Agent
from jarvis.llm import LLMResponse, ToolCall
from jarvis.tools.registry import Registry


class Bus:
    def emit_nowait(self, *a, **k):
        pass

    async def emit(self, *a, **k):
        pass


class Confirm:
    async def ask(self, summary):
        return True


class FakeLLM:
    def __init__(self, steps):
        self.steps = list(steps)
        self.seen_tool_results = []

    async def chat(self, messages, tools=None):
        for m in messages:
            if m.get("role") == "tool":
                self.seen_tool_results.append(m.get("content"))
        return self.steps.pop(0)


def make(steps):
    reg = Registry()
    calls = []

    @reg.tool("open a claude session")
    def claude_open_session(topic: str = "") -> str:
        """Open.

        Args:
            topic: topic
        """
        calls.append(topic)
        return "opened"

    llm = FakeLLM(steps)
    agent = Agent(llm, reg, Bus(), Confirm(), stream_replies=False)
    return agent, llm, calls


def call(topic=""):
    return LLMResponse(content="", tool_calls=[ToolCall("claude_open_session", {"topic": topic})])


@pytest.mark.parametrize("text", ["open", "open the latest one", "open my project"])
def test_blocks_claude_tools_without_mention(text):
    agent, llm, calls = make([call(), LLMResponse(content="Open what, sir?")])
    reply = asyncio.run(agent.handle(text))
    assert calls == [] and CLAUDE_GUARD_MSG in llm.seen_tool_results and reply == "Open what, sir?"


def test_allows_when_claude_mentioned():
    agent, llm, calls = make([call("login"), LLMResponse(content="The session is open, sir.")])
    asyncio.run(agent.handle("open the login session in Claude"))
    assert calls == ["login"]


def test_allows_follow_up_answer_to_claude_question():
    agent, llm, calls = make([LLMResponse(content="Which session, sir? The login one or the settings one?"),
                              call("login"), LLMResponse(content="The session is open, sir.")])
    asyncio.run(agent.handle("open my claude session"))
    asyncio.run(agent.handle("the first one"))
    assert calls == ["login"]


def test_bare_open_fast_path():
    from jarvis import fastpath

    fp = fastpath.match("open")
    assert fp is not None and fp.kind == "clarify" and fp.reply.endswith("?")
    assert fastpath.match("open chrome").kind == "open_app"
