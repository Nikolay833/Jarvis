import pytest
import asyncio

from jarvis.agent import Agent, Confirmer, parse_yes_no
from jarvis.bus import EventBus
from jarvis.llm import LLMError, LLMResponse, ToolCall, strip_think
from jarvis.tools.registry import Registry


class FakeLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def chat(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            raise LLMError("script exhausted")
        return self.script.pop(0)


def tc(name, **args):
    return LLMResponse(content="", tool_calls=[ToolCall(name, args)],
                       raw_tool_calls=[{"function": {"name": name, "arguments": args}}])


def say(text):
    return LLMResponse(content=text)


def setup(script, timeout=0.05, approve=None):
    bus = EventBus()
    events = []
    bus.add_listener(events.append)
    reg = Registry()
    ran = []

    @reg.tool()
    def ping(x: str) -> str:
        ran.append(("ping", x))
        return f"pong {x}"

    @reg.tool(risk="risky")
    def nuke(target: str) -> str:
        ran.append(("nuke", target))
        return "boom"

    spoken = []

    async def speak(t):
        spoken.append(t)
        if approve is not None:
            for ev in reversed(events):
                if ev["type"] == "confirm":
                    conf.resolve(ev["id"], approve)
                    break

    conf = Confirmer(bus, speak, timeout=timeout)
    llm = FakeLLM(script)
    return Agent(llm, reg, bus, conf, max_steps=4, max_history=6), llm, ran, events, spoken


def test_plain_reply():
    agent, llm, *_ = setup([say("Good morning, sir.")])
    assert asyncio.run(agent.handle("hi")) == "Good morning, sir."
    assert llm.calls[0][0]["role"] == "system"


def test_safe_tool_loop():
    agent, llm, ran, *_ = setup([tc("ping", x="1"), say("Done, sir.")])
    assert asyncio.run(agent.handle("ping it")) == "Done, sir."
    assert ran == [("ping", "1")]
    tool_msg = llm.calls[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"] == "pong 1"


def test_unknown_tool_reported():
    agent, llm, *_ = setup([tc("nope"), say("Sorry.")])
    asyncio.run(agent.handle("x"))
    assert "unknown tool" in llm.calls[1][-1]["content"]


def test_max_steps():
    agent, llm, ran, *_ = setup([tc("ping", x=str(i)) for i in range(10)])
    reply = asyncio.run(agent.handle("loop"))
    assert len(ran) == 4 and "steps" in reply


def test_risky_approved():
    agent, llm, ran, events, spoken = setup([tc("nuke", target="x"), say("Done.")], approve=True)
    assert asyncio.run(agent.handle("nuke")) == "Done."
    assert ran == [("nuke", "x")]
    assert "Shall I proceed" in spoken[0]
    types = [e["type"] for e in events]
    assert "confirm" in types and "confirm_resolved" in types
    assert [e for e in events if e["type"] == "confirm_resolved"][0]["approved"] is True


def test_risky_denied():
    agent, llm, ran, events, _ = setup([tc("nuke", target="x"), say("Cancelled.")], approve=False)
    asyncio.run(agent.handle("nuke"))
    assert ran == []
    assert "NOT performed" in llm.calls[1][-1]["content"]


def test_confirm_timeout_is_no():
    agent, llm, ran, events, _ = setup([tc("nuke", target="x"), say("Cancelled.")], timeout=0.05)
    asyncio.run(agent.handle("nuke"))
    assert ran == []
    res = [e for e in events if e["type"] == "confirm_resolved"]
    assert res and res[0]["approved"] is False


def test_confirm_by_voice_listener():
    async def run():
        bus = EventBus()

        async def speak(t):
            pass

        async def listen():
            return "yes please"

        c = Confirmer(bus, speak, timeout=2, listen=listen)
        return await c.ask("do a thing")

    assert asyncio.run(run()) is True


def test_confirm_resolve_text_and_id():
    async def run():
        bus = EventBus()
        events = []
        bus.add_listener(events.append)

        async def speak(t):
            pass

        c = Confirmer(bus, speak, timeout=2)
        task = asyncio.create_task(c.ask("x"))
        await asyncio.sleep(0.05)
        assert c.pending
        assert not c.resolve_text("what?")
        assert c.resolve_text("no")
        return await task

    assert asyncio.run(run()) is False


def test_llm_error_message_and_history_clean():
    agent, *_ = setup([])
    reply = asyncio.run(agent.handle("hi"))
    assert "Ollama" in reply
    assert agent.history("default") == []


def test_trim_starts_at_user():
    agent, *_ = setup([])
    h = agent.history("s")
    for i in range(5):
        h += [{"role": "user", "content": str(i)}, {"role": "assistant", "content": "a"},
              {"role": "tool", "content": "t"}]
    agent.trim("s")
    h = agent.history("s")
    assert len(h) <= 6 and h[0]["role"] == "user"


def test_sessions_separate():
    agent, *_ = setup([say("a"), say("b")])
    asyncio.run(agent.handle("one", session="s1"))
    asyncio.run(agent.handle("two", session="s2"))
    assert len(agent.history("s1")) == 2 and len(agent.history("s2")) == 2


def test_parse_yes_no():
    assert parse_yes_no("Yes, go ahead") is True
    assert parse_yes_no("no don't") is False
    assert parse_yes_no("yes no") is False
    assert parse_yes_no("hmm") is None
    assert parse_yes_no("") is None


def test_strip_think():
    assert strip_think("<think>plan\nmore</think>\nHello sir.") == "Hello sir."
    assert strip_think("Hello") == "Hello"
    assert strip_think("<think>unclosed") == ""
    assert strip_think("stray</think> ok") == "ok"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("do not do it", False),
        ("hold on", False),
        ("wait", False),
        ("please", None),
        ("do it", True),
        ("yeah go ahead", True),
        ("of course", True),
    ],
)
def test_parse_yes_no_negation_and_phrases(text, expected):
    from jarvis.agent import parse_yes_no

    assert parse_yes_no(text) is expected
