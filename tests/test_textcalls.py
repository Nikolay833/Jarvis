import asyncio
import json

import pytest

from jarvis.agent import Agent, Confirmer, looks_like_announcement, NUDGE
from jarvis.bus import EventBus
from jarvis.llm import LLMResponse, ToolCall, ToolCallStreamFilter, extract_text_tool_calls
from jarvis.tools.registry import Registry
from tests.test_stream import RecSpeaker, StreamLLM

KNOWN = {"ping", "create_folder"}


def run_filter(chunks):
    f = ToolCallStreamFilter()
    return "".join(f.feed(c) for c in chunks) + f.flush(), f.triggered


# ---- extraction -------------------------------------------------------------------------------
def test_extract_tag_form():
    c, calls = extract_text_tool_calls(
        'On it.<tool_call>\n{"name": "ping", "arguments": {"x": "1"}}\n</tool_call>', KNOWN)
    assert c == "On it." and calls == [ToolCall("ping", {"x": "1"})]


def test_extract_multiple_tags_and_unclosed():
    txt = ('<tool_call>{"name":"ping","arguments":{"x":"a"}}</tool_call>'
           '<tool_call>{"name":"create_folder","arguments":{"path":"d"}}</tool_call>')
    _, calls = extract_text_tool_calls(txt, KNOWN)
    assert [c.name for c in calls] == ["ping", "create_folder"]
    _, calls = extract_text_tool_calls('<tool_call>{"name":"ping","arguments":{"x":"a"}}', KNOWN)
    assert len(calls) == 1


def test_extract_bare_and_fenced_and_parameters():
    c, calls = extract_text_tool_calls('{"name": "ping", "parameters": {"x": "2"}}', KNOWN)
    assert c == "" and calls == [ToolCall("ping", {"x": "2"})]
    c, calls = extract_text_tool_calls('Sure.\n```json\n{"name": "ping", "arguments": {"x": "3"}}\n```', KNOWN)
    assert c == "Sure." and calls[0].arguments == {"x": "3"}
    _, calls = extract_text_tool_calls('[{"name":"ping","arguments":{"x":"1"}},{"name":"ping","arguments":{"x":"2"}}]',
                                       KNOWN)
    assert len(calls) == 2
    _, calls = extract_text_tool_calls('{"name": "ping", "arguments": "{\\"x\\": \\"s\\"}"}', KNOWN)
    assert calls[0].arguments == {"x": "s"}


def test_extract_rejects_unknown_and_plain():
    t = '{"name": "rm_rf", "arguments": {}}'
    assert extract_text_tool_calls(t, KNOWN) == (t, [])
    t = 'My name is {"name": "ping"} ok'  # no arguments key: not a call
    assert extract_text_tool_calls(t, KNOWN) == (t, [])
    assert extract_text_tool_calls("Hello sir.", KNOWN) == ("Hello sir.", [])
    assert extract_text_tool_calls('{"name":"ping","arguments":{}}', set()) [1] == []


# ---- stream filter ----------------------------------------------------------------------------
CALL = '<tool_call>{"name": "ping", "arguments": {"x": "1"}}</tool_call>'
BARE = '{"name": "ping", "arguments": {"x": "1"}}'


@pytest.mark.parametrize("raw,spoken", [
    ("I'll check. " + CALL, "I'll check. "),
    (CALL, ""),
    (BARE, ""),
    ('```json\n' + BARE + '\n```', ""),
    ("Sure.\n" + BARE, "Sure.\n"),
    ("Plain reply, sir.", "Plain reply, sir."),
    ("a < b and {curly} text", "a < b and {curly} text"),
    ('{"other": 1} stays', '{"other": 1} stays'),
    ("Use <b>bold</b>", "Use <b>bold</b>"),
])
def test_filter_every_split(raw, spoken):
    for i in range(0, len(raw) + 1):
        out, _ = run_filter([raw[:i], raw[i:]])
        assert out == spoken, (i, out)
    assert run_filter(list(raw))[0] == spoken


def test_filter_triggered_flag():
    assert run_filter(["<tool", "_call>{"])[1] is True
    assert run_filter(["hello"])[1] is False


# ---- agent: streaming with text tool calls ----------------------------------------------------
def make(llm):
    bus = EventBus()
    reg = Registry()
    ran = []

    @reg.tool()
    def ping(x: str) -> str:
        ran.append(x)
        return f"pong {x}"

    async def speak(t):
        pass

    return Agent(llm, reg, bus, Confirmer(bus, speak, timeout=0.05), max_steps=5, max_history=10), bus, ran


def test_agent_runs_text_tool_call_and_never_speaks_json():
    log = []
    chunks = ["Right away", ", sir. <tool", "_call>{\"name\": \"ping\", ", "\"arguments\": {\"x\": \"7\"}}", "</tool_call>"]
    llm = StreamLLM([(chunks, []), (["Done, sir."], [])])
    agent, bus, ran = make(llm)
    asyncio.run(agent.handle("go", speaker=RecSpeaker(bus, log)))
    assert ran == ["7"]
    assert all("tool_call" not in s and "{" not in s for _, s in log)
    assert log[-1] == ("say", "Done, sir.")


def test_agent_recovers_bare_json_without_speaker():
    class L(StreamLLM):
        async def chat(self, messages, tools=None):
            self.chat_calls += 1
            if self.chat_calls == 1:
                return LLMResponse(content='{"name": "ping", "arguments": {"x": "9"}}')
            return LLMResponse(content="All done.")

    agent, _, ran = make(L([]))
    assert asyncio.run(agent.handle("go")) == "All done." and ran == ["9"]


# ---- promise-without-action guard ---------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("I'll check that for you, sir.", True),
    ("Certainly, sir. Let me look at that.", True),
    ("One moment, sir.", True),
    ("Creating the folder now.", True),
    ("On it, sir.", True),
    ("Let's see.", True),
    ("Allow me a moment.", True),
    ("Done, sir. I created the folder.", False),
    ("Shall I delete it, sir?", False),
    ("I'll check that, sir. Which folder?", False),
    ("The time is 6 pm.", False),
    ("Let me know if you need anything else, sir.", False),
    ("", False),
])
def test_looks_like_announcement(text, expected):
    assert looks_like_announcement(text) is expected


def test_nudge_then_tool_call_and_history_clean():
    log = []
    llm = StreamLLM([(["I'll check that for you, sir."], []), ([""], [ToolCall("ping", {"x": "1"})]),
                     (["It says pong, sir."], [])])
    agent, bus, ran = make(llm)
    reply = asyncio.run(agent.handle("check", speaker=RecSpeaker(bus, log)))
    assert ran == ["1"] and reply == "It says pong, sir."
    assert llm.calls[1][-1]["content"] == NUDGE and llm.calls[1][-1]["role"] == "user"
    # time prefix stays on the real user message, not the nudge
    assert llm.calls[1][-1]["content"].startswith("(system)")
    assert [m["content"] for m in llm.calls[1] if m["role"] == "user"][0].startswith("[Now:")
    assert all("_nudge" not in m for m in llm.calls[1])
    hist = agent.history("default")
    assert not any(NUDGE in str(m.get("content")) for m in hist)
    assert not any("I'll check" in str(m.get("content")) for m in hist)
    assert [s for _, s in log][0] == "I'll check that for you, sir."  # acknowledgement was spoken


def test_nudge_capped_at_two():
    llm = StreamLLM([(["I'll do it."], [])] * 4)
    agent, bus, ran = make(llm)
    reply = asyncio.run(agent.handle("x", speaker=RecSpeaker(bus, [])))
    assert len(llm.calls) == 3 and reply == "I'll do it." and ran == []


def test_no_nudge_for_question_or_result():
    llm = StreamLLM([(["Which folder, sir?"], [])])
    agent, bus, _ = make(llm)
    asyncio.run(agent.handle("x", speaker=RecSpeaker(bus, [])))
    assert len(llm.calls) == 1


def test_no_nudge_without_tools():
    llm = StreamLLM([(["I'll do it."], [])])
    agent, bus, _ = make(llm)
    agent.registry = Registry()
    asyncio.run(agent.handle("x", speaker=RecSpeaker(bus, [])))
    assert len(llm.calls) == 1


def test_turn_logs_tools(caplog):
    import logging

    llm = StreamLLM([([""], [ToolCall("ping", {"x": "1"})]), (["Ok."], [])])
    agent, bus, _ = make(llm)
    with caplog.at_level(logging.INFO, logger="jarvis.agent"):
        asyncio.run(agent.handle("x", speaker=RecSpeaker(bus, [])))
        llm2 = StreamLLM([(["Hello."], [])])
        agent2, bus2, _ = make(llm2)
        asyncio.run(agent2.handle("x", speaker=RecSpeaker(bus2, [])))
    text = caplog.text
    assert "tools called ['ping']" in text and "no tools called" in text
