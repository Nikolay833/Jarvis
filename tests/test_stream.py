import asyncio
import json
import sys
import types

import httpx
import numpy as np

from jarvis.agent import Agent, Confirmer
from jarvis.audio import tts
from jarvis.audio.tts import ConsoleSpeaker, KokoroSpeaker, SentenceBuffer
from jarvis.bus import EventBus
from jarvis.config import config_from_dict
from jarvis.llm import LLMError, LLMResponse, OllamaClient, ThinkStripper, ToolCall, normalize_keep_alive, strip_think
from jarvis.tools.registry import Registry


# ---- think stripping across chunks -----------------------------------------------------------
def strip_chunks(chunks):
    st = ThinkStripper()
    return "".join(st.feed(c) for c in chunks) + st.flush()


def test_think_split_everywhere():
    raw = "<think>plan\nmore</think>\nHello sir. How are you?"
    for i in range(1, len(raw)):
        assert strip_chunks([raw[:i], raw[i:]]).strip() == strip_think(raw) == "Hello sir. How are you?"
    assert strip_chunks(list(raw)).strip() == "Hello sir. How are you?"


def test_think_misc():
    assert strip_chunks(["Hello"]) == "Hello"
    assert strip_chunks(["<think>unclosed ", "stuff"]) == ""
    assert strip_chunks(["a < b and c<", "d"]) == "a < b and c<d"  # lone '<' is not a tag
    assert strip_chunks(["stray</th", "ink> ok"]).strip() == "stray ok"
    assert strip_chunks(["x<thi"]) == "x<thi"  # held back, released on flush


def test_sentence_buffer():
    b = SentenceBuffer()
    out = []
    for d in ["Good eve", "ning, sir. The wea", "ther is 3.5 degrees", ". Anything", " else?"]:
        out += b.feed(d)
    assert out == ["Good evening, sir.", "The weather is 3.5 degrees."]
    assert b.flush() == ["Anything else?"]


def test_keep_alive_and_config():
    assert normalize_keep_alive("-1") == -1 and normalize_keep_alive(-1) == -1
    assert normalize_keep_alive("30m") == "30m" and normalize_keep_alive("300") == 300
    c = config_from_dict({"ollama": {"keep_alive": -1}})
    assert c.ollama.keep_alive == "-1"
    assert config_from_dict({"ollama": {"keep_alive": "10m"}}).ollama.keep_alive == "10m"
    d = config_from_dict({})
    assert d.audio.silence_seconds == 1.2 and d.ollama.max_reply_tokens == 450 and d.agent.fast_paths


def test_payload_options():
    cl = OllamaClient("http://x", "m", keep_alive="-1", max_reply_tokens=200)
    p = cl.build_payload([], None)
    assert p["keep_alive"] == -1 and p["options"]["num_predict"] == 200
    assert "num_predict" not in OllamaClient("http://x", "m", max_reply_tokens=0).build_payload([], None)["options"]


# ---- OllamaClient.chat_stream over a mocked HTTP transport ------------------------------------
def ndjson(*objs):
    return "".join(json.dumps(o) + "\n" for o in objs).encode()


def stream_client(handler):
    cl = OllamaClient("http://ollama", "m")
    cl._client = httpx.AsyncClient(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return cl


async def collect(cl, msgs=None, tools=None):
    items = []
    async for it in cl.chat_stream(msgs or [{"role": "user", "content": "hi"}], tools):
        items.append(it)
    return items


def test_chat_stream_text_and_think():
    body = ndjson(
        {"message": {"role": "assistant", "content": "<thi"}, "done": False},
        {"message": {"role": "assistant", "content": "nk>hmm</think>"}, "done": False},
        {"message": {"role": "assistant", "content": "\n\nHello"}, "done": False},
        {"message": {"role": "assistant", "content": " sir."}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    )
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, content=body)

    items = asyncio.run(collect(stream_client(handler)))
    assert seen["stream"] is True
    assert "".join(i for i in items if isinstance(i, str)).strip() == "Hello sir."
    final = items[-1]
    assert isinstance(final, LLMResponse) and final.content == "Hello sir." and not final.tool_calls


def test_chat_stream_tool_call():
    call = {"function": {"name": "ping", "arguments": {"x": "1"}}}
    body = ndjson(
        {"message": {"role": "assistant", "content": "", "tool_calls": [call]}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    )
    items = asyncio.run(collect(stream_client(lambda r: httpx.Response(200, content=body))))
    final = items[-1]
    assert [c.name for c in final.tool_calls] == ["ping"] and final.tool_calls[0].arguments == {"x": "1"}
    assert final.raw_tool_calls == [call]


def test_chat_stream_errors():
    import pytest

    with pytest.raises(LLMError):
        asyncio.run(collect(stream_client(lambda r: httpx.Response(500, text="boom"))))

    def refuse(req):
        raise httpx.ConnectError("nope")

    with pytest.raises(LLMError):
        asyncio.run(collect(stream_client(refuse)))
    with pytest.raises(LLMError):
        asyncio.run(collect(stream_client(lambda r: httpx.Response(200, content=b"not json\n"))))


# ---- agent streaming into a speaker ----------------------------------------------------------
class StreamLLM:
    """Scripted steps. A step is a list of str chunks plus an optional ToolCall list."""

    def __init__(self, steps, fail_stream=False):
        self.steps = list(steps)
        self.fail_stream = fail_stream
        self.calls = []
        self.chat_calls = 0

    async def chat_stream(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        if self.fail_stream:
            raise LLMError("stream broke")
        chunks, calls = self.steps.pop(0)
        st = ThinkStripper()
        vis = []
        for c in chunks:
            await asyncio.sleep(0)
            t = st.feed(c)
            if t:
                vis.append(t)
                yield t
        t = st.flush()
        if t:
            vis.append(t)
            yield t
        yield LLMResponse(content="".join(vis).strip(), tool_calls=calls,
                          raw_tool_calls=[{"function": {"name": c.name, "arguments": c.arguments}} for c in calls])

    async def chat(self, messages, tools=None):
        self.chat_calls += 1
        self.calls.append([dict(m) for m in messages])
        return LLMResponse(content="Fallback reply, sir.")


class RecSpeaker(ConsoleSpeaker):
    def __init__(self, bus, log):
        super().__init__(bus)
        self.log = log

    def say_sentence(self, sentence):
        self.log.append(("say", sentence))


def make_agent(llm):
    bus = EventBus()
    reg = Registry()
    ran = []

    @reg.tool()
    def ping(x: str) -> str:
        ran.append(x)
        return f"pong {x}"

    async def speak(t):
        pass

    return Agent(llm, reg, bus, Confirmer(bus, speak, timeout=0.05), max_steps=4, max_history=10), bus, ran


def test_agent_streams_sentences_in_order_before_llm_finishes():
    log = []
    llm = StreamLLM([(["<think>x", "</think>Good eve", "ning, sir. It is", " fine", ". Bye", " now."], [])])
    agent, bus, _ = make_agent(llm)
    sp = RecSpeaker(bus, log)
    reply = asyncio.run(agent.handle("hello", speaker=sp))
    assert reply == "Good evening, sir. It is fine. Bye now."
    assert [s for _, s in log] == ["Good evening, sir.", "It is fine.", "Bye now."]
    assert agent.streamed and agent.timing["first_token"] is not None


def test_agent_tool_step_not_spoken_then_reply():
    log = []
    llm = StreamLLM([([""], [ToolCall("ping", {"x": "1"})]), (["Done, sir."], [])])
    agent, bus, ran = make_agent(llm)
    reply = asyncio.run(agent.handle("ping", speaker=RecSpeaker(bus, log)))
    assert ran == ["1"] and reply == "Done, sir."
    assert log == [("say", "Done, sir.")]


def test_agent_tool_step_unfinished_content_discarded():
    log = []
    llm = StreamLLM([(["Let me check"], [ToolCall("ping", {"x": "1"})]), (["Done."], [])])
    agent, bus, _ = make_agent(llm)
    asyncio.run(agent.handle("ping", speaker=RecSpeaker(bus, log)))
    assert log == [("say", "Done.")]


def test_agent_stream_failure_falls_back_to_chat():
    log = []
    llm = StreamLLM([], fail_stream=True)
    agent, bus, _ = make_agent(llm)
    reply = asyncio.run(agent.handle("hi", speaker=RecSpeaker(bus, log)))
    assert reply == "Fallback reply, sir." and llm.chat_calls == 1
    assert log == [("say", "Fallback reply, sir.")]


def test_agent_without_speaker_uses_chat_and_does_not_stream():
    llm = StreamLLM([])
    agent, *_ = make_agent(llm)
    assert asyncio.run(agent.handle("hi")) == "Fallback reply, sir."
    assert llm.chat_calls == 1 and not agent.streamed


# ---- static prompt, time prefix, history -------------------------------------------------------
def test_system_prompt_static_and_now_prefix_only_on_latest_user():
    llm = StreamLLM([(["One."], []), (["Two."], [])])
    agent, *_ = make_agent(llm)
    asyncio.run(agent.handle("first", speaker=None))  # chat() path
    llm2 = StreamLLM([(["One."], []), (["Two."], [])])
    agent2, bus, _ = make_agent(llm2)
    sp = RecSpeaker(bus, [])
    asyncio.run(agent2.handle("first", speaker=sp))
    asyncio.run(agent2.handle("second", speaker=sp))
    c1, c2 = llm2.calls
    assert c1[0] == c2[0] and "{now}" not in c1[0]["content"] and "Current date" not in c1[0]["content"]
    assert c1[-1]["content"].startswith("[Now: ") and c1[-1]["content"].endswith("]\nfirst")
    assert c2[-1]["content"].startswith("[Now: ") and c2[-1]["content"].endswith("]\nsecond")
    assert c2[1]["content"] == "first"  # older user turn has no prefix
    assert sum(m["content"].startswith("[Now:") for m in c2 if m["role"] == "user") == 1
    hist = agent2.history("default")
    assert [m["content"] for m in hist if m["role"] == "user"] == ["first", "second"]


def test_prefix_stable_across_tool_steps_and_not_in_history():
    llm = StreamLLM([([""], [ToolCall("ping", {"x": "1"})]), (["Ok."], [])])
    agent, bus, _ = make_agent(llm)
    asyncio.run(agent.handle("go", speaker=RecSpeaker(bus, [])))
    u0 = [m for m in llm.calls[0] if m["role"] == "user"][0]["content"]
    u1 = [m for m in llm.calls[1] if m["role"] == "user"][0]["content"]
    assert u0 == u1 and u0.startswith("[Now: ")
    assert all(not str(m.get("content", "")).startswith("[Now:") for m in agent.history("default"))


def test_add_exchange():
    agent, *_ = make_agent(StreamLLM([]))
    agent.add_exchange("what time is it", "It's 6:42 PM, sir.")
    agent.add_exchange("stop", "")
    assert agent.history("default") == [{"role": "user", "content": "what time is it"},
                                        {"role": "assistant", "content": "It's 6:42 PM, sir."}]


# ---- speaker streams ---------------------------------------------------------------------------
def test_console_stream_ordering(capsys):
    async def run():
        sp = ConsoleSpeaker(EventBus())
        s = sp.start_stream()
        s.push("One.")
        s.push("Two.")
        await s.finish()
        return s

    s = asyncio.run(run())
    assert capsys.readouterr().out.splitlines() == ["Jarvis: One.", "Jarvis: Two."]
    assert s.first_audio_at is not None


def test_kokoro_stream_overlaps_synthesis_and_playback(monkeypatch):
    events = []

    class FakeOut:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def write(self, block):
            if not events or events[-1] != ("play", self.cur):
                events.append(("play", self.cur))

        cur = None

    fake_sd = types.SimpleNamespace(OutputStream=FakeOut)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    class Bus:
        def emit_nowait(self, *a, **k):
            pass

    sp = KokoroSpeaker(Bus())
    sp._pipeline = object()

    def synth(sentence):
        events.append(("synth", sentence))
        FakeOut.cur = sentence
        return np.ones(1600, dtype=np.float32)

    monkeypatch.setattr(sp, "synth", synth)

    async def run():
        s = sp.start_stream()
        s.push("One.")
        await asyncio.sleep(0.05)
        s.push("Two.")
        s.push("Three.")
        await s.finish()
        return s

    s = asyncio.run(run())
    synths = [e[1] for e in events if e[0] == "synth"]
    assert synths == ["One.", "Two.", "Three."]
    assert s.first_audio_at is not None and s.pushed == 3
    # sentence one is synthesized before the later ones were even pushed
    assert events[0] == ("synth", "One.")
    # a second utterance afterwards works (lock released)
    asyncio.run(sp.speak("Again. And again."))
