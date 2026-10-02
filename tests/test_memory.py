import asyncio

import pytest

from jarvis import memory
from jarvis.agent import Agent, Confirmer, SYSTEM_PROMPT, full_system_prompt
from jarvis.bus import EventBus
from jarvis.llm import LLMResponse
from jarvis.memory import FactError, Memory, second_person
from jarvis.tools import load_all
from jarvis.tools.registry import Registry


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 1
        return self.t


def mem(tmp_path, **kw):
    return Memory(tmp_path / "memory.json", clock=Clock(), **kw)


def test_add_and_persist(tmp_path):
    m = mem(tmp_path)
    fact, created = m.add("my main project is Jarvis")
    assert created and fact["source"] == "explicit" and len(fact["id"]) == 8
    assert [f["text"] for f in Memory(tmp_path / "memory.json").facts()] == ["my main project is Jarvis"]


def test_dedupe_same_and_near_duplicate_updates(tmp_path):
    m = mem(tmp_path)
    m.add("I prefer Spotify", "inferred")
    _, created = m.add("i prefer spotify.")
    assert not created
    _, created = m.add("User prefers Spotify")  # same tokens after stemming
    assert not created
    facts = m.facts()
    assert len(facts) == 1 and facts[0]["text"] == "User prefers Spotify"
    assert facts[0]["source"] == "explicit"  # an explicit save upgrades an inferred fact


def test_negation_is_not_a_duplicate(tmp_path):
    m = mem(tmp_path)
    m.add("I like tea")
    _, created = m.add("I don't like tea")
    assert created and len(m.facts()) == 2


def test_different_facts_kept(tmp_path):
    m = mem(tmp_path)
    m.add("my main project is Jarvis")
    m.add("my dog is called Rex")
    assert len(m.facts()) == 2


def test_cap_drops_inferred_first(tmp_path):
    m = mem(tmp_path, max_facts=5)
    m.add("explicit alpha fact one", "explicit")
    for i in range(7):
        m.add(f"inferred thing{i} uniqueword{i}", "inferred")
    facts = m.facts()
    assert len(facts) == 5
    assert any(f["text"].startswith("explicit alpha") for f in facts)


def test_secrets_and_empty_refused(tmp_path):
    m = mem(tmp_path)
    for bad in ("my password is hunter2", "the api key is abc", "my pin is 1234", "", "x" * 400):
        with pytest.raises(FactError):
            m.add(bad)
    assert m.facts() == []


def test_forget_best_match(tmp_path):
    m = mem(tmp_path)
    m.add("my favourite colour is green")
    m.add("my main project is Jarvis")
    gone = m.forget("favourite colour")
    assert [g["text"] for g in gone] == ["my favourite colour is green"]
    assert [f["text"] for f in m.facts()] == ["my main project is Jarvis"]
    assert m.forget("something unrelated") == [] and m.forget("") == []


def test_relevance_search(tmp_path):
    m = mem(tmp_path)
    for t in ("my main project is Jarvis", "I prefer Spotify for music", "my dog is called Rex",
              "I live in Sofia", "I work as a developer", "my wife is Maria"):
        m.add(t)
    assert m.search("play some music")[0]["text"] == "I prefer Spotify for music"
    assert m.search("how is the jarvis project going")[0]["text"] == "my main project is Jarvis"
    assert len(m.search("")) == 6  # empty query: everything
    assert m.search("quantum physics") == []


def test_few_facts_all_returned(tmp_path):
    m = mem(tmp_path)
    m.add("I live in Sofia")
    m.add("my dog is called Rex")
    assert len(m.search("totally unrelated words")) == 2


def test_block_small_memory_includes_everything(tmp_path):
    m = mem(tmp_path)
    m.add("I live in Sofia")
    m.add("my dog is called Rex")
    b = m.block("hello")
    assert b.startswith("[Known about the user: ") and b.endswith("]\n")
    assert "Sofia" in b and "Rex" in b


def test_block_is_capped_and_keeps_core_facts(tmp_path):
    m = mem(tmp_path)
    m.add("call me Nikolay")
    for i in range(20):
        m.add(f"trivia{i} about widget{i} gadget{i}")
    m.add("I prefer Spotify for music")
    b = m.block("what is widget7 gadget7")
    assert b.count(";") <= 7  # at most 8 facts
    assert "call me Nikolay" in b and "I prefer Spotify" in b  # core facts always there
    assert "widget7" in b  # the relevant one too


def test_block_empty_when_nothing_stored(tmp_path):
    assert mem(tmp_path).block("anything") == ""


def test_name(tmp_path):
    m = mem(tmp_path)
    assert m.name() == ""
    m.add("call me nikolay")
    assert m.name() == "Nikolay"
    m.add("my name is Niki")
    assert m.name() == "Niki"


def test_second_person():
    assert second_person("my main project is Jarvis") == "Your main project is Jarvis"
    assert second_person("I prefer Spotify") == "You prefer Spotify"
    assert second_person("I'm a developer") == "You're a developer"
    assert second_person("call me Nikolay") == "You like to be called Nikolay"


def test_broken_file_is_ignored(tmp_path):
    p = tmp_path / "memory.json"
    p.write_text("{not json")
    assert Memory(p).facts() == []
    assert Memory(p).add("I live in Sofia")[1]


# ---- injection ------------------------------------------------------------------------------------------
class CaptureLLM:
    def __init__(self):
        self.calls = []

    async def chat(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        return LLMResponse(content="Hello there.")


def make_agent(block):
    bus = EventBus()
    llm = CaptureLLM()
    conf = Confirmer(bus, lambda t: asyncio.sleep(0), timeout=0.05)
    return Agent(llm, Registry(), bus, conf, memory_block=block), llm


def test_block_goes_on_newest_user_message_only_not_in_history_or_system():
    agent, llm = make_agent(lambda text: "[Known about the user: I prefer Spotify]\n")
    asyncio.run(agent.handle("play some music"))
    wire = llm.calls[0]
    assert wire[0]["role"] == "system" and "Known about the user: I prefer" not in wire[0]["content"]
    assert wire[-1]["content"].startswith("[Now: ")
    assert "[Known about the user: I prefer Spotify]\nplay some music" in wire[-1]["content"]
    assert agent.history("default")[0] == {"role": "user", "content": "play some music"}
    assert all("Known about" not in str(m["content"]) for m in agent.history("default"))
    asyncio.run(agent.handle("and again"))
    older = llm.calls[1][1]  # first user message of the second call: no block any more
    assert "Known about" not in older["content"] and "[Now" not in older["content"]


def test_system_prompt_static_and_mentions_memory_rules():
    assert "Known about" in SYSTEM_PROMPT and "remember" in SYSTEM_PROMPT
    assert "inferred" in SYSTEM_PROMPT and "secrets" in SYSTEM_PROMPT
    assert full_system_prompt() == full_system_prompt()


def test_memory_failure_never_breaks_a_turn():
    def boom(text):
        raise RuntimeError("disk gone")

    agent, llm = make_agent(boom)
    assert asyncio.run(agent.handle("hi")) == "Hello there."


def test_no_provider_no_block():
    agent, llm = make_agent(None)
    asyncio.run(agent.handle("hi"))
    assert "Known about" not in llm.calls[0][-1]["content"]


def test_default_memory_follows_appdata(tmp_path, monkeypatch):
    memory.default_memory().add("I live in Sofia")
    assert "Sofia" in memory.known_block("where")
    monkeypatch.setenv("APPDATA", str(tmp_path / "other"))
    assert memory.known_block("where") == ""


# ---- tools ----------------------------------------------------------------------------------------------
def call(name, **args):
    return asyncio.run(load_all().call(name, args))


def test_memory_tools_roundtrip():
    assert call("remember", fact="my main project is Jarvis") == "Noted"
    assert call("remember", fact="my main project is Jarvis") == "Updated that"
    assert call("remember", fact="I prefer Spotify", source="inferred") == "Noted"
    assert "Your main project is Jarvis" in call("recall")
    assert call("recall", query="spotify").startswith("You told me: You prefer Spotify")
    assert "forgotten that your main project is Jarvis" in call("forget", query="main project")
    assert "main project" not in call("recall")
    assert call("forget", query="main project").startswith("I don't remember")


def test_remember_secret_is_an_error():
    assert call("remember", fact="my password is hunter2").startswith("Error:")


def test_forget_everything_refused():
    call("remember", fact="I live in Sofia")
    assert call("forget", query="everything").startswith("Error:")
    assert "Sofia" in call("recall")


def test_recall_empty():
    assert call("recall") == "I don't remember anything yet"
