import asyncio
from datetime import datetime
from types import SimpleNamespace

from jarvis import briefing as brief
from jarvis import reminders as rem
from jarvis import store
from jarvis.config import Config
from jarvis.extras import Extras

NOW = datetime(2026, 10, 2, 7, 30)


class FakeAssistant:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.voice = False
        self.said = []
        self.agent = SimpleNamespace(memory_block=None, exchanges=[],
                                     add_exchange=lambda u, r: self.agent.exchanges.append((u, r)))
        self.turn_lock = asyncio.Lock()

    async def say_safe(self, text):
        self.said.append(text)

    async def announce(self, text):
        self.said.append(("announce", text))


def fake_now(monkeypatch):
    monkeypatch.setattr("jarvis.extras.datetime", type("D", (), {"now": staticmethod(lambda: NOW)}))


def setup(monkeypatch, cfg=None):
    fake_now(monkeypatch)

    async def fake_build(cfg, now=None, held=None, **kw):
        return "Good morning." + (" HELD" if held else "")

    monkeypatch.setattr(brief, "build", fake_build)
    a = FakeAssistant(cfg)
    return a, Extras(a)


def req(kind="activate", source="wake"):
    return SimpleNamespace(kind=kind, source=source)


def test_memory_wired_into_agent(monkeypatch):
    a, _ = setup(monkeypatch)
    assert a.agent.memory_block is not None
    cfg = Config()
    cfg.memory.enabled = False
    a2, _ = setup(monkeypatch, cfg)
    assert a2.agent.memory_block is None


def test_first_wake_speaks_briefing_once_per_day(monkeypatch):
    a, ex = setup(monkeypatch)
    asyncio.run(ex.first_wake_briefing(req(), "what time is it"))
    assert a.said == ["Good morning."]
    assert a.agent.exchanges and a.agent.exchanges[0][1] == "Good morning."
    assert store.load_state()["last_briefing"] == "2026-10-02"
    asyncio.run(ex.first_wake_briefing(req(), "what time is it"))  # same day: nothing more
    assert a.said == ["Good morning."]


def test_not_for_text_or_barge_or_disabled(monkeypatch):
    a, ex = setup(monkeypatch)
    asyncio.run(ex.first_wake_briefing(req("text", "hotkey"), "hi"))
    asyncio.run(ex.first_wake_briefing(req("activate", "barge"), "hi"))
    assert a.said == []
    cfg = Config()
    cfg.briefing.auto_first_wake = False
    a2, ex2 = setup(monkeypatch, cfg)
    asyncio.run(ex2.first_wake_briefing(req(), "hi"))
    assert a2.said == []
    asyncio.run(ex.first_wake_briefing(req("activate", "hotkey"), "hi"))
    assert a.said == ["Good morning."]  # hotkey counts


def test_asking_for_the_briefing_skips_the_automatic_one(monkeypatch):
    a, ex = setup(monkeypatch)
    asyncio.run(ex.first_wake_briefing(req(), "give me my briefing"))
    assert a.said == [] and "last_briefing" not in store.load_state()


def test_failed_briefing_does_not_block_and_keeps_held(monkeypatch):
    a, ex = setup(monkeypatch)

    async def broken(*args, **kw):
        raise RuntimeError("x")

    monkeypatch.setattr(brief, "build", broken)

    async def announce(text):
        pass

    sched = rem.Scheduler(rem.default_store(), announce, hold_missed=lambda: True)
    sched.held = [{"kind": "reminder", "text": "x", "label": "", "seconds": 0, "due": 0}]
    rem.set_active(sched)
    try:
        asyncio.run(ex.first_wake_briefing(req(), "hi"))
        assert a.said == [] and len(sched.held) == 1
        assert "last_briefing" not in store.load_state()
    finally:
        rem.set_active(None)


def test_missed_reminders_go_into_briefing(monkeypatch):
    a, ex = setup(monkeypatch)

    async def announce(text):
        a.said.append(("late", text))

    sched = rem.Scheduler(rem.default_store(), announce, hold_missed=ex.briefing_pending)
    rem.default_store().add_reminder("call mom", datetime(2026, 10, 2, 3, 0), datetime(2026, 10, 1, 20, 0))

    async def go():
        await sched.catch_up()  # at start (07:30, briefing still due): held
        await sched.drain()

    monkeypatch.setattr(sched, "clock", lambda: NOW)
    asyncio.run(go())
    assert a.said == [] and len(sched.held) == 1
    rem.set_active(sched)
    try:
        asyncio.run(ex.first_wake_briefing(req(), "hi"))
    finally:
        rem.set_active(None)
    assert a.said == ["Good morning. HELD"]


def test_start_runs_scheduler(monkeypatch):
    a, ex = setup(monkeypatch)
    rem.default_store().add_timer(0.01, "", datetime.now())

    async def go():
        bg = []
        ex.start(bg)
        assert len(bg) == 1
        await asyncio.sleep(1.3)
        bg[0].cancel()
        await asyncio.gather(bg[0], return_exceptions=True)
        await ex.scheduler.drain()

    asyncio.run(go())
    rem.set_active(None)
    assert ("announce", "Your 0 second timer is done.") in a.said or any(
        isinstance(s, tuple) and "timer is done" in s[1] for s in a.said)


def test_start_disabled(monkeypatch):
    cfg = Config()
    cfg.reminders.enabled = False
    a, ex = setup(monkeypatch, cfg)
    bg = []
    ex.start(bg)
    assert bg == [] and ex.scheduler is None
