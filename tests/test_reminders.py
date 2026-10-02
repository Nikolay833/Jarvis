import asyncio
from datetime import datetime, timedelta

import pytest

from jarvis import reminders as rem
from jarvis.reminders import Ambiguous, ReminderStore, Scheduler
from jarvis.tools import load_all

T0 = datetime(2026, 10, 2, 12, 0, 0)


class FakeClock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


def make(tmp_path, clock=None):
    clock = clock or FakeClock()
    said = []

    async def announce(text):
        said.append(text)

    store = ReminderStore(tmp_path / "reminders.json")
    return store, Scheduler(store, announce, clock=clock, hold_missed=None), clock, said


def run(coro):
    return asyncio.run(coro)


def test_store_persists(tmp_path):
    store, *_ = make(tmp_path)
    store.add_timer(600, "", T0)
    store.add_reminder("call mom", T0 + timedelta(hours=6), T0)
    again = ReminderStore(tmp_path / "reminders.json")
    assert [i["kind"] for i in again.pending()] == ["timer", "reminder"]
    assert again.pending()[1]["text"] == "call mom"


def test_store_sorted_by_due(tmp_path):
    store, *_ = make(tmp_path)
    store.add_reminder("later", T0 + timedelta(hours=2), T0)
    store.add_reminder("sooner", T0 + timedelta(hours=1), T0)
    assert [i["text"] for i in store.pending()] == ["sooner", "later"]


def test_broken_file(tmp_path):
    (tmp_path / "reminders.json").write_text("garbage")
    assert ReminderStore(tmp_path / "reminders.json").pending() == []


def test_scheduler_fires_due_only_and_once(tmp_path):
    store, sched, clock, said = make(tmp_path)
    store.add_reminder("call mom", T0 + timedelta(minutes=5), T0)
    store.add_timer(600, "", T0)

    async def go():
        await sched.tick()
        await sched.drain()
        assert said == []
        clock.advance(minutes=5)
        await sched.tick()
        await sched.drain()
        assert said == ["Sir, reminder: call mom."]
        await sched.tick()  # not again
        await sched.drain()
        assert len(said) == 1
        clock.advance(minutes=5)
        await sched.tick()
        await sched.drain()

    run(go())
    assert said[1] == "Your 10 minute timer is done."
    assert store.pending() == [] and ReminderStore(tmp_path / "reminders.json").pending() == []


def test_labelled_timer_text(tmp_path):
    store, sched, clock, said = make(tmp_path)
    store.add_timer(90, "pasta", T0)
    clock.advance(seconds=90)
    run(sched.tick())
    run(sched.drain())
    assert said == ["Sir, your pasta timer is done."]


def test_adjective_forms():
    assert rem._adjective(90) == "1 minute 30 second"
    assert rem._adjective(3600) == "1 hour"
    assert rem._adjective(7200) == "2 hour"


def test_missed_announced_once_at_start(tmp_path):
    store, sched, clock, said = make(tmp_path)
    store.add_reminder("call mom", T0 + timedelta(minutes=5), T0)
    store.add_timer(60, "", T0)
    store.add_reminder("far future", T0 + timedelta(days=1), T0)
    clock.advance(hours=3)  # PC was off

    async def go():
        await sched.catch_up()
        await sched.drain()
        await sched.tick()
        await sched.drain()

    run(go())
    assert len(said) == 1
    assert said[0].startswith("Sir, while you were away:")
    assert "reminder, call mom" in said[0] and "1 minute timer finished" in said[0]
    assert [i["text"] for i in store.pending()] == ["far future"]


def test_missed_held_for_briefing(tmp_path):
    store, _, clock, said = make(tmp_path)

    async def announce(text):
        said.append(text)

    sched = Scheduler(store, announce, clock=clock, hold_missed=lambda: True)
    store.add_reminder("call mom", T0 + timedelta(minutes=5), T0)
    clock.advance(hours=3)
    run(sched.catch_up())
    run(sched.drain())
    assert said == [] and len(sched.held) == 1
    rem.set_active(sched)
    try:
        assert [i["text"] for i in rem.take_held()] == ["call mom"]
        assert rem.take_held() == []
    finally:
        rem.set_active(None)


def test_run_loop_survives_bad_tick(tmp_path):
    store, sched, clock, said = make(tmp_path)
    calls = {"n": 0}
    real = sched.tick

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        await real()

    sched.tick = flaky
    sched.interval = 0.01
    store.add_reminder("x", T0 - timedelta(seconds=1) + timedelta(seconds=1), T0)

    async def go():
        task = asyncio.create_task(sched.run())
        await asyncio.sleep(0.1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    run(go())
    assert calls["n"] >= 2


def test_announce_failure_is_contained(tmp_path):
    store, _, clock, _ = make(tmp_path)

    async def bad(text):
        raise RuntimeError("speaker gone")

    sched = Scheduler(store, bad, clock=clock)
    store.add_reminder("x", T0, T0)
    run(sched.tick())
    run(sched.drain())  # no exception


# ---- cancel ---------------------------------------------------------------------------------------------
def test_cancel_variants(tmp_path):
    store, *_ = make(tmp_path)
    store.add_timer(600, "", T0)
    store.add_reminder("call mom", T0 + timedelta(hours=1), T0)
    store.add_reminder("pay rent", T0 + timedelta(hours=2), T0)
    assert store.cancel("timer")[0]["kind"] == "timer"
    assert store.cancel("the reminder about call mom")[0]["text"] == "call mom"
    assert store.cancel("something else entirely") == []
    assert [i["text"] for i in store.pending()] == ["pay rent"]
    assert len(store.cancel("")) == 1 and store.pending() == []


def test_cancel_ambiguous_and_all(tmp_path):
    store, *_ = make(tmp_path)
    store.add_timer(600, "", T0)
    store.add_timer(300, "", T0)
    with pytest.raises(Ambiguous):
        store.cancel("timer")
    assert [i["seconds"] for i in store.cancel("the 5 minute timer")] == [300]
    store.add_reminder("x", T0 + timedelta(hours=1), T0)
    assert len(store.cancel("all timers")) == 1 and len(store.pending()) == 1
    assert len(store.cancel("all")) == 1


# ---- tools ----------------------------------------------------------------------------------------------
def call(name, **args):
    return asyncio.run(load_all().call(name, args))


def test_tools_registered():
    names = load_all().names()
    for n in ("set_timer", "set_reminder", "list_reminders", "cancel_reminder", "remember", "recall", "forget",
              "briefing"):
        assert n in names
        assert load_all().get(n).risk == "safe"
    params = load_all().get("set_reminder").parameters
    assert params["required"] == ["text", "when"]
    assert load_all().get("set_timer").parameters["required"] == []


def test_set_timer_and_list_and_cancel():
    assert call("set_timer", minutes=10) == "Timer set for 10 minutes"
    assert call("set_timer", seconds=30, label="tea") == "Tea timer set for 30 seconds"
    assert call("set_timer").startswith("Error:")
    out = call("list_reminders")
    assert out.startswith("You have 2:") and "10 minute timer" in out and "tea timer" in out
    assert call("cancel_reminder", query="tea").startswith("Cancelled the tea timer")
    assert call("cancel_reminder", query="timer").startswith("Cancelled the 10 minute timer")
    assert call("list_reminders") == "You have no reminders or timers"
    assert call("cancel_reminder", query="timer") == "I found no matching reminder or timer"


def test_set_reminder_tool():
    out = call("set_reminder", text="call mom", when="in 20 minutes")
    assert out == "Reminder set to call mom in 20 minutes"
    assert call("set_reminder", text="x", when="whenever").startswith("Error: I couldn't understand")
    assert call("set_reminder", text=" ", when="in 5 minutes").startswith("Error:")
    assert "call mom" in call("list_reminders")


def test_reminders_persist_across_store_instances():
    call("set_reminder", text="call mom", when="tomorrow at 9")
    rem._stores.clear()  # a fresh process
    assert "call mom" in call("list_reminders")
