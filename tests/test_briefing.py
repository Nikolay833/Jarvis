import asyncio
import json
from datetime import datetime, timedelta

import pytest

from jarvis import briefing as brief
from jarvis import claude_sessions as cs
from jarvis import reminders as rem
from jarvis import store
from jarvis.config import BriefingConfig, Config
from jarvis.memory import default_memory

NOW = datetime(2026, 10, 2, 7, 30)  # Friday morning
CFG = BriefingConfig()
WEATHER = {"temp": 14.4, "code": 2, "hi": 19.2, "lo": 8.1, "rain": 10}


def session(title, project, when, kind="assistant_text", sid="s1"):
    return cs.Session(id=sid, path="/x", project_dir=f"/p/{project}", project_name=project, title=title,
                      last_activity=when.timestamp(), last_kind=kind)


@pytest.mark.parametrize("code,word", [
    (0, "clear"), (1, "mostly clear"), (2, "partly cloudy"), (3, "overcast"), (45, "foggy"), (48, "foggy"),
    (51, "drizzly"), (61, "lightly raining"), (63, "raining"), (65, "raining heavily"), (71, "lightly snowing"),
    (75, "snowing heavily"), (80, "showery"), (95, "stormy"), (96, "stormy with hail"), (99, "stormy with hail"),
    (1234, ""), (None, ""), ("x", ""),
])
def test_wmo_mapping(code, word):
    assert brief.weather_words(code) == word


def test_weather_sentence():
    s = brief.weather_sentence("Sofia", WEATHER)
    assert s == "In Sofia it's 14 degrees and partly cloudy right now, with a high of 19 and a low of 8."
    s = brief.weather_sentence("Sofia", {**WEATHER, "temp": -3.2, "lo": -6, "rain": 60, "code": 73})
    assert "minus 3 degrees and snowing" in s and "low of minus 6" in s and "60 percent chance of rain" in s
    assert "unknown" not in brief.weather_sentence("X", {**WEATHER, "code": 999})


def test_parse_weather():
    data = {"current": {"temperature_2m": 12.5, "weather_code": 3},
            "daily": {"temperature_2m_max": [15.0], "temperature_2m_min": [7.0], "precipitation_probability_max": [40]}}
    assert brief.parse_weather(data) == {"temp": 12.5, "code": 3, "hi": 15.0, "lo": 7.0, "rain": 40}
    assert brief.parse_weather({"current": {}}) is None and brief.parse_weather({}) is None


def test_fetch_weather_failure_is_silent(monkeypatch):
    import httpx

    class Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("offline")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", Boom)
    assert asyncio.run(brief.fetch_weather(CFG)) is None


def test_fetch_weather_url_and_parse(monkeypatch):
    import httpx

    seen = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"current": {"temperature_2m": 10, "weather_code": 0},
                    "daily": {"temperature_2m_max": [12], "temperature_2m_min": [5]}}

    class Client:
        def __init__(self, *a, **k):
            seen["timeout"] = k.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            seen["url"], seen["params"] = url, params
            return Resp()

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    w = asyncio.run(brief.fetch_weather(CFG))
    assert w["hi"] == 12.0 and w["rain"] is None
    assert seen["url"] == "https://api.open-meteo.com/v1/forecast" and seen["timeout"] == 3.0
    assert seen["params"]["latitude"] == 42.6977 and seen["params"]["longitude"] == 23.3219
    assert seen["params"]["timezone"] == "auto" and "weather_code" in seen["params"]["current"]


def test_greeting_and_date():
    assert brief.greeting(NOW, "Nikolay") == "Good morning, Nikolay."
    assert brief.greeting(datetime(2026, 1, 1, 14), "") == "Good afternoon."
    assert brief.greeting(datetime(2026, 1, 1, 20), "X") == "Good evening, X."
    assert brief.date_sentence(NOW) == "It's Friday, the 2nd of October."
    assert brief.date_sentence(datetime(2026, 10, 11)).endswith("the 11th of October.")
    assert brief.date_sentence(datetime(2026, 10, 23)).endswith("the 23rd of October.")


def reminder_items():
    s = rem.ReminderStore()
    s.add_reminder("call mom", datetime(2026, 10, 2, 18, 0), NOW)
    s.add_reminder("pay rent", datetime(2026, 10, 2, 9, 0), NOW)
    s.add_reminder("tomorrow thing", datetime(2026, 10, 3, 9, 0), NOW)
    s.add_timer(600, "", NOW)
    return s.pending()


def test_compose_everything():
    sessions = [session("settings page", "Jarvis", datetime(2026, 10, 1, 23, 10))]
    held = [{"kind": "reminder", "text": "take pills", "label": "", "seconds": 0, "due": 0}]
    text = brief.compose(NOW, CFG, name="Nikolay", weather=WEATHER, reminders=reminder_items(), sessions=sessions,
                         missed=held)
    assert text.startswith("Good morning, Nikolay. It's Friday, the 2nd of October. In Sofia it's 14 degrees")
    assert "While you were away: reminder, take pills." in text
    assert "You have 2 reminders today: pay rent at 9 AM and call mom at 6 PM." in text
    assert "tomorrow thing" not in text and "timer" not in text
    assert "Claude finished work on settings page in Jarvis last night." in text


def test_compose_minimal_no_data():
    assert brief.compose(NOW, CFG) == "Good morning. It's Friday, the 2nd of October."


def test_compose_respects_flags():
    cfg = BriefingConfig(include_weather=False, include_reminders=False, include_claude=False)
    sessions = [session("x", "Jarvis", NOW - timedelta(hours=3))]
    text = brief.compose(NOW, cfg, weather=WEATHER, reminders=reminder_items(), sessions=sessions)
    assert text == "Good morning. It's Friday, the 2nd of October."


def test_single_reminder_wording():
    s = rem.ReminderStore()
    s.add_reminder("call mom", datetime(2026, 10, 2, 18, 0), NOW)
    assert brief.reminders_sentence(s.pending(), NOW) == "You have one reminder today: call mom at 6 PM."


def test_claude_sentences():
    sessions = [
        session("old", "A", NOW - timedelta(days=3), sid="a"),                      # too old
        session("login bug", "Jarvis", datetime(2026, 10, 1, 17, 0), sid="b"),      # yesterday daytime
        session("night job", "Site", datetime(2026, 10, 2, 2, 0), sid="c"),         # overnight
        session("third", "Three", datetime(2026, 10, 2, 6, 0), sid="d"),            # earlier today
        session("interrupted", "X", NOW - timedelta(hours=2), kind="tool_use", sid="e"),  # not finished
        session("busy", "Busy", NOW - timedelta(seconds=5), kind="tool_use", sid="f"),    # still working
    ]
    lines = brief.claude_sentences(sessions, NOW)
    assert "Claude is still working in Busy." in lines
    finished = [l for l in lines if "finished" in l]
    assert finished == ["Claude finished work on third in Three earlier today.",
                        "Claude finished work on night job in Site overnight."]
    assert lines[-1] == "1 other Claude session had activity too."
    assert not any("old" in l or "interrupted" in l for l in lines)


def test_claude_waiting_for_permission():
    class P:
        project = "Jarvis"

    assert brief.claude_sentences([], NOW, {"s": P()}) == ["Claude is waiting for your permission in Jarvis."]


def test_build_with_mocks_and_memory_name():
    default_memory().add("my name is Nikolay")
    rem.default_store().add_reminder("call mom", datetime(2026, 10, 2, 18, 0), NOW)
    calls = {}

    async def fetch(cfg):
        calls["weather"] = cfg.city
        return WEATHER

    sessions = [session("settings page", "Jarvis", datetime(2026, 10, 1, 23, 10))]
    cfg = Config()
    text = asyncio.run(brief.build(cfg, NOW, fetch=fetch, sessions_fn=lambda: sessions, pending={}))
    assert calls["weather"] == "Sofia"
    assert text.startswith("Good morning, Nikolay.")
    assert "partly cloudy" in text and "call mom at 6 PM" in text and "last night" in text


def test_build_survives_failures():
    async def fetch(cfg):
        raise RuntimeError("offline")

    def scan():
        raise OSError("no disk")

    text = asyncio.run(brief.build(Config(), NOW, fetch=fetch, sessions_fn=scan, pending={}))
    assert text == "Good morning. It's Friday, the 2nd of October."


def test_build_skips_disabled_sources():
    cfg = Config()
    cfg.briefing.include_weather = False
    cfg.briefing.include_claude = False

    async def fetch(c):
        raise AssertionError("must not be called")

    text = asyncio.run(brief.build(cfg, NOW, fetch=fetch, sessions_fn=lambda: 1 / 0, pending={}))
    assert text.startswith("Good morning.")


def test_briefing_tool_marks_day(monkeypatch):
    from jarvis.tools import load_all
    from jarvis.tools.context import set_context

    async def fake_build(cfg, now=None, **kw):
        return "Good morning. Test."

    monkeypatch.setattr(brief, "build", fake_build)
    set_context(Config())
    monkeypatch.setattr("jarvis.tools.briefing_tool.datetime", type("D", (), {"now": staticmethod(lambda: NOW)}))
    assert asyncio.run(load_all().call("briefing", {})) == "Good morning. Test."
    assert store.load_state()["last_briefing"] == "2026-10-02"
    cfg = Config()
    cfg.briefing.enabled = False
    set_context(cfg)
    assert asyncio.run(load_all().call("briefing", {})).startswith("Error:")
    set_context(Config())


# ---- first wake of the day ------------------------------------------------------------------------------
def test_first_wake_logic_with_state_file():
    cfg = BriefingConfig()
    assert brief.due_today(cfg, NOW)                                   # no state: due
    assert not brief.due_today(cfg, datetime(2026, 10, 2, 4, 59))      # before 05:00
    brief.mark_done(cfg, NOW)
    assert json.loads(store.state_path().read_text())["last_briefing"] == "2026-10-02"
    assert not brief.due_today(cfg, NOW + timedelta(hours=5))          # same day, even after a restart
    assert brief.due_today(cfg, datetime(2026, 10, 3, 5, 0))           # next day at 05:00
    assert not brief.due_today(cfg, datetime(2026, 10, 3, 4, 0))


def test_early_wake_does_not_use_up_the_day():
    cfg = BriefingConfig()
    brief.mark_done(cfg, datetime(2026, 10, 3, 1, 0))  # a 1 a.m. on-demand briefing does not count
    assert brief.due_today(cfg, datetime(2026, 10, 3, 7, 0))


def test_due_today_flags_and_explicit_state():
    assert not brief.due_today(BriefingConfig(enabled=False), NOW)
    assert not brief.due_today(BriefingConfig(auto_first_wake=False), NOW)
    assert not brief.due_today(BriefingConfig(), NOW, {"last_briefing": "2026-10-02"})
    assert brief.due_today(BriefingConfig(after_hour=8), datetime(2026, 10, 2, 8, 0), {})
    assert not brief.due_today(BriefingConfig(after_hour=8), NOW, {})


def test_state_survives_other_keys():
    store.update_state(other=1)
    brief.mark_done(BriefingConfig(), NOW)
    st = store.load_state()
    assert st["other"] == 1 and st["last_briefing"] == "2026-10-02"
