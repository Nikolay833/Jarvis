from datetime import datetime, timedelta

import pytest

from jarvis.timeparse import format_when, is_time_phrase, parse_duration, parse_when, spoken_duration

NOW = datetime(2026, 10, 2, 18, 42)  # a Friday


@pytest.mark.parametrize("text,secs", [
    ("10 minutes", 600), ("ten minutes", 600), ("1 minute", 60), ("a minute", 60), ("an hour", 3600),
    ("half an hour", 1800), ("a quarter of an hour", 900), ("an hour and a half", 5400),
    ("1 hour 30 minutes", 5400), ("2 hours and 15 minutes", 8100), ("90 seconds", 90), ("for 45 secs", 45),
    ("1.5 hours", 5400), ("twenty five minutes", 1500), ("thirty seconds", 30), ("in 5 mins", 300),
    ("2 days", 172800), ("one hour", 3600), ("2 and a half hours", 9000),
])
def test_parse_duration(text, secs):
    assert parse_duration(text) == secs


@pytest.mark.parametrize("text", ["", "6pm", "the oven", "5 minutes of silence", "0 minutes", "minutes", "soon"])
def test_parse_duration_none(text):
    assert parse_duration(text) is None


@pytest.mark.parametrize("text,delta", [
    ("in 20 minutes", timedelta(minutes=20)), ("in an hour", timedelta(hours=1)),
    ("in half an hour", timedelta(minutes=30)), ("in 2 hours and 30 minutes", timedelta(hours=2, minutes=30)),
    ("in 90 seconds", timedelta(seconds=90)), ("in twenty minutes", timedelta(minutes=20)),
    ("10 minutes from now", timedelta(minutes=10)), ("in a minute", timedelta(minutes=1)),
    ("in 2 days", timedelta(days=2)), ("after 5 minutes", timedelta(minutes=5)),
])
def test_relative(text, delta):
    assert parse_when(text, NOW) == NOW + delta


@pytest.mark.parametrize("text,expected", [
    ("at 6pm", datetime(2026, 10, 3, 18, 0)),            # 6 PM already passed today
    ("at 11pm", datetime(2026, 10, 2, 23, 0)),
    ("at 6:30 pm", datetime(2026, 10, 3, 18, 30)),
    ("at 6.30pm", datetime(2026, 10, 3, 18, 30)),
    ("at 18:30", datetime(2026, 10, 3, 18, 30)),
    ("at 19:00", datetime(2026, 10, 2, 19, 0)),
    ("at 7am", datetime(2026, 10, 3, 7, 0)),
    ("at 9", datetime(2026, 10, 2, 21, 0)),               # bare hour: next time it happens
    ("at 6", datetime(2026, 10, 3, 6, 0)),
    ("at 6 o'clock", datetime(2026, 10, 3, 6, 0)),
    ("at noon", datetime(2026, 10, 3, 12, 0)),
    ("at midnight", datetime(2026, 10, 3, 0, 0)),
    ("half past 7", datetime(2026, 10, 2, 19, 30)),
    ("quarter to 8", datetime(2026, 10, 2, 19, 45)),
    ("tomorrow", datetime(2026, 10, 3, 9, 0)),
    ("tomorrow at 9", datetime(2026, 10, 3, 9, 0)),
    ("tomorrow at 3", datetime(2026, 10, 3, 15, 0)),
    ("tomorrow at 7:15 am", datetime(2026, 10, 3, 7, 15)),
    ("tomorrow morning", datetime(2026, 10, 3, 8, 0)),
    ("tomorrow evening", datetime(2026, 10, 3, 18, 0)),
    ("tomorrow at 8 in the morning", datetime(2026, 10, 3, 8, 0)),
    ("at 6pm tomorrow", datetime(2026, 10, 3, 18, 0)),
    ("day after tomorrow at 8am", datetime(2026, 10, 4, 8, 0)),
    ("tonight at 9", datetime(2026, 10, 2, 21, 0)),
    ("today at 11pm", datetime(2026, 10, 2, 23, 0)),
    ("monday", datetime(2026, 10, 5, 9, 0)),
    ("on monday at 3pm", datetime(2026, 10, 5, 15, 0)),
    ("next tuesday at noon", datetime(2026, 10, 6, 12, 0)),
    ("saturday at 18:30", datetime(2026, 10, 3, 18, 30)),
    ("friday at 6pm", datetime(2026, 10, 9, 18, 0)),     # it is Friday 18:42: next week
    ("friday at 9pm", datetime(2026, 10, 2, 21, 0)),     # still ahead today
    ("friday", datetime(2026, 10, 9, 9, 0)),
])
def test_clock_and_days(text, expected):
    assert parse_when(text, NOW) == expected


@pytest.mark.parametrize("text", ["", "blah", "at 25", "at 13pm", "at 6:75", "soon", "later", "yesterday at 3"])
def test_unparseable(text):
    assert parse_when(text, NOW) is None


def test_result_always_in_the_future():
    for text in ("at 6:42pm", "at 18:42", "today at 6pm", "at 6:41 pm"):
        r = parse_when(text, NOW)
        assert r is None or r > NOW


def test_early_morning_context():
    now = datetime(2026, 10, 2, 7, 0)
    assert parse_when("at 9", now) == datetime(2026, 10, 2, 9, 0)
    assert parse_when("at 6pm", now) == datetime(2026, 10, 2, 18, 0)
    assert parse_when("at 6am", now) == datetime(2026, 10, 3, 6, 0)


def test_strict_requires_pure_time_phrase():
    assert parse_when("tomorrow at 9", NOW, strict=True)
    assert parse_when("on a walk tomorrow at 9", NOW, strict=True) is None
    assert parse_when("at the office at 6pm", NOW, strict=True) is None
    assert is_time_phrase("in 20 minutes") and is_time_phrase("friday at 6:30 pm") and not is_time_phrase("call mom")


def test_spoken_helpers():
    assert spoken_duration(600) == "10 minutes" and spoken_duration(60) == "1 minute"
    assert spoken_duration(5400) == "1 hour 30 minutes" and spoken_duration(90) == "1 minute 30 seconds"
    assert format_when(NOW + timedelta(minutes=20), NOW) == "in 20 minutes"
    assert format_when(datetime(2026, 10, 2, 21, 0), NOW) == "at 9 PM today"
    assert format_when(datetime(2026, 10, 3, 9, 30), NOW) == "tomorrow at 9:30 AM"
    assert format_when(datetime(2026, 10, 5, 15, 0), NOW) == "Monday at 3 PM"
    assert format_when(datetime(2026, 10, 20, 15, 0), NOW) == "on Tuesday 20 October at 3 PM"
