from datetime import datetime

import pytest

from jarvis.fastpath import match, normalize

NOW = datetime(2026, 10, 1, 18, 42)  # a Thursday


@pytest.mark.parametrize("text", [
    "what time is it", "What time is it?", "Jarvis, what time is it please", "hey jarvis what's the time",
    "what is the time now", "can you tell me the time", "whats the current time", "the time",
])
def test_time(text):
    fp = match(text, NOW)
    assert fp and fp.kind == "time" and fp.reply == "It's 6:42 PM, sir."


def test_time_morning_no_leading_zero():
    assert match("what time is it", datetime(2026, 1, 2, 9, 5)).reply == "It's 9:05 AM, sir."


def test_date_and_day():
    assert match("what's the date today", NOW).reply == "Today is Thursday, 1 October 2026, sir."
    assert match("what is today's date", NOW).kind == "date"
    assert match("what day is it", NOW).reply == "It's Thursday, sir."
    assert match("what day is it today?", NOW).kind == "day"


def test_open_known_app():
    fp = match("Jarvis, open Chrome please.", NOW)
    assert fp and fp.kind == "open_app" and fp.action == ("open_app", "chrome")
    assert match("open the visual studio code app", NOW).action == ("open_app", "visual studio code")
    assert match("could you launch spotify", NOW).action == ("open_app", "spotify")
    assert match("open discord", NOW).action == ("open_app", "discord")


def test_lock():
    for t in ("lock the pc", "lock my computer", "Jarvis lock the computer please", "lock pc"):
        assert match(t, NOW).action == ("lock_pc",)


def test_stop():
    for t in ("stop", "Stop.", "never mind", "nevermind", "cancel", "jarvis stop talking", "forget it"):
        fp = match(t, NOW)
        assert fp and fp.stop_speaking and fp.reply == ""


def test_volume():
    assert match("volume up", NOW).action == ("volume", "up")
    assert match("turn the volume down please", NOW).action == ("volume", "down")
    assert match("mute", NOW).action == ("volume", "mute")
    assert match("raise the volume", NOW).action == ("volume", "up")


@pytest.mark.parametrize("text", [
    "", "   ", "jarvis", "hello",
    "what time is it in tokyo", "what time is the meeting", "what time does the shop close",
    "remind me what time it is at five", "the time machine",
    "open chrome and search for cats", "open the pod bay doors", "open my downloads folder",
    "open", "open chrome tabs", "start a timer for five minutes",
    "lock the door", "lock the pc in ten minutes", "unlock the pc",
    "stop the music", "stop the timer", "cancel my subscription", "stop it from raining",
    "what day is my birthday", "what is the date of the next holiday",
    "turn the volume up to eighty", "volume", "mute the microphone", "mute john",
    "can you help me with the time",
])
def test_negative_goes_to_llm(text):
    assert match(text, NOW) is None


def test_normalize():
    assert normalize("  Hey Jarvis,  What's the TIME?! Please, sir. ") == "what's the time"
    assert normalize("It’s fine") == "it's fine"
