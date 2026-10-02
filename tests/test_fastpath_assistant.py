"""Fast paths for memory, timers, reminders and the briefing."""
import json
from datetime import datetime

import pytest

from jarvis.fastpath import match

NOW = datetime(2026, 10, 2, 18, 42)


def call_of(text):
    fp = match(text, NOW)
    assert fp is not None, text
    assert fp.action[0] == "call" and fp.speak_result
    return fp.action[1], json.loads(fp.action[2])


@pytest.mark.parametrize("text,fact", [
    ("remember that my main project is Jarvis", "my main project is Jarvis"),
    ("Remember my name is Nikolay.", "my name is Nikolay"),
    ("Jarvis, remember I prefer Spotify", "I prefer Spotify"),
    ("please remember that I like my tea strong, please", "I like my tea strong"),
    ("could you remember my dog is called Rex", "my dog is called Rex"),
])
def test_remember(text, fact):
    assert call_of(text) == ("remember", {"fact": fact})


@pytest.mark.parametrize("text", [
    "remember to buy milk", "remember what I said", "do you remember my name", "I remember that day",
    "remember", "remember when we met", "remember me",
])
def test_remember_negatives(text):
    fp = match(text, NOW)
    assert fp is None or fp.kind != "remember"


@pytest.mark.parametrize("text,query", [
    ("what do you remember", ""), ("What do you remember about me?", ""), ("what do you know about me", ""),
    ("what have I told you", ""), ("tell me what you remember", ""), ("what do you remember about my project", "my project"),
])
def test_recall(text, query):
    assert call_of(text) == ("recall", {"query": query})


@pytest.mark.parametrize("text", ["what do you know about quantum physics", "what do you know", "do you remember"])
def test_recall_negatives(text):
    assert match(text, NOW) is None


@pytest.mark.parametrize("text,query", [
    ("forget my favourite colour", "my favourite colour"), ("forget that my dog is called Rex", "my dog is called rex"),
    ("forget about the project", "the project"), ("Jarvis forget my name please", "my name"),
])
def test_forget(text, query):
    assert call_of(text) == ("forget", {"query": query})


@pytest.mark.parametrize("text", ["forget that", "forget about it", "forget everything", "forget"])
def test_forget_negatives(text):
    fp = match(text, NOW)
    assert fp is None or fp.kind != "forget"


def test_forget_it_is_still_stop():
    assert match("forget it", NOW).stop_speaking


@pytest.mark.parametrize("text,minutes,seconds", [
    ("set a timer for 10 minutes", 10, 0), ("Set a timer for ten minutes please", 10, 0),
    ("timer 5 minutes", 5, 0), ("timer for 5 minutes", 5, 0), ("set a 10 minute timer", 10, 0),
    ("start a timer for 90 seconds", 1, 30), ("set a timer for an hour", 60, 0),
    ("set a timer for 1 hour 30 minutes", 90, 0), ("can you set a timer for half an hour", 30, 0),
    ("set a timer for 1.5 minutes", 1, 30), ("Jarvis, set a 2 minute timer", 2, 0), ("set timer for 20 seconds", 0, 20),
])
def test_timer(text, minutes, seconds):
    assert call_of(text) == ("set_timer", {"minutes": minutes, "seconds": seconds})


@pytest.mark.parametrize("text", [
    "set a timer", "set a timer for the oven", "timer", "what is a timer", "timer please", "set an alarm for 6",
])
def test_timer_negatives(text):
    assert match(text, NOW) is None


@pytest.mark.parametrize("text,reminder,when", [
    ("remind me in 20 minutes to call mom", "call mom", "in 20 minutes"),
    ("remind me to call mom at 6pm", "call mom", "at 6pm"),
    ("remind me to call mom at 6:30 pm", "call mom", "at 6:30 pm"),
    ("remind me at 18:30 to take my pills", "take my pills", "at 18:30"),
    ("remind me to go on a walk tomorrow at 9", "go on a walk", "tomorrow at 9"),
    ("remind me to pay rent on friday at 3pm", "pay rent", "on friday at 3pm"),
    ("remind me tomorrow to call bob", "call bob", "tomorrow"),
    ("remind me to stretch in half an hour", "stretch", "in half an hour"),
    ("remind me to call mom at the office at 6pm", "call mom at the office", "at 6pm"),
    ("set a reminder for 6pm to call dad", "call dad", "6pm"),
    ("set a reminder to call dad at 6pm", "call dad", "at 6pm"),
    ("Jarvis remind me that the oven is on in 10 minutes", "the oven is on", "in 10 minutes"),
])
def test_reminder(text, reminder, when):
    assert call_of(text) == ("set_reminder", {"text": reminder, "when": when})


@pytest.mark.parametrize("text", [
    "remind me to buy milk", "remind me", "remind me what time it is at five", "remind me later",
    "set a reminder", "what is a reminder", "remind me to call mom whenever",
])
def test_reminder_negatives(text):
    assert match(text, NOW) is None


@pytest.mark.parametrize("text", [
    "what reminders do I have", "what timers do i have", "list my reminders", "show me my timers",
    "do I have any reminders", "how much time is left", "how long is left on the timer", "what reminders are set",
])
def test_list_reminders(text):
    assert call_of(text) == ("list_reminders", {})


@pytest.mark.parametrize("text,query", [
    ("cancel the timer", "timer"), ("stop the timer", "timer"), ("cancel all timers", "all timers"),
    ("cancel my reminders", "reminders"), ("cancel the reminder about the dentist", "reminder the dentist"),
    ("delete the pasta timer", "pasta timer"), ("cancel the 10 minute timer", "10 minute timer"),
    ("cancel all reminders", "all reminders"),
])
def test_cancel(text, query):
    assert call_of(text) == ("cancel_reminder", {"query": query})


@pytest.mark.parametrize("text", ["cancel my subscription", "stop the music", "cancel the meeting", "stop it from raining"])
def test_cancel_negatives(text):
    fp = match(text, NOW)
    assert fp is None or fp.kind != "reminders"


@pytest.mark.parametrize("text", [
    "give me my briefing", "morning briefing", "briefing", "Jarvis, my briefing please", "what's my day",
    "what's my day like", "tell me my briefing", "brief me", "what does my day look like",
])
def test_briefing(text):
    assert call_of(text) == ("briefing", {})


@pytest.mark.parametrize("text", ["what's the weather", "good morning", "brief the team", "what's my name", "my day was long"])
def test_briefing_negatives(text):
    fp = match(text, NOW)
    assert fp is None or fp.kind != "briefing"


def test_existing_fast_paths_unaffected():
    assert match("what time is it", NOW).kind == "time"
    assert match("pause the music", NOW).kind == "music"
    assert match("stop", NOW).stop_speaking
