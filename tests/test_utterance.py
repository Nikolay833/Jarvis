import pytest

from jarvis.utterance import asks_question, sounds_unfinished


@pytest.mark.parametrize("text", ["Open Chrome with", "open chrome with...", "create a folder called",
                                  "play some", "open chrome as", "I want to, um", "search for the"])
def test_unfinished(text):
    assert sounds_unfinished(text)


@pytest.mark.parametrize("text", ["Open Chrome.", "open chrome with my work profile", "what time is it?",
                                  "I'm nervous", "", "play numb on spotify"])
def test_finished(text):
    assert not sounds_unfinished(text)


def test_asks_question():
    assert asks_question("Shall I play some relaxing music, sir?")
    assert asks_question('Would you like me to "open it"?')
    assert not asks_question("Playing relaxing music, sir.")
    assert not asks_question("")


@pytest.mark.parametrize("heard, fixed", [
    ("Open cloud Jarvis PC assistant AI session", "Open Claude Jarvis PC assistant AI session"),
    ("What's the last message I got from Cloud Code?", "What's the last message I got from Claude Code?"),
    ("ask clyde what 11 times 12 is", "ask Claude what 11 times 12 is"),
    ("open the cloud session for jarvis", "open the Claude session for jarvis"),
    ("tell Clod to add tests", "tell Claude to add tests"),
    ("upload this to cloud storage", "upload this to cloud storage"),
    ("open a new cloud chat", "open a new Claude chat"),
])
def test_fix_names(heard, fixed):
    from jarvis.utterance import fix_names

    assert fix_names(heard) == fixed


def test_fix_names_leaves_weather_clouds():
    from jarvis.utterance import fix_names

    assert fix_names("are there clouds today") == "are there clouds today"
    assert fix_names("it is cloudy") == "it is cloudy"


def test_list_numbers_do_not_split_sentences():
    from jarvis.audio.tts import SentenceBuffer, split_sentences

    text = "Here they are. 1. Login bug, two hours ago. 2. Settings page, yesterday."
    assert split_sentences(text) == ["Here they are.", "1. Login bug, two hours ago.", "2. Settings page, yesterday."]
    buf, got = SentenceBuffer(), []
    for ch in text + " ":
        got += buf.feed(ch)
    got += buf.flush()
    assert got == ["Here they are.", "1. Login bug, two hours ago.", "2. Settings page, yesterday."]
    assert split_sentences("Ask Dr. Smith. Then go.") == ["Ask Dr. Smith.", "Then go."]


@pytest.mark.parametrize("text", ["thanks", "Thank you, Jarvis.", "that's all", "okay thanks", "no", "never mind",
                                  "bye", "I'm good"])
def test_conversation_enders(text):
    from jarvis.__main__ import ENDERS

    assert ENDERS.match(text.strip().lower())


@pytest.mark.parametrize("text", ["no, open chrome instead", "thanks, now play some music", "what's the time"])
def test_not_enders(text):
    from jarvis.__main__ import ENDERS

    assert not ENDERS.match(text.strip().lower())


def test_spoken_session_list_uses_ordinals():
    from types import SimpleNamespace

    from jarvis.claude_sessions import format_list

    sessions = [SimpleNamespace(label="Login bug", project_name="Jarvis", last_activity=0.0),
                SimpleNamespace(label="Settings", project_name="Jarvis", last_activity=0.0)]
    out = format_list(sessions, now=7200.0)
    assert out.startswith('The first is "Login bug" in Jarvis, from 2 hours ago. The second is')
    assert "1." not in out
