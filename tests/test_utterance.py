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
