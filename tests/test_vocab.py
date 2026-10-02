import tomllib

from jarvis import vocab

OLD = """[ollama]
model = "qwen3:14b"

[whisper]
model = "large-v3-turbo"
language = "en"

[tts]
voice = "bm_george"
"""


def test_add_words_and_model_without_existing_vocab(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(OLD)
    assert vocab.main(["add", "Linkin Park", "Claude", "--config", str(p)]) == 0
    assert vocab.main(["model", "large-v3", "--config", str(p)]) == 0
    data = tomllib.loads(p.read_text())
    assert data["whisper"]["model"] == "large-v3"
    assert data["whisper"]["vocabulary"][-1] == "Linkin Park"
    assert data["whisper"]["vocabulary"].count("Claude") == 1
    assert data["tts"]["voice"] == "bm_george" and data["ollama"]["model"] == "qwen3:14b"
    assert (tmp_path / "config.toml.bak").exists()


def test_remove_and_existing_list(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(OLD.replace('language = "en"', 'language = "en"\nvocabulary = ["Jarvis", "Discord"]'))
    vocab.main(["remove", "discord", "--config", str(p)])
    assert tomllib.loads(p.read_text())["whisper"]["vocabulary"] == ["Jarvis"]
