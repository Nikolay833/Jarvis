import json
import os
import time
from pathlib import Path

import pytest

from jarvis import paths, safety
from jarvis.tools import claude_history as ch
from jarvis.tools import files
from jarvis.tools.registry import ToolError, registry


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    paths.clear_cache()
    yield tmp_path.resolve()
    monkeypatch.undo()
    paths.clear_cache()


# ---- paths --------------------------------------------------------------------------------------
def test_resolve_names_and_relative(home):
    assert paths.resolve_path("desktop") == home / "Desktop"
    assert paths.resolve_path("My Desktop/x y") == home / "Desktop" / "x y"
    assert paths.resolve_path("DOWNLOADS\\a\\b") == home / "Downloads" / "a" / "b"
    assert paths.resolve_path("~/stuff") == home / "stuff"
    assert paths.resolve_path("stuff") == home / "stuff"
    assert paths.resolve_path("desktopper") == home / "desktopper"


def test_resolve_redirected_desktop(home, monkeypatch):
    real = home / "OneDrive" / "Desktop"
    real.mkdir(parents=True)
    monkeypatch.setattr(paths, "known_folder", lambda n: real if n == "desktop" else home / n.capitalize())
    assert paths.resolve_path("desktop/a") == real / "a"
    assert paths.resolve_path("~/Desktop/a") == real / "a"  # home/Desktop does not exist


def test_folder_hint(home):
    h = paths.folder_hint()
    assert str(home) in h and "Desktop" in h and "Downloads" in h


# ---- file tools -----------------------------------------------------------------------------------
def test_create_folder_and_write_file(home):
    assert "Created folder" in files.create_folder("desktop/new/deep")
    assert (home / "Desktop" / "new" / "deep").is_dir()
    assert "already exists" in files.create_folder("desktop/new/deep")
    (home / "f.txt").write_text("x")
    with pytest.raises(ToolError):
        files.create_folder("f.txt")
    assert "Created" in files.write_file("desktop/a/n.txt", "hello")
    assert (home / "Desktop" / "a" / "n.txt").read_text() == "hello"
    with pytest.raises(ToolError):
        files.write_file("desktop/a/n.txt", "again")
    assert "Overwrote" in files.write_file("desktop/a/n.txt", "again", overwrite=True)


def test_registered_with_risk():
    import jarvis.tools as t

    reg = t.load_all()
    assert reg.get("create_folder").risk == "safe" and reg.get("write_file").risk == "safe"
    assert reg.get("claude_code_history").risk == "safe"


def test_write_file_safety(home):
    c = safety.classify_call
    assert not c("write_file", {"path": "desktop/new.txt", "content": "x"}).is_risky
    (home / "old.txt").write_text("x")
    assert not c("write_file", {"path": "old.txt", "content": "x"}).is_risky  # tool refuses itself
    assert c("write_file", {"path": "old.txt", "content": "x", "overwrite": True}).is_risky
    assert c("write_file", {"path": "old.txt", "content": "x", "overwrite": "true"}).is_risky
    assert c("write_file", {"path": "run.bat", "content": "x"}).is_risky
    assert c("write_file", {"path": "a.PS1", "content": "x"}).is_risky
    assert not c("create_folder", {"path": "x"}).is_risky
    assert "write" in safety.describe_call("write_file", {"path": "a.txt"})


# ---- claude history --------------------------------------------------------------------------------
def write_session(home, project, name, entries, mtime=None):
    d = home / ".claude" / "projects" / ch.encode_project(project)
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_text("\n".join(json.dumps(e) if not isinstance(e, str) else e for e in entries) + "\n")
    if mtime:
        os.utime(f, (mtime, mtime))
    return f


def U(content, **kw):
    return {"type": "user", "message": {"role": "user", "content": content}, "cwd": "C:\\Users\\Nik\\proj", **kw}


def A(content, **kw):
    return {"type": "assistant", "message": {"role": "assistant", "content": content}, **kw}


def test_encode():
    assert ch.encode_project("C:\\Users\\Nikolay\\proj") == "C--Users-Nikolay-proj"


def test_history_parses_and_skips(home):
    write_session(home, "C:\\Users\\Nik\\proj", "s1.jsonl", [
        {"type": "summary", "summary": "x"},
        U("fix the bug"),
        A([{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Looking."},
           {"type": "tool_use", "name": "Read", "input": {}}]),
        U([{"type": "tool_result", "content": "file data"}]),
        U("meta noise", isMeta=True),
        A([{"type": "text", "text": "side"}], isSidechain=True),
        "not json {",
        A([{"type": "text", "text": "Fixed it. " + "z" * 600}]),
    ])
    out = ch.claude_code_history()
    assert "C:\\Users\\Nik\\proj" in out
    assert "user: fix the bug" in out and "claude: Looking." in out and "claude: Fixed it." in out
    for bad in ("tool_use", "file data", "meta noise", "side", "hmm"):
        assert bad not in out
    assert "..." in out and len(out) <= ch.TOTAL_CHARS
    assert "user: fix the bug" not in ch.claude_code_history(count=2)


def test_history_newest_project_and_folder_match(home):
    now = time.time()
    write_session(home, "C:\\Users\\Nik\\old", "a.jsonl", [U("old msg", cwd="C:\\Users\\Nik\\old")], now - 5000)
    write_session(home, "C:\\Users\\Nik\\New-App", "b.jsonl", [U("new msg", cwd="C:\\Users\\Nik\\New-App")], now)
    assert "new msg" in ch.claude_code_history()
    assert "old msg" in ch.claude_code_history(folder="C:\\Users\\Nik\\old")
    assert "old msg" in ch.claude_code_history(folder="c:\\users\\nik\\OLD")   # case-insensitive
    assert "new msg" in ch.claude_code_history(folder="new-app")               # fuzzy last component
    assert "old msg" in ch.claude_code_history(folder="ol")                    # contains
    with pytest.raises(ToolError):
        ch.claude_code_history(folder="nothing-like-it")


def test_history_total_cap_and_no_transcripts(home):
    with pytest.raises(ToolError, match="No Claude Code transcripts"):
        ch.claude_code_history()
    write_session(home, "/p/x", "s.jsonl", [U("q" * 390), A([{"type": "text", "text": "w" * 390}])] * 10)
    out = ch.claude_code_history(count=20)
    assert len(out) <= ch.TOTAL_CHARS


def test_history_mentions_jobs(home):
    from jarvis.tools.claude_code import Job, jobs

    write_session(home, "/p/x", "s.jsonl", [U("hi")])
    assert "claude_code_status" in ch.claude_code_history()
    jobs.jobs["zz"] = Job(id="zz", folder="/p/x", prompt="do")
    try:
        assert "[zz]" in ch.claude_code_history()
    finally:
        jobs.jobs.pop("zz")
