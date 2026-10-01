"""Shared helpers for the Claude session tests (imported as a plain module; pytest puts tests/ on sys.path)."""

import json
import os

import pytest

from jarvis import claude_sessions as cs
from jarvis import paths


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    paths.clear_cache()
    cs.clear_cache()
    yield tmp_path.resolve()
    paths.clear_cache()
    cs.clear_cache()


def U(text, cwd=None, ts=None, **kw):
    e = {"type": "user", "message": {"role": "user", "content": text}, **kw}
    if cwd:
        e["cwd"] = cwd
    if ts:
        e["timestamp"] = ts
    return e


def A(blocks, **kw):
    return {"type": "assistant", "message": {"role": "assistant", "content": blocks}, **kw}


def T(text):
    return {"type": "text", "text": text}


def make(home, cwd, sid, entries, mtime=None):
    d = home / ".claude" / "projects" / cs.encode_project(str(cwd))
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{sid}.jsonl"
    f.write_text("\n".join(e if isinstance(e, str) else json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    if mtime:
        os.utime(f, (mtime, mtime))
    return f


def proj(home, *parts):
    p = home.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
