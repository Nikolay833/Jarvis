"""Read recent Claude Code conversations from the local transcripts (~/.claude/projects)."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from ..paths import resolve_path
from .claude_code import _describe, jobs
from .registry import ToolError, tool

MSG_CHARS = 400
TOTAL_CHARS = 2500
TAIL_BYTES = 2 * 1024 * 1024
MAX_COUNT = 20


def projects_dir() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR")
    return (Path(base) if base else Path.home() / ".claude") / "projects"


def encode_project(path: str) -> str:
    """Claude Code's folder naming: every non-alphanumeric character becomes '-'."""
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def _newest(d: Path) -> float:
    try:
        return max((f.stat().st_mtime for f in d.glob("*.jsonl")), default=0.0)
    except OSError:
        return 0.0


def find_project_dir(folder: str) -> Path:
    root = projects_dir()
    dirs = [d for d in root.iterdir() if d.is_dir()] if root.is_dir() else []
    dirs = [d for d in dirs if any(d.glob("*.jsonl"))]
    if not dirs:
        raise ToolError(f"No Claude Code transcripts found in {root}")
    if not folder.strip():
        return max(dirs, key=_newest)
    wanted = encode_project(str(resolve_path(folder)))
    for d in dirs:
        if d.name == wanted:
            return d
    for d in dirs:
        if d.name.lower() == wanted.lower():
            return d
    parts = [p for p in re.split(r"[\\/]", folder.strip().rstrip("\\/")) if p]
    last = encode_project(parts[-1]).lower() if parts else ""
    if last:
        hits = [d for d in dirs if d.name.lower().endswith("-" + last)] or [d for d in dirs if last in d.name.lower()]
        if hits:
            return max(hits, key=_newest)
    raise ToolError(f"No Claude Code transcripts found for folder '{folder}'")


def _read_tail_lines(path: Path) -> list[str]:
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > TAIL_BYTES:
            fh.seek(size - TAIL_BYTES)
            fh.readline()  # drop the partial first line
        data = fh.read()
    return data.decode("utf-8", errors="replace").splitlines()


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)]
        return "\n".join(p.strip() for p in parts if p.strip())
    return ""


def parse_transcript(lines: list[str]) -> tuple[list[tuple[str, str]], str]:
    """Return ([(role, text)], cwd). role is "user" or "claude"."""
    msgs: list[tuple[str, str]] = []
    cwd = ""
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict) or ev.get("type") not in ("user", "assistant"):
            continue
        if ev.get("isMeta") or ev.get("isSidechain"):
            continue
        if isinstance(ev.get("cwd"), str) and ev["cwd"]:
            cwd = ev["cwd"]
        message = ev.get("message")
        if not isinstance(message, dict):
            continue
        text = _text_of(message.get("content"))
        if text:
            msgs.append(("user" if ev["type"] == "user" else "claude", text))
    return msgs, cwd


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


@tool("Show the latest messages of the most recent Claude Code conversation on this PC (optionally for one "
      "project folder). Use it to check what Claude said or did last.")
def claude_code_history(folder: str = "", count: int = 4) -> str:
    """Recent Claude Code conversation.

    Args:
        folder: Project folder or just its name. Empty means the most recently active conversation.
        count: How many of the last messages to show.
    """
    count = max(1, min(int(count), MAX_COUNT))
    pdir = find_project_dir(folder)
    files = sorted(pdir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    msgs: list[tuple[str, str]] = []
    cwd = ""
    session = files[0]
    for f in files[:3]:  # newest session may be empty (just started)
        msgs, cwd = parse_transcript(_read_tail_lines(f))
        session = f
        if msgs:
            break
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(session.stat().st_mtime))
    head = f"Project: {cwd or pdir.name}\nLast activity: {when}\n"
    tail = [f"{role}: {_clip(text, MSG_CHARS)}" for role, text in msgs[-count:]] or ["(no messages)"]
    extra = ""
    if jobs.jobs:
        extra = "\nJarvis-run Claude Code jobs:\n" + "\n".join(_describe(j) for j in list(jobs.jobs.values())[-3:])
    else:
        extra = "\n(Jobs started by Jarvis: use claude_code_status.)"
    budget = TOTAL_CHARS - len(head) - len(extra)
    while len(tail) > 1 and sum(len(t) + 1 for t in tail) > budget:
        tail.pop(0)  # drop oldest first
    body = "\n".join(tail)
    if len(body) > budget:
        body = body[: max(budget - 3, 0)] + "..."
    return head + body + extra
