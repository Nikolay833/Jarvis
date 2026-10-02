"""What Claude Code has been doing: progress and change summaries from tool-call records.

Records come from two places: the transcript (`~/.claude/projects/.../<id>.jsonl`, tool_use + tool_result
blocks) and live hook events (PostToolUse / PostToolUseFailure forwarded by jarvis.claude_hook). Both are
folded into the same `ToolRec`, so one summariser serves the progress tool, the change tool and the Stop
announcement. Pure logic plus one read-only `git` call in `git_numstat`. Nothing here raises on bad input.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .claude_sessions import _blocks, _is_noise, _parse_ts, _text_of, basename

EDIT_TOOLS = {"Edit", "MultiEdit", "Write", "NotebookEdit"}
SHELL_TOOLS = {"Bash", "PowerShell"}
MAX_TRANSCRIPT_BYTES = 8 * 1024 * 1024  # newest part of a transcript that is read
MAX_RESULT_CHARS = 1500
MAX_RECORDS = 600

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"


@dataclass
class ToolRec:
    name: str
    file: str = ""  # file_path / notebook_path for edit tools
    command: str = ""  # shell command
    result: str = ""  # tail of the tool result text
    is_error: bool = False
    ts: float | None = None
    done: bool = True  # False: tool_use without a result yet (still running)
    created: bool = False  # a Write that made a new file
    use_id: str = ""


def _target(rec: ToolRec) -> str:
    return basename(rec.file) if rec.file else ""


# ---- tests ---------------------------------------------------------------------------------------------------
_TEST_CMD = re.compile(
    r"(?<![\w-])(?:pytest|py\.test|unittest|tox|nox|jest|vitest|mocha|playwright test|rspec|phpunit|ctest|"
    r"(?:npm|yarn|pnpm|bun)(?: run)? (?:test|t)\b|cargo (?:test|nextest)|go test|dotnet test|mvn(?:w)? (?:\S+ )*test|"
    r"gradle(?:w)? (?:\S+ )*test|make (?:test|check)|swift test|deno test)\b", re.I)
_FAIL_RES = [re.compile(p, re.I | re.M) for p in (
    r"\b([1-9]\d*) (?:failed|failing|failures?)\b", r"\bfailures?=([1-9]\d*)", r"\berrors?=([1-9]\d*)",
    r"^(?:FAIL|FAILED)\b", r"\btest result: FAILED", r"^--- FAIL", r"\bTests?:.*\b[1-9]\d* failed",
    r"\b([1-9]\d*) errors?\b(?! generated)", r"\bERRORS?\b.*\bcollecting\b", r"^=+ .*(?:FAILURES|ERRORS) =+")]
_PASS_RES = [re.compile(p, re.I | re.M) for p in (
    r"\b[1-9]\d* (?:passed|passing)\b", r"\btest result: ok\b", r"^OK\b", r"^ok\s+\S+", r"\bTests?:.*\bpassed\b",
    r"\ball tests passed\b", r"\bpassed\b.*\bin [\d.]+s")]


def is_test_command(command: str) -> bool:
    return bool(_TEST_CMD.search(command or ""))


def test_outcome(result: str, is_error: bool = False) -> str:
    """pass / fail / unknown from a test command's output text."""
    text = result or ""
    if any(rx.search(text) for rx in _FAIL_RES):
        return FAIL
    if any(rx.search(text) for rx in _PASS_RES):
        return PASS
    return FAIL if is_error else UNKNOWN


test_outcome.__test__ = False  # not a pytest test
is_test_command.__test__ = False


# ---- summary -------------------------------------------------------------------------------------------------
@dataclass
class Progress:
    files: list[str] = field(default_factory=list)  # unique paths, in first-touched order
    created: list[str] = field(default_factory=list)
    commands: int = 0  # shell commands that are not test runs
    tests: list[str] = field(default_factory=list)  # outcome per test run, in order
    tools: Counter = field(default_factory=Counter)
    started: float | None = None
    last_ts: float | None = None
    current: ToolRec | None = None  # tool call still running
    last: ToolRec | None = None  # newest tool call
    count: int = 0


def summarize(recs: list[ToolRec], started: float | None = None) -> Progress:
    p = Progress(started=started)
    for r in recs:
        p.count += 1
        p.tools[r.name] += 1
        if r.ts is not None:
            p.last_ts = r.ts if p.last_ts is None else max(p.last_ts, r.ts)
            if p.started is None:
                p.started = r.ts
        if r.name in EDIT_TOOLS and r.file:
            if r.file not in p.files:
                p.files.append(r.file)
            if r.created and r.file not in p.created:
                p.created.append(r.file)
        elif r.name in SHELL_TOOLS and r.command:
            if is_test_command(r.command):
                p.tests.append(test_outcome(r.result, r.is_error) if r.done else UNKNOWN)
            else:
                p.commands += 1
        p.last = r
        if not r.done:
            p.current = r
    return p


_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def number_word(n: int) -> str:
    return _WORDS[n] if 0 <= n < len(_WORDS) else str(n)


def plural(n: int, noun: str) -> str:
    return f"{number_word(n)} {noun}" + ("" if n == 1 else "s")


def duration(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 45:
        return f"{max(s, 1)} second" + ("" if s == 1 else "s")
    m = round(s / 60)
    if m < 60:
        return "a minute" if m <= 1 else f"{m} minutes"
    h, m = divmod(m, 60)
    out = "an hour" if h == 1 else f"{h} hours"
    return out + (f" {m} minutes" if m else "")


def _times(n: int) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{number_word(n)} times")


def tests_clause(p: Progress, short: bool = False) -> str:
    """'tests ran twice, last run passing' (or 'tests passing' when short); '' without test runs."""
    if not p.tests:
        return ""
    last = p.tests[-1]
    word = {PASS: "passing", FAIL: "failing"}.get(last)
    if short:
        return f"tests {word}" if word else "tests ran"
    base = f"tests ran {_times(len(p.tests))}"
    if len(p.tests) == 1:
        return base + (f", {word}" if word else ", result unclear")
    return base + (f", last run {word}" if word else ", last result unclear")


_VERB = {"Edit": "editing", "MultiEdit": "editing", "Write": "writing", "NotebookEdit": "editing", "Read": "reading",
         "Grep": "searching", "Glob": "searching", "WebFetch": "fetching", "WebSearch": "searching the web",
         "Task": "delegating", "Agent": "delegating", "TodoWrite": "planning"}


def activity(rec: ToolRec | None) -> str:
    """'editing map.ts' / 'running tests' / 'running a command'."""
    if rec is None:
        return ""
    if rec.name in SHELL_TOOLS:
        return "running tests" if is_test_command(rec.command) else "running a command"
    verb = _VERB.get(rec.name, f"using {rec.name}" if rec.name else "")
    t = _target(rec)
    return f"{verb} {t}".strip() if t else verb


def progress_sentence(p: Progress, project: str, busy: bool, now: float | None = None) -> str:
    """Spoken status of the current turn."""
    where = f" in {project}" if project else ""
    if p.count == 0:
        return f"Claude has not used any tools yet in this turn{where}, sir." if busy else \
            f"Claude has nothing to report{where}, sir."
    parts = []
    if p.files:
        parts.append(f"{plural(len(p.files), 'file')} edited")
    if p.created:
        parts.append(f"{number_word(len(p.created))} of them new")
    if p.commands:
        parts.append(f"{plural(p.commands, 'shell command')} run")
    t = tests_clause(p)
    if t:
        parts.append(t)
    if not parts:
        parts.append(f"{plural(p.count, 'tool call')} so far")
    end = (now if busy and now is not None else p.last_ts)
    secs = (end - p.started) if (end is not None and p.started is not None) else None
    if busy:
        head = f"Claude has been at it for {duration(secs)}{where}" if secs is not None else f"Claude is at work{where}"
    else:
        head = f"Claude worked for {duration(secs)}{where}" if secs is not None else f"Claude has been working{where}"
    out = f"{head}: {', '.join(parts)}."
    cur = activity(p.current or (p.last if busy else None))
    if busy and cur:
        out += f" He's currently {cur}."
    elif not busy:
        out += " He is idle at the moment."
    return out


def stop_sentence(p: Progress, project: str, said: str) -> str:
    """The 'finished' announcement."""
    where = f" in {project}" if project else ""
    bits = []
    if p.files:
        bits.append(f"{plural(len(p.files), 'file')} changed")
    t = tests_clause(p, short=True)
    if t:
        bits.append(t)
    out = f"Sir, Claude has finished{where}" + (f": {', '.join(bits)}." if bits else ".")
    if said:
        out += f" He says: {said}"
    return out


# ---- transcript ----------------------------------------------------------------------------------------------
def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(str(b.get("text", "")) for b in _blocks(content) if b.get("type") == "text")


def _tail(text: str, n: int = MAX_RESULT_CHARS) -> str:
    return text if len(text) <= n else text[-n:]


def _read_lines(path: str, max_bytes: int) -> list[str]:
    try:
        p = Path(path)
        size = p.stat().st_size
        with p.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                data = fh.read().decode("utf-8", "replace").splitlines()[1:]  # first line is cut off
            else:
                data = fh.read().decode("utf-8", "replace").splitlines()
        return data
    except OSError:
        return []


def read_transcript(path: str, max_bytes: int = MAX_TRANSCRIPT_BYTES) -> tuple[list[ToolRec], int, float | None]:
    """(records oldest first, index of the first record of the current turn, timestamp of its user message)."""
    recs: list[ToolRec] = []
    by_id: dict[str, ToolRec] = {}
    turn_idx, turn_ts = 0, None
    if not path:
        return recs, turn_idx, turn_ts
    for line in _read_lines(path, max_bytes):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict) or ev.get("type") not in ("user", "assistant") or ev.get("isMeta"):
            continue
        msg = ev.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        ts = _parse_ts(ev.get("timestamp"))
        if ev["type"] == "assistant":
            for b in _blocks(content):
                if b.get("type") != "tool_use":
                    continue
                inp = b.get("input") if isinstance(b.get("input"), dict) else {}
                rec = ToolRec(name=str(b.get("name", "")), ts=ts, done=False, use_id=str(b.get("id", "")),
                              file=str(inp.get("file_path") or inp.get("notebook_path") or ""),
                              command=str(inp.get("command") or ""))
                recs.append(rec)
                if rec.use_id:
                    by_id[rec.use_id] = rec
            continue
        blocks = _blocks(content)
        results = [b for b in blocks if b.get("type") == "tool_result"]
        for b in results:
            rec = by_id.get(str(b.get("tool_use_id", "")))
            if rec is None:
                continue
            rec.done = True
            rec.is_error = bool(b.get("is_error"))
            rec.result = _tail(_result_text(b.get("content")))
            if rec.name == "Write" and re.search(r"\bcreated\b", rec.result[:200], re.I):
                rec.created = True
            if ts is not None:
                rec.ts = ts
        text = _text_of(content)
        if not results and text and not _is_noise(text) and not ev.get("isSidechain"):
            turn_idx, turn_ts = len(recs), ts or turn_ts
    return recs, turn_idx, turn_ts


def turn_records(path: str) -> tuple[list[ToolRec], float | None]:
    recs, idx, ts = read_transcript(path)
    return recs[idx:], ts


# ---- live hook events ----------------------------------------------------------------------------------------
def rec_from_event(msg: dict[str, Any], ts: float | None) -> ToolRec | None:
    """ToolRec from a forwarded PostToolUse / PostToolUseFailure `claude_event` (None for other events)."""
    event = str(msg.get("event") or "")
    if event not in ("PostToolUse", "PostToolUseFailure"):
        return None
    name = str(msg.get("tool_name") or "")
    if not name:
        return None
    result = str(msg.get("tool_result") or "")
    return ToolRec(name=name, file=str(msg.get("file_path") or ""), command=str(msg.get("command") or ""),
                   result=result, is_error=event == "PostToolUseFailure" or bool(msg.get("is_error")), ts=ts,
                   created=name == "Write" and bool(re.search(r"\bcreated\b", result[:200], re.I)))


# ---- changes -------------------------------------------------------------------------------------------------
@dataclass
class Changes:
    files: list[str]  # paths as Claude used them
    added: int | None = None
    removed: int | None = None
    git: bool = False


def changed_files(recs: list[ToolRec]) -> list[str]:
    out: list[str] = []
    for r in recs:
        if r.name in EDIT_TOOLS and r.file and r.file not in out:
            out.append(r.file)
    return out


def git_numstat(project_dir: str, files: list[str], timeout: float = 8.0) -> tuple[int, int] | None:
    """(added, removed) lines from read-only `git diff --numstat` for `files` (all changes if empty), or None.

    New untracked files count all their lines as added. None when the folder is not a git repository."""
    def run(args: list[str]) -> str | None:
        try:
            r = subprocess.run(["git", "-C", project_dir, *args], capture_output=True, text=True, timeout=timeout,
                               check=False, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout if r.returncode == 0 else None

    if run(["rev-parse", "--is-inside-work-tree"]) is None:
        return None
    rel = []
    for f in files:
        try:
            rel.append(os.path.relpath(f, project_dir) if os.path.isabs(f) else f)
        except ValueError:
            rel.append(f)
    spec = ["--", *rel] if rel else []
    out = run(["diff", "HEAD", "--numstat", *spec])
    if out is None:  # no commits yet
        out = run(["diff", "--numstat", *spec]) or ""
    added = removed = 0
    for line in out.splitlines():
        a, _, rest = line.partition("\t")
        d = rest.partition("\t")[0]
        if a.isdigit():
            added += int(a)
        if d.isdigit():
            removed += int(d)
    untracked = (run(["ls-files", "--others", "--exclude-standard", *spec]) or "").splitlines()
    for u in untracked[:50]:
        try:
            with open(os.path.join(project_dir, u), "rb") as fh:
                added += sum(1 for _ in fh)
        except OSError:
            pass
    return added, removed


def changes_sentence(files: list[str], stat: tuple[int, int] | None, project: str = "") -> str:
    where = f" in {project}" if project else ""
    if not files:
        if stat and (stat[0] or stat[1]):
            return f"Claude's transcript shows no edits{where}, sir, but git sees about {_lines(stat)}."
        return f"Claude has not changed any files{where}, sir."
    names = [basename(f) for f in files]
    shown = ", ".join(names[:4]) + (f" and {number_word(len(names) - 4)} more" if len(names) > 4 else "")
    out = f"{plural(len(names), 'file').capitalize()}{where}: {shown}"
    if stat is not None and (stat[0] or stat[1]):
        out += f"; about {_lines(stat)}"
    return out + "."


def _lines(stat: tuple[int, int]) -> str:
    return f"{stat[0]} lines added, {stat[1]} removed"
