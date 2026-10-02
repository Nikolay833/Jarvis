"""Turns Claude Code hook events (bus message type "claude_event") into spoken announcements.

Pure logic: `ClaudeWatch.handle(msg)` returns the sentence to announce, or None. The assistant speaks it
through `Assistant.announce`, which waits for the current turn to end.

"Finished" announcements honour `min_turn_seconds`: the turn length is the time between the last real user
message in the transcript and now. If the transcript or its timestamps cannot be read, Jarvis announces anyway.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from . import claude_progress as cp
from .claude_sessions import basename, first_sentences, speakable, turn_seconds
from .config import ClaudeWatchConfig
from .safety import RISKY, classify_shell

THROTTLE_SECONDS = 30.0
MAX_WORDS = 25

LIVE_TTL = 3600.0  # live tool records of a session that went quiet this long are dropped
ALLOW, DENY, ASK = "allow", "deny", "ask"  # ASK: no decision, Claude shows its own prompt

QUESTION_TYPES = ("agent_needs_input", "elicitation")  # prefix match: elicitation, elicitation_dialog, ...


def _records_from_transcript(path: str) -> list[cp.ToolRec]:
    return cp.turn_records(path)[0]


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "..."


def describe_permission(tool: str, tool_input: dict[str, Any]) -> tuple[str, str]:
    """(what Claude wants to do, warning or ''): 'run npm install', 'edit map.ts'."""
    inp = tool_input if isinstance(tool_input, dict) else {}
    if tool in cp.SHELL_TOOLS:
        cmd = str(inp.get("command") or "")
        risk = classify_shell(cmd)
        return f"run {_clip(speakable(cmd) or cmd, 120)}", (risk.reason if risk.risk == RISKY else "")
    f = basename(str(inp.get("file_path") or inp.get("notebook_path") or ""))
    if tool in ("Edit", "MultiEdit", "NotebookEdit"):
        return f"edit {f}" if f else "edit a file", ""
    if tool == "Write":
        return f"write {f}" if f else "write a file", ""
    if tool == "Read":
        return f"read {f}" if f else "read a file", ""
    if tool == "WebFetch":
        host = re.sub(r"^https?://", "", str(inp.get("url") or "")).split("/")[0]
        return f"fetch {host}" if host else "fetch a web page", ""
    if tool == "WebSearch":
        return "search the web", ""
    if tool.startswith("mcp__"):
        return f"use the {tool.split('__')[-1].replace('_', ' ')} tool", ""
    return f"use the {tool or 'a'} tool", ""


def permission_texts(msg: dict[str, Any]) -> tuple[str, str]:
    """(summary for the orb button, spoken question) for a forwarded PermissionRequest."""
    project = basename(str(msg.get("cwd") or "")) or "your project"
    what, warn = describe_permission(str(msg.get("tool_name") or ""), msg.get("tool_input") or {})
    spoken = f"Sir, Claude wants to {what} in {project}. "
    if warn:
        spoken += f"Careful: {warn}. "
    return f"let Claude {what} in {project}", spoken + "Shall I allow it?"


@dataclass
class Pending:
    kind: str  # "permission"
    message: str
    project: str
    since: float


class ClaudeWatch:
    def __init__(self, cfg: ClaudeWatchConfig | None = None, clock: Callable[[], float] = time.time,
                 turn_seconds_fn: Callable[[str], float | None] = lambda path: turn_seconds(path),
                 records_fn: Callable[[str], list[cp.ToolRec]] = _records_from_transcript) -> None:
        self.cfg = cfg or ClaudeWatchConfig()
        self._clock = clock
        self._turn_seconds = turn_seconds_fn
        self._records = records_fn
        self.live: dict[str, list[cp.ToolRec]] = {}  # session id -> tool calls since its last Stop (hook events)
        self.asking: set[str] = set()  # sessions with a voice permission question in flight
        self.pending: dict[str, Pending] = {}  # session id -> latest pending permission request
        self._last: dict[tuple[str, str], float] = {}

    def pending_for(self, session_id: str) -> Pending | None:
        return self.pending.get(session_id)

    def _throttled(self, session_id: str, kind: str) -> bool:
        now = self._clock()
        last = self._last.get((session_id, kind))
        if last is not None and now - last < THROTTLE_SECONDS:
            return True
        self._last[(session_id, kind)] = now
        return False

    def live_records(self, session_id: str) -> list[cp.ToolRec]:
        return list(self.live.get(session_id, []))

    def _track(self, sid: str, msg: dict[str, Any]) -> None:
        now = self._clock()
        for k in [k for k, v in self.live.items() if v and (v[-1].ts or 0) < now - LIVE_TTL]:
            del self.live[k]
        rec = cp.rec_from_event(msg, now)
        if rec is not None and sid:
            recs = self.live.setdefault(sid, [])
            recs.append(rec)
            del recs[:-cp.MAX_RECORDS]

    def _finish_progress(self, sid: str, msg: dict[str, Any]) -> cp.Progress:
        recs = self.live.pop(sid, [])
        if not recs:
            try:
                recs = self._records(str(msg.get("transcript_path") or ""))
            except Exception:  # noqa: BLE001
                recs = []
        return cp.summarize(recs)

    async def decide_permission(self, msg: dict[str, Any], confirmer: Any) -> str:
        """Ask the user by voice about a Claude permission prompt. Returns ALLOW, DENY or ASK (no answer or off)."""
        sid = str(msg.get("session_id") or "")
        if not (self.cfg.enabled and self.cfg.voice_approval):
            return ASK
        summary, spoken = permission_texts(msg)
        self.asking.add(sid)
        try:
            verdict = await confirmer.ask(summary, prompt=spoken, timeout_none=True)
        except Exception:  # noqa: BLE001 - never decide on a failure
            return ASK
        finally:
            self.asking.discard(sid)
        return ASK if verdict is None else (ALLOW if verdict else DENY)

    def handle(self, msg: dict[str, Any]) -> str | None:
        event = str(msg.get("event") or "")
        sid = str(msg.get("session_id") or "")
        project = basename(str(msg.get("cwd") or "")) or "your project"
        if event in ("PostToolUse", "PostToolUseFailure"):
            self._track(sid, msg)
            return None
        if event == "Stop":
            self.pending.pop(sid, None)
            prog = self._finish_progress(sid, msg)
            if not (self.cfg.enabled and self.cfg.announce_finish):
                return None
            secs = None
            try:
                secs = self._turn_seconds(str(msg.get("transcript_path") or ""))
            except Exception:  # noqa: BLE001
                secs = None
            if secs is not None and secs < self.cfg.min_turn_seconds:
                return None
            if self._throttled(sid, "stop"):
                return None
            said = first_sentences(str(msg.get("last_assistant_message") or ""), 1, MAX_WORDS)
            return cp.stop_sentence(prog, project, said)
        if event == "Notification":
            ntype = str(msg.get("notification_type") or "")
            if ntype == "permission_prompt":
                self.pending[sid] = Pending("permission", str(msg.get("message") or ""), project, self._clock())
                if (not (self.cfg.enabled and self.cfg.announce_permission) or sid in self.asking
                        or self._throttled(sid, ntype)):
                    return None
                return f"Sir, Claude needs your permission in {project}."
            if ntype.startswith(QUESTION_TYPES):
                if not (self.cfg.enabled and self.cfg.announce_permission) or self._throttled(sid, "question"):
                    return None
                return f"Sir, Claude has a question for you in {project}."
        return None  # idle_prompt (Stop already covered it) and anything else


watch = ClaudeWatch()
