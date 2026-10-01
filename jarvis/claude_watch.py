"""Turns Claude Code hook events (bus message type "claude_event") into spoken announcements.

Pure logic: `ClaudeWatch.handle(msg)` returns the sentence to announce, or None. The assistant speaks it
through `Assistant.announce`, which waits for the current turn to end.

"Finished" announcements honour `min_turn_seconds`: the turn length is the time between the last real user
message in the transcript and now. If the transcript or its timestamps cannot be read, Jarvis announces anyway.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .claude_sessions import basename, first_sentences, turn_seconds
from .config import ClaudeWatchConfig

THROTTLE_SECONDS = 30.0
MAX_WORDS = 25

QUESTION_TYPES = ("agent_needs_input", "elicitation")  # prefix match: elicitation, elicitation_dialog, ...


@dataclass
class Pending:
    kind: str  # "permission"
    message: str
    project: str
    since: float


class ClaudeWatch:
    def __init__(self, cfg: ClaudeWatchConfig | None = None, clock: Callable[[], float] = time.time,
                 turn_seconds_fn: Callable[[str], float | None] = lambda path: turn_seconds(path)) -> None:
        self.cfg = cfg or ClaudeWatchConfig()
        self._clock = clock
        self._turn_seconds = turn_seconds_fn
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

    def handle(self, msg: dict[str, Any]) -> str | None:
        event = str(msg.get("event") or "")
        sid = str(msg.get("session_id") or "")
        project = basename(str(msg.get("cwd") or "")) or "your project"
        if event == "Stop":
            self.pending.pop(sid, None)
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
            return f"Sir, Claude finished in {project}: {said}" if said else f"Sir, Claude finished in {project}."
        if event == "Notification":
            ntype = str(msg.get("notification_type") or "")
            if ntype == "permission_prompt":
                self.pending[sid] = Pending("permission", str(msg.get("message") or ""), project, self._clock())
                if not (self.cfg.enabled and self.cfg.announce_permission) or self._throttled(sid, ntype):
                    return None
                return f"Sir, Claude needs your permission in {project}."
            if ntype.startswith(QUESTION_TYPES):
                if not (self.cfg.enabled and self.cfg.announce_permission) or self._throttled(sid, "question"):
                    return None
                return f"Sir, Claude has a question for you in {project}."
        return None  # idle_prompt (Stop already covered it) and anything else


watch = ClaudeWatch()
