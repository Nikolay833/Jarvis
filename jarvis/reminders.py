"""Reminders and timers: a persistent store plus an asyncio scheduler.

Store: %APPDATA%/Jarvis/reminders.json, a list of {id, kind: "reminder"|"timer", text, label, due (epoch s),
created, seconds}. The scheduler checks once a second and announces what is due through the callback it is
given (the assistant's out-of-turn announce, which waits for the current turn). Items that came due while
Jarvis was not running are announced once at start ("While you were away: ..."), or handed to the morning
briefing when that is about to speak anyway.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import store
from .memory import tokens
from .timeparse import format_when, spoken_duration

log = logging.getLogger("jarvis.reminders")


def _ts(dt: datetime) -> float:
    return dt.timestamp()


def _adjective(seconds: float) -> str:
    """'10 minute', '1 hour 30 minute' (for "your 10 minute timer")."""
    return re.sub(r"(\d+ (?:hour|minute|second))s\b", r"\1", spoken_duration(seconds))


class Ambiguous(Exception):
    """cancel() matched several items equally well."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        super().__init__("ambiguous")
        self.items = items


class ReminderStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or store.app_dir() / "reminders.json"
        self._lock = threading.RLock()
        data = store.read_json(self.path, [])
        self.items: list[dict[str, Any]] = [i for i in data if isinstance(i, dict) and "due" in i] \
            if isinstance(data, list) else []

    def _save(self) -> None:
        store.write_json(self.path, self.items)

    def _add(self, **fields: Any) -> dict[str, Any]:
        item = {"id": uuid.uuid4().hex[:8], "text": "", "label": "", "seconds": 0.0, **fields}
        with self._lock:
            self.items.append(item)
            self.items.sort(key=lambda i: i["due"])
            self._save()
        return item

    def add_timer(self, seconds: float, label: str, now: datetime) -> dict[str, Any]:
        return self._add(kind="timer", label=label.strip(), seconds=float(seconds), created=_ts(now),
                         due=_ts(now) + float(seconds))

    def add_reminder(self, text: str, due: datetime, now: datetime) -> dict[str, Any]:
        return self._add(kind="reminder", text=text.strip(), created=_ts(now), due=_ts(due))

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.items)

    def pop_due(self, now: datetime) -> list[dict[str, Any]]:
        """Remove and return everything due at `now` (oldest first)."""
        with self._lock:
            due = [i for i in self.items if i["due"] <= _ts(now)]
            if due:
                self.items = [i for i in self.items if i not in due]
                self._save()
            return due

    def cancel(self, query: str = "") -> list[dict[str, Any]]:
        """Cancel what `query` names: "all", a kind ("timer"), words from the text/label/duration.
        An empty query cancels the only item; several equally good matches raise Ambiguous."""
        q = (query or "").lower()
        with self._lock:
            if not self.items:
                return []
            if re.search(r"\b(all|every|everything)\b", q):
                kind = "timer" if "timer" in q else "reminder" if "reminder" in q else ""
                gone = [i for i in self.items if not kind or i["kind"] == kind]
            else:
                kind = "timer" if re.search(r"\btimers?\b", q) else "reminder" if re.search(r"\breminders?\b", q) else ""
                pool = [i for i in self.items if not kind or i["kind"] == kind]
                words = set(tokens(re.sub(r"\b(timers?|reminders?|cancel|stop|delete|remove|the|my)\b", " ", q)))
                if not words:
                    gone = pool if len(pool) == 1 else []
                    if len(pool) > 1:
                        raise Ambiguous(pool)
                else:
                    scored = [(len(words & set(tokens(describe_item(i)))) / len(words), i) for i in pool]
                    best = max((s for s, _ in scored), default=0.0)
                    if best < 0.5:
                        return []
                    gone = [i for s, i in scored if s == best]
                    if len(gone) > 1:
                        raise Ambiguous(gone)
            if gone:
                self.items = [i for i in self.items if i not in gone]
                self._save()
            return gone


def describe_item(item: dict[str, Any]) -> str:
    if item["kind"] == "timer":
        return f"{item['label']} timer" if item.get("label") else f"{_adjective(item.get('seconds') or 0)} timer"
    return f"reminder {item.get('text', '')}"


def describe_pending(item: dict[str, Any], now: datetime) -> str:
    due = datetime.fromtimestamp(item["due"])
    if item["kind"] == "timer":
        left = max(0.0, item["due"] - _ts(now))
        name = f"{item['label']} timer" if item.get("label") else f"{_adjective(item.get('seconds') or 0)} timer"
        return f"a {name} with {spoken_duration(left)} left" if left >= 1 else f"a {name} about to ring"
    return f"reminder to {item['text']} {format_when(due, now)}"


def due_text(item: dict[str, Any]) -> str:
    """What to say when `item` comes due."""
    if item["kind"] == "timer":
        if item.get("label"):
            return f"Sir, your {item['label']} timer is done."
        return f"Your {_adjective(item.get('seconds') or 0)} timer is done."
    return f"Sir, reminder: {item['text']}."


def missed_text(items: list[dict[str, Any]]) -> str:
    parts = []
    for i in items:
        if i["kind"] == "timer":
            name = f"{i['label']} timer" if i.get("label") else f"{_adjective(i.get('seconds') or 0)} timer"
            parts.append(f"your {name} finished")
        else:
            parts.append(f"reminder, {i['text']}")
    return "Sir, while you were away: " + "; ".join(parts) + "."


def play_reminder_chime(device: Any = None) -> None:
    """Gentle three-note descending chime (distinct from the listening chime). Never raises."""
    try:
        import numpy as np
        import sounddevice as sd

        rate = 24000
        parts = []
        for freq, dur in ((784.0, 0.11), (659.0, 0.11), (523.0, 0.2)):
            t = np.arange(int(dur * rate)) / rate
            env = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01) * np.exp(-t * 6)
            parts.append(np.sin(2 * np.pi * freq * t) * env)
        sd.play((np.concatenate(parts) * 0.18).astype(np.float32), rate, device=device)
        sd.wait()
    except Exception:  # noqa: BLE001 - no audio output must not break a reminder
        log.debug("reminder chime failed", exc_info=True)


class Scheduler:
    def __init__(self, reminders: ReminderStore, announce: Callable[[str], Awaitable[None]],
                 clock: Callable[[], datetime] = datetime.now, interval: float = 1.0,
                 hold_missed: Callable[[], bool] | None = None) -> None:
        self.store = reminders
        self.announce = announce
        self.clock = clock
        self.interval = interval
        self.hold_missed = hold_missed
        self.held: list[dict[str, Any]] = []  # missed items waiting for the morning briefing
        self._tasks: set[asyncio.Task] = set()

    def take_held(self) -> list[dict[str, Any]]:
        held, self.held = self.held, []
        return held

    def _spawn(self, text: str) -> None:
        task = asyncio.ensure_future(self._say(text))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _say(self, text: str) -> None:
        try:
            await self.announce(text)
        except Exception:  # noqa: BLE001
            log.exception("could not announce %r", text)

    async def drain(self) -> None:
        """Wait for announcements in flight (tests, shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def catch_up(self) -> None:
        """Items that came due while Jarvis was off: one combined announcement (or held for the briefing)."""
        missed = self.store.pop_due(self.clock())
        if not missed:
            return
        log.info("%d reminder(s) came due while away", len(missed))
        if self.hold_missed is not None and self.hold_missed():
            self.held += missed
        else:
            self._spawn(missed_text(missed))

    async def tick(self) -> None:
        for item in self.store.pop_due(self.clock()):
            log.info("reminder due: %s", describe_item(item))
            self._spawn(due_text(item))

    async def run(self) -> None:
        try:
            await self.catch_up()
        except Exception:  # noqa: BLE001
            log.exception("catching up on missed reminders failed")
        while True:
            try:
                await self.tick()
            except Exception:  # noqa: BLE001 - a bad tick must not stop the scheduler
                log.exception("reminder tick failed")
            await asyncio.sleep(self.interval)


_stores: dict[Path, ReminderStore] = {}
_active: Scheduler | None = None  # the running scheduler (its held items go to the briefing)


def set_active(scheduler: Scheduler | None) -> None:
    global _active
    _active = scheduler


def restore_held(items: list[dict[str, Any]]) -> None:
    if _active is not None:
        _active.held[:0] = items


def take_held() -> list[dict[str, Any]]:
    """Missed reminders waiting for the briefing (empty if no scheduler runs)."""
    return _active.take_held() if _active is not None else []


def default_store() -> ReminderStore:
    path = store.app_dir() / "reminders.json"
    return _stores.setdefault(path, ReminderStore(path))
