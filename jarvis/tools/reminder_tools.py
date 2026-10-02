"""Timers and reminders by voice. They persist in %APPDATA%/Jarvis/reminders.json (see reminders.py)."""

from __future__ import annotations

from datetime import datetime

from .. import reminders as rem
from ..timeparse import format_when, parse_when, spoken_duration
from .registry import ToolError, tool


def _now() -> datetime:
    return datetime.now()


@tool("Start a countdown timer that announces itself when done. Use for 'set a timer for 10 minutes'.")
def set_timer(minutes: float = 0, seconds: float = 0, label: str = "") -> str:
    """Start a timer.

    Args:
        minutes: minutes (can be fractional); 1 hour = 60
        seconds: extra seconds
        label: optional name, e.g. "pasta"
    """
    total = float(minutes or 0) * 60 + float(seconds or 0)
    if total <= 0:
        raise ToolError("How long should the timer run?")
    if total > 7 * 86400:
        raise ToolError("That is too long for a timer; set a reminder instead.")
    now = _now()
    rem.default_store().add_timer(total, label, now)
    name = f"{label.strip().capitalize()} timer" if label.strip() else "Timer"
    return f"{name} set for {spoken_duration(total)}"


@tool("Set a reminder that is announced out loud at the given time. 'when' is natural: 'in 20 minutes', "
      "'at 6pm', 'tomorrow at 9', 'friday at 18:30'.")
def set_reminder(text: str, when: str) -> str:
    """Set a reminder.

    Args:
        text: what to remind the user of, without "remind me", e.g. "call mom"
        when: when to announce it, e.g. "in 20 minutes", "at 6pm", "tomorrow at 9", "friday at 3pm"
    """
    if not text.strip():
        raise ToolError("What should I remind you of?")
    now = _now()
    due = parse_when(when, now)
    if due is None:
        raise ToolError(f"I couldn't understand the time '{when.strip()}'. Ask the user when, e.g. 'in 20 minutes' or 'at 6pm'.")
    rem.default_store().add_reminder(text, due, now)
    return f"Reminder set to {text.strip()} {format_when(due, now)}"


@tool("List the pending reminders and running timers.")
def list_reminders() -> str:
    now = _now()
    items = rem.default_store().pending()
    if not items:
        return "You have no reminders or timers"
    lines = [rem.describe_pending(i, now) for i in items]
    n = len(lines)
    head = "You have one" if n == 1 else f"You have {n}"
    return f"{head}: " + "; ".join(lines)


@tool("Cancel a reminder or timer. Query: words from it ('the timer', 'call mom', '10 minute'), or 'all'.")
def cancel_reminder(query: str = "") -> str:
    """Cancel pending reminders or timers.

    Args:
        query: which one: "timer", "reminder about X", a few words from it, or "all"
    """
    try:
        gone = rem.default_store().cancel(query)
    except rem.Ambiguous as exc:
        names = "; ".join(rem.describe_item(i) for i in exc.items)
        raise ToolError(f"Several match: {names}. Ask the user which one.")
    if not gone:
        return "I found no matching reminder or timer"
    if len(gone) == 1:
        return f"Cancelled the {rem.describe_item(gone[0])}"
    return f"Cancelled {len(gone)} items"
