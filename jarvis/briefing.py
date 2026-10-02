"""Morning briefing: greeting, date, weather (Open-Meteo), today's reminders, Claude sessions, missed items.

`compose()` is pure (everything passed in) and `build()` gathers the data. The automatic briefing speaks on the
first wake of the day after `after_hour`; the day is remembered in %APPDATA%/Jarvis/state.json so restarts do
not repeat it.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable

from . import claude_sessions as cs
from . import reminders as rem
from . import store
from .config import BriefingConfig, Config
from .memory import default_memory
from .timeparse import format_clock

log = logging.getLogger("jarvis.briefing")

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT = 3.0
CLAUDE_WINDOW = 24 * 3600.0  # sessions with activity in the last 24 hours
MAX_CLAUDE_LINES = 2

WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
       51: "drizzly", 53: "drizzly", 55: "drizzly", 56: "freezing drizzle", 57: "freezing drizzle",
       61: "lightly raining", 63: "raining", 65: "raining heavily", 66: "freezing rain", 67: "freezing rain",
       71: "lightly snowing", 73: "snowing", 75: "snowing heavily", 77: "snowing", 80: "showery",
       81: "showery", 82: "very showery", 85: "snow showers", 86: "heavy snow showers", 95: "stormy",
       96: "stormy with hail", 99: "stormy with hail"}


def weather_words(code: int | None) -> str:
    """WMO weather code to a phrase ('' if unknown)."""
    try:
        return WMO.get(int(code), "")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""


def _degrees(value: float) -> str:
    n = int(round(value))
    return f"minus {abs(n)}" if n < 0 else str(n)


# ---- weather --------------------------------------------------------------------------------------------
async def fetch_weather(cfg: BriefingConfig, timeout: float = WEATHER_TIMEOUT) -> dict[str, Any] | None:
    """Today's weather from Open-Meteo, or None on any failure (the briefing just skips it)."""
    params = {"latitude": cfg.latitude, "longitude": cfg.longitude, "timezone": "auto",
              "current": "temperature_2m,weather_code",
              "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max"}
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await asyncio.wait_for(client.get(WEATHER_URL, params=params), timeout + 1)
            resp.raise_for_status()
            return parse_weather(resp.json())
    except Exception:  # noqa: BLE001 - offline, slow, bad JSON: skip silently
        log.info("weather unavailable")
        log.debug("weather error", exc_info=True)
        return None


def parse_weather(data: dict[str, Any]) -> dict[str, Any] | None:
    try:
        cur, daily = data["current"], data["daily"]
        out = {"temp": float(cur["temperature_2m"]), "code": cur.get("weather_code"),
               "hi": float(daily["temperature_2m_max"][0]), "lo": float(daily["temperature_2m_min"][0])}
        rain = daily.get("precipitation_probability_max") or [None]
        out["rain"] = rain[0]
        return out
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def weather_sentence(city: str, w: dict[str, Any]) -> str:
    words = weather_words(w.get("code"))
    now_part = f"{_degrees(w['temp'])} degrees" + (f" and {words}" if words else "")
    s = f"In {city} it's {now_part} right now, with a high of {_degrees(w['hi'])} and a low of {_degrees(w['lo'])}"
    rain = w.get("rain")
    if isinstance(rain, (int, float)) and rain >= 30:
        s += f", and a {int(round(rain))} percent chance of rain"
    return s + "."


# ---- pieces ---------------------------------------------------------------------------------------------
def greeting(now: datetime, name: str = "") -> str:
    part = "morning" if now.hour < 12 else "afternoon" if now.hour < 18 else "evening"
    return f"Good {part}, {name}." if name else f"Good {part}."


def date_sentence(now: datetime) -> str:
    d = now.day
    suffix = "th" if 10 <= d % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
    return f"It's {now.strftime('%A')}, the {d}{suffix} of {now.strftime('%B')}."


def reminders_sentence(items: list[dict[str, Any]], now: datetime) -> str:
    """Today's reminders still to come (timers do not count)."""
    today = [i for i in items if i["kind"] == "reminder" and now.timestamp() <= i["due"]
             and datetime.fromtimestamp(i["due"]).date() == now.date()]
    if not today:
        return ""
    parts = [f"{i['text']} at {format_clock(datetime.fromtimestamp(i['due']))}" for i in today[:3]]
    lead = "You have one reminder today" if len(today) == 1 else f"You have {len(today)} reminders today"
    more = f", and {len(today) - 3} more" if len(today) > 3 else ""
    return f"{lead}: {', '.join(parts[:-1]) + ' and ' + parts[-1] if len(parts) > 1 else parts[0]}{more}."


def _when(dt: datetime, now: datetime) -> str:
    if dt.date() == now.date():
        return "overnight" if dt.hour < 5 else "earlier today"
    return "last night" if dt.hour >= 19 else "yesterday"


def claude_sentences(sessions: list[cs.Session], now: datetime, pending: dict[str, Any] | None = None) -> list[str]:
    """What Claude Code did lately: waiting for you, still working, or finished since yesterday."""
    t = now.timestamp()
    out: list[str] = []
    for p in (pending or {}).values():
        out.append(f"Claude is waiting for your permission in {getattr(p, 'project', 'a project')}.")
    finished = []
    for s in sessions:
        if t - s.last_activity > CLAUDE_WINDOW or s.last_activity > t + 60:
            continue
        if cs.looks_busy(s, t):
            out.append(f"Claude is still working in {s.project_name}.")
        elif s.last_kind == "assistant_text":
            finished.append(s)
    finished.sort(key=lambda s: -s.last_activity)
    for s in finished[:MAX_CLAUDE_LINES]:
        label = cs.spoken_title(s.label, 40).rstrip(".")
        out.append(f"Claude finished work on {label} in {s.project_name} "
                   f"{_when(datetime.fromtimestamp(s.last_activity), now)}.")
    extra = len(finished) - MAX_CLAUDE_LINES
    if extra > 0:
        out.append(f"{extra} other Claude session{'s' if extra > 1 else ''} had activity too.")
    return out


def compose(now: datetime, cfg: BriefingConfig, *, name: str = "", weather: dict[str, Any] | None = None,
            reminders: list[dict[str, Any]] | None = None, sessions: list[cs.Session] | None = None,
            pending: dict[str, Any] | None = None, missed: list[dict[str, Any]] | None = None) -> str:
    parts = [f"{greeting(now, name)} {date_sentence(now)}"]
    if cfg.include_weather and weather:
        parts.append(weather_sentence(cfg.city, weather))
    if cfg.include_reminders:
        if missed:
            parts.append(rem.missed_text(missed).replace("Sir, while", "While"))
        line = reminders_sentence(reminders or [], now)
        if line:
            parts.append(line)
    if cfg.include_claude:
        parts += claude_sentences(sessions or [], now, pending)
    return " ".join(parts)


# ---- gathering ------------------------------------------------------------------------------------------
async def build(cfg: Config, now: datetime | None = None, *, held: list[dict[str, Any]] | None = None,
                fetch: Callable[[BriefingConfig], Awaitable[dict[str, Any] | None]] = fetch_weather,
                sessions_fn: Callable[[], list[cs.Session]] = cs.scan_sessions,
                reminder_store: rem.ReminderStore | None = None, name: str | None = None,
                pending: dict[str, Any] | None = None) -> str:
    """Collect everything (weather and the session scan in parallel) and compose the briefing."""
    b = cfg.briefing
    now = now or datetime.now()
    if pending is None:
        from .claude_watch import watch

        pending = watch.pending

    async def no_weather() -> None:
        return None

    async def no_sessions() -> list[cs.Session]:
        return []

    async def safe_sessions() -> list[cs.Session]:
        try:
            return await asyncio.to_thread(sessions_fn)
        except Exception:  # noqa: BLE001
            log.warning("session scan failed", exc_info=True)
            return []

    async def safe_weather() -> dict[str, Any] | None:
        try:
            return await fetch(b)
        except Exception:  # noqa: BLE001
            return None

    weather, sessions = await asyncio.gather(safe_weather() if b.include_weather else no_weather(),
                                             safe_sessions() if b.include_claude else no_sessions())
    mem_name = name if name is not None else _safe_name()
    items = (reminder_store or rem.default_store()).pending()
    return compose(now, b, name=mem_name, weather=weather, reminders=items, sessions=sessions,
                   pending=pending, missed=held)


def _safe_name() -> str:
    try:
        return default_memory().name()
    except Exception:  # noqa: BLE001
        return ""


# ---- first wake of the day ------------------------------------------------------------------------------
def due_today(cfg: BriefingConfig, now: datetime, state: dict[str, Any] | None = None) -> bool:
    """True if the automatic briefing has not run yet today and it is after `after_hour`."""
    if not (cfg.enabled and cfg.auto_first_wake) or now.hour < cfg.after_hour:
        return False
    state = store.load_state() if state is None else state
    return state.get("last_briefing") != now.date().isoformat()


def mark_done(cfg: BriefingConfig, now: datetime) -> None:
    """Remember that today's briefing was given (only counts after `after_hour`)."""
    if now.hour >= cfg.after_hour:
        store.update_state(last_briefing=now.date().isoformat())


__all__ = ["build", "compose", "due_today", "mark_done", "weather_words", "fetch_weather"]
