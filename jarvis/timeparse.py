"""Natural time phrases to datetimes. Pure: everything is relative to the `now` you pass in.

parse_duration("ten minutes and 30 seconds") -> 630.0
parse_when("tomorrow at 9", now)            -> datetime (naive, local)
Both return None when the text is not understood.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_NUM_WORD = "|".join(sorted([*_UNITS, *_TENS], key=len, reverse=True))
_NUM_RE = re.compile(rf"\b(?:({'|'.join(_TENS)})(?:[ -]({'|'.join(list(_UNITS)[1:10])}))?|({_NUM_WORD}))\b")
_UNIT_NAMES = r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)"
_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_PART_HOUR = {"morning": 8, "afternoon": 15, "evening": 18, "night": 21, "tonight": 21}
_WEEKDAY_RE = "|".join(WEEKDAYS)


def _numberize(text: str) -> str:
    """'twenty five' -> '25', 'a'/'an' before a unit -> '1', 'half' phrases -> numbers."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"\bhalf an hour\b|\ba half hour\b|\bhalf hour\b", "30 minutes", t)
    t = re.sub(r"\b(?:a )?quarter of an hour\b|\ba quarter hour\b", "15 minutes", t)
    t = re.sub(r"\ban hour and a half\b|\b1 hour and a half\b", "90 minutes", t)

    def num(m: re.Match[str]) -> str:
        if m.group(1):
            return str(_TENS[m.group(1)] + (_UNITS[m.group(2)] if m.group(2) else 0))
        return str(_UNITS.get(m.group(3), _TENS.get(m.group(3))))

    t = _NUM_RE.sub(num, t)
    t = re.sub(rf"\b(\d+) and a half ({_UNIT_NAMES})", r"\1.5 \2", t)
    t = re.sub(rf"\b(?:a|an) (?={_UNIT_NAMES}\b)", "1 ", t)
    return t


def parse_duration(text: str) -> float | None:
    """Total seconds in phrases like '10 minutes', '1 hour 30 minutes', 'half an hour', '90 seconds'."""
    t = _numberize(text)
    total = 0.0
    found = False
    for m in re.finditer(rf"(\d+(?:\.\d+)?)\s*({_UNIT_NAMES})\b", t):
        total += float(m.group(1)) * _SECONDS[m.group(2)[0]]
        found = True
    if not found:
        return None
    # nothing but filler may remain, so "5 minutes of silence" or a clock time is not a duration
    rest = re.sub(rf"(\d+(?:\.\d+)?)\s*{_UNIT_NAMES}\b", " ", t)
    rest = re.sub(r"\b(?:in|for|of|after|within|and|about|around|a|the|timer|time|from now|me|set|a timer)\b", " ", rest)
    if re.search(r"[a-z0-9]", rest):
        return None
    return total if total > 0 else None


def _clock(t: str) -> tuple[int, int, str] | None:
    """(hour, minute, meridiem) from the text, meridiem '' | 'am' | 'pm' | '24' (explicitly 24 hour)."""
    if re.search(r"\bnoon\b", t):
        return 12, 0, "24"
    if re.search(r"\bmidnight\b", t):
        return 0, 0, "24"
    m = re.search(r"\b(half|quarter) (past|to) (\d{1,2})\b", t)
    if m:
        hour = int(m.group(3))
        mins = {"half": 30, "quarter": 15}[m.group(1)]
        if m.group(2) == "to":
            hour, mins = hour - 1, 60 - mins
        if 0 <= hour <= 12:
            return hour or 12, mins, ""
        return None
    m = re.search(r"(?:\bat |@ ?|\b)(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])", t)
    if m:
        hour, mins, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)[0] + "m"
        return (hour, mins, mer) if 1 <= hour <= 12 and mins < 60 else None
    m = re.search(r"\bat (\d{1,2}):(\d{2})\b|\b(\d{1,2}):(\d{2})\b", t)
    if m:
        hour = int(m.group(1) or m.group(3))
        mins = int(m.group(2) or m.group(4))
        if hour > 23 or mins > 59:
            return None
        if hour == 0 or hour > 12 or (m.group(1) or m.group(3)).startswith("0"):
            return hour, mins, "24"
        return hour, mins, ""
    m = re.search(r"\bat (\d{1,2})(?: o'?clock)?(?!\d|:)", t)
    if m:
        hour = int(m.group(1))
        if hour > 23:
            return None
        return (hour, 0, "24") if hour == 0 or hour > 12 else (hour, 0, "")
    return None


def _at(day: datetime, hour: int, minute: int) -> datetime:
    return day.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)


_TIME_WORDS = re.compile(
    rf"\b(?:in|at|on|after|from now|the|of|next|this|and|a half|half past|quarter past|quarter to|day after tomorrow|"
    rf"tomorrow|today|tonight|morning|afternoon|evening|night|noon|midnight|o'?clock|{_WEEKDAY_RE})\b")


def is_time_phrase(text: str) -> bool:
    """True if `text` is nothing but a time expression ('tomorrow at 9' yes, 'go on a walk tomorrow' no)."""
    t = _numberize(text)
    t = re.sub(rf"\d+(?:\.\d+)?\s*{_UNIT_NAMES}\b", " ", t)
    t = re.sub(r"\d{1,2}(?:[:.]\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?", " ", t)
    t = _TIME_WORDS.sub(" ", t)
    return not re.search(r"[a-z0-9]", t)


def parse_when(text: str, now: datetime, strict: bool = False) -> datetime | None:
    """A moment in the future from phrases like 'in 20 minutes', 'at 6pm', 'tomorrow at 9', 'friday at 18:30'.

    With strict=True the whole text must be a time expression (used to split "call mom at 6pm")."""
    if strict and not is_time_phrase(text):
        return None
    t = _numberize(text).strip(" .,")
    t = re.sub(r"^(?:remind me |please )+", "", t).strip()
    t = re.sub(r"\b(\d{1,2})\.(\d{2})(?=\s*[ap]\.?m)", r"\1:\2", t)  # "6.30pm"
    if not t or re.search(r"\b(?:yesterday|ago|last)\b", t):
        return None
    clock_hint = _clock(t)
    has_day_word = re.search(rf"\b(today|tonight|tomorrow|{_WEEKDAY_RE}|this (?:morning|afternoon|evening)|"
                             r"in the (?:morning|afternoon|evening))\b", t)
    # relative: "in 20 minutes", "after 2 hours", "10 minutes from now"
    if clock_hint is None and not has_day_word:
        secs = parse_duration(t)
        if secs is not None:
            return now + timedelta(seconds=secs)
        return None
    if clock_hint is None and re.search(rf"\bin \d", t) and not re.search(r"\bat\b", t):
        secs = parse_duration(re.sub(rf"\b(?:today|tomorrow|{_WEEKDAY_RE})\b", " ", t))
        if secs is not None:
            return now + timedelta(seconds=secs)

    part = next((p for p in ("tonight", "morning", "afternoon", "evening", "night") if re.search(rf"\b{p}\b", t)), "")
    day: datetime | None = None
    explicit_day = True
    if re.search(r"\bday after tomorrow\b", t):
        day = now + timedelta(days=2)
    elif re.search(r"\btomorrow\b", t):
        day = now + timedelta(days=1)
    elif re.search(r"\b(today|tonight|this (?:morning|afternoon|evening))\b", t):
        day = now
    else:
        m = re.search(rf"\b({_WEEKDAY_RE})\b", t)
        if m:
            ahead = (WEEKDAYS.index(m.group(1)) - now.weekday()) % 7
            day = now + timedelta(days=ahead)
            weekday_target = True
        else:
            explicit_day = False
    weekday_target = bool(re.search(rf"\b({_WEEKDAY_RE})\b", t)) and not re.search(r"\b(today|tomorrow)\b", t)

    if clock_hint is None:
        if day is None:
            return None
        hour = _PART_HOUR.get(part, 9)
        result = _at(day, hour, 0)
        if result <= now and weekday_target:
            result += timedelta(days=7)
        return result if result > now else None

    hour, minute, mer = clock_hint
    if mer == "am":
        hours = [hour % 12]
    elif mer == "pm":
        hours = [hour % 12 + 12]
    elif mer == "24":
        hours = [hour]
    elif part in ("afternoon", "evening", "night", "tonight"):
        hours = [hour % 12 + 12]
    elif part == "morning":
        hours = [hour % 12]
    elif explicit_day:
        hours = [hour] if hour == 12 else [hour + 12 if hour <= 6 else hour]
    else:  # bare "at 6": the next time that happens, morning or evening
        hours = [hour % 12, hour % 12 + 12]

    if explicit_day:
        assert day is not None
        result = _at(day, hours[0], minute)
        if result <= now and weekday_target:
            result += timedelta(days=7)
        return result if result > now else None
    candidates = [_at(now + timedelta(days=d), h, minute) for d in (0, 1) for h in hours]
    future = [c for c in candidates if c > now]
    return min(future) if future else None


def format_clock(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0").replace(":00 ", " ")


def format_when(dt: datetime, now: datetime) -> str:
    """Spoken: '6:30 PM today', 'tomorrow at 9 AM', 'Friday at 6 PM', 'in 20 minutes'."""
    secs = (dt - now).total_seconds()
    if secs < 90 * 60 and dt.date() == now.date():
        return "in " + spoken_duration(max(secs, 1))
    days = (dt.date() - now.date()).days
    clock = format_clock(dt)
    if days == 0:
        return f"at {clock} today"
    if days == 1:
        return f"tomorrow at {clock}"
    if days < 7:
        return f"{dt.strftime('%A')} at {clock}"
    return f"on {dt.strftime('%A')} {dt.day} {dt.strftime('%B')} at {clock}"


def spoken_duration(seconds: float) -> str:
    """90 -> '1 minute 30 seconds', 600 -> '10 minutes', 5400 -> '1 hour 30 minutes'."""
    s = int(round(seconds))
    parts = []
    for name, size in (("hour", 3600), ("minute", 60), ("second", 1)):
        n, s = divmod(s, size)
        if n:
            parts.append(f"{n} {name}{'' if n == 1 else 's'}")
    return " ".join(parts) or "0 seconds"
