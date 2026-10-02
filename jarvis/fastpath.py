"""Fast paths: answer trivial, unambiguous requests without the LLM.

Pure matching only (no I/O). Every pattern is anchored on the whole normalized utterance, so
anything with extra words ("what time is it in Tokyo", "open chrome and search cats") returns
None and goes to the agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime

from .timeparse import WEEKDAYS, parse_duration, parse_when
from .tools.apps import APP_MAP, SPECIAL
from .tools.windows import _EXTRA_EXE


@dataclass(frozen=True)
class FastPath:
    kind: str                       # time | date | day | open_app | lock | stop | volume | window | music | search
    reply: str                      # spoken reply ("" = say nothing)
    # for the caller to run: ("open_app", name) | ("lock_pc",) | ("volume", up|down|mute)
    # | ("call", tool_name, json_args) = run that registry tool
    action: tuple[str, ...] = ()
    stop_speaking: bool = False
    speak_result: bool = False      # speak the action's result text instead of `reply` (reply = fallback)


_LEAD = re.compile(r"^(?:(?:hey|ok|okay)\s+)?jarvis\b\s*")
_TRAIL = re.compile(r"\s*\b(?:please|sir|jarvis|thanks|thank you)$")
_POLITE = r"(?:(?:can|could|would|will) you\s+)?(?:please\s+)?"


def normalize(text: str) -> str:
    """Lowercase, drop punctuation (keep apostrophes), collapse spaces, strip jarvis/please."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _LEAD.sub("", t)
    for _ in range(3):  # "... please sir"
        t2 = _TRAIL.sub("", t).strip()
        if t2 == t:
            break
        t = t2
    return t


def _re(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"^(?:{pattern})$")


_NOW = r"(?: (?:right )?now)?"
_TIME = _re(_POLITE + r"(?:tell me |give me )?(?:what(?:'s| is|s) )?the (?:current )?time" + _NOW
            + r"|" + _POLITE + r"what time is it" + _NOW
            + r"|current time")
_DAY = _re(_POLITE + r"(?:what day is it(?: today)?|what(?:'s| is|s) the day(?: today)?|what(?:'s| is|s) today|"
           r"what day of the week is it(?: today)?)")
_DATE = _re(_POLITE + r"(?:what(?:'s| is|s) (?:the |today's |todays )(?:date|current date)(?: today)?|"
            r"what is today's date|what(?:'s| is|s) the date|tell me the date|what date is it(?: today)?)")
_OPEN = _re(_POLITE + r"(?:open|launch)(?: up)? (?:the )?(.+?)(?: app| application)?")
_LOCK = _re(_POLITE + r"lock (?:the |my )?(?:pc|computer)")
_STOP = _re(r"(?:stop(?: talking| speaking)?|never ?mind|cancel(?: that)?|forget it|be quiet)")
_VOL_UP = _re(_POLITE + r"(?:turn |put |bring )?(?:the )?(?:volume|sound) up|" + _POLITE + r"(?:raise|increase) the volume")
_VOL_DOWN = _re(_POLITE + r"(?:turn |put |bring )?(?:the )?(?:volume|sound) down|"
                + _POLITE + r"(?:lower|reduce|decrease) the volume")
_VOL_MUTE = _re(_POLITE + r"(?:un ?mute|mute)(?: the)?(?: (?:volume|sound|pc|computer))?|(?:volume|sound) mute")


_WIN_APP = r"(?:the )?(.+?)(?: window| windows| app| application)?"
_MINIMIZE = _re(_POLITE + r"minimi[sz]e " + _WIN_APP)
_MAXIMIZE = _re(_POLITE + r"maximi[sz]e " + _WIN_APP)
_MIN_ALL = _re(_POLITE + r"(?:minimi[sz]e (?:everything|all(?: (?:the |of the )?windows)?)|"
               r"show (?:me )?(?:the |my )?desktop|go to (?:the |my )?desktop)")
_M_NOUN = r"(?:the |my |this |that )?(?:music|song|track|playback|spotify)"
_M_PAUSE = _re(_POLITE + r"pause(?: " + _M_NOUN + r")?")
_M_PLAY = _re(_POLITE + r"(?:(?:play|resume|unpause|continue)(?: playing)? " + _M_NOUN + r"|resume|unpause)")
_M_NEXT = _re(_POLITE + r"(?:(?:skip|next)(?: (?:the |this |that |to the next )?(?:song|track))?|"
              r"play the next (?:song|track)|skip(?: to)? (?:the )?next (?:song|track))")
_M_PREV = _re(_POLITE + r"(?:(?:previous|last) (?:song|track)|go back(?: a| one)? (?:song|track)|"
              r"play the (?:previous|last) (?:song|track)|previous)")
_M_STOP = _re(_POLITE + r"stop (?:the |my )?(?:music|song|track|playback|spotify)")
_M_WHAT = _re(r"(?:what(?:'s| is|s) (?:currently |now )?playing(?: right now| now)?|"
              r"what (?:song|track) is (?:this|playing|that)(?: right now| now)?|what am i listening to)")
_PLAY_SPOTIFY = _re(_POLITE + r"play (.+?) (?:on|in|with|using) spotify")
# Web search only when the user clearly means the web. A plain "search for X" is ambiguous
# (web or PC?) and goes to the model, which asks.
_WEB = r"(?:in|on|using|with) (?:google )?chrome|on google|online|on the (?:web|internet)"
_SEARCH = re.compile(
    rf"^{_POLITE}(?:google (?P<q1>.+?)(?: (?:{_WEB}))?"
    rf"|(?:search|look) (?:the web|online|google|the internet) for (?P<q2>.+?)"
    rf"|(?:search(?: for)?|look up) (?P<q3>.+?) (?:{_WEB}))$")
# "search for X on my pc", "find X in my documents" -> local file search
_PLACES = {"pc": "~", "computer": "~", "laptop": "~", "files": "~", "drive": "~", "desktop": "desktop",
           "documents": "documents", "downloads": "downloads", "pictures": "pictures", "music": "music",
           "videos": "videos"}
_FIND_LOCAL = re.compile(
    rf"^{_POLITE}(?:search(?: for)?|find|look for|locate) (?:a |the |my )?(?:(?:file|folder) (?:called |named )?)?"
    rf"(?P<q>.+?) (?:on|in) (?:my |the )?(?P<place>{'|'.join(_PLACES)})$")
_OPEN_WHAT = re.compile(r"^(?:open|open it|open up|open the|open a|open my|open that|launch|start)$")
# "what did Claude say", "what's Claude doing in the jarvis project", "is Claude done"
_CLAUDE_STATUS = re.compile(
    r"^(?:(?:what|whats|what's|what has|what did|what is)\b.*\bclaude\b.*\b(?:say|said|saying|answer|answered|"
    r"reply|replied|respond|responded|write|wrote|last message|doing|up to|done)"
    r"|(?:is|did|has) claude (?:done|finish|finished|finish(?:ed)? (?:yet|working)|done yet)"
    r"|(?:tell me |read me )?(?:what )?claude(?:'s| last)? (?:said|answer|reply|last message))"
    r"(?P<rest>.*)$")
_PROJECT_IN = re.compile(r"\b(?:in|on|for|from|with) (?:the |my )?(?P<project>[a-z0-9][a-z0-9 ._-]*?)(?: project| session| folder)?$")

_SEARCH_OTHER_SITE = re.compile(r" (?:on|in|at) (?:youtube|amazon|ebay|reddit|github|wikipedia|twitter|netflix|"
                                r"spotify|maps|google maps|facebook|instagram|linkedin|bing)$")


# ---- memory, timers and reminders, briefing -------------------------------------------------------------
def _normalize_time(text: str) -> str:
    """Like normalize() but keeps '6:30' and '1.5' intact and turns '10-minute' into '10 minute'."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"\b([ap])\.m\.?", r"\1m", t).replace("-", " ")
    t = re.sub(r"[^a-z0-9':. ]+", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    t = re.sub(r"(?<!\d):|:(?!\d)", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _LEAD.sub("", t)
    for _ in range(3):
        t2 = _TRAIL.sub("", t).strip()
        if t2 == t:
            break
        t = t2
    return t


_TIMER_FOR = re.compile(rf"^{_POLITE}(?:set|start|create|make|begin)(?: me)?(?: a| an| the)? timer (?:for|of) (?P<d>.+)$")
_TIMER_ADJ = re.compile(rf"^{_POLITE}(?:set|start|create|make)(?: me)?(?: a| an)? (?P<d>.+?) timer$")
_TIMER_BARE = re.compile(r"^timer (?:for )?(?P<d>.+)$")
_REMIND_ME = re.compile(rf"^{_POLITE}remind me (?P<rest>.+)$")
_SET_REMINDER = re.compile(rf"^{_POLITE}(?:set|create|add|make)(?: me)?(?: a| an)? reminder (?P<rest>(?:for|to|at|in|on) .+)$")
_REM_LIST = _re(
    r"(?:what|which) (?:reminders|timers)(?: and (?:reminders|timers))? (?:do i have|have i got|are (?:set|there|active|running))"
    r"(?: set| right now| today)?"
    r"|(?:list|show|read)(?: me| out)?(?: all)?(?: of)?(?: my| the)? (?:reminders|timers)"
    r"|do i have any (?:reminders|timers)(?: set| today)?"
    r"|how (?:much time|long) (?:is )?left(?: on (?:the|my) timer)?"
    r"|how long (?:until|till) (?:my|the) timer(?: is done| goes off| ends)?")
_REM_CANCEL = re.compile(
    rf"^{_POLITE}(?:cancel|stop|delete|remove|clear|turn off)(?P<all> all| every)?(?: of)?(?: the| my)?"
    r"(?: (?P<q>.+?))? (?P<kind>timers?|reminders?)(?: (?:for|about|to) (?P<q2>.+))?$")
_BRIEFING = _re(
    r"(?:(?:give me|tell me|read me|read out|play|do|run)? ?(?:my |the )?(?:morning |daily )?briefing"
    r"|what(?:'s| is|s) my (?:morning )?briefing|brief me|what(?:'s| is|s) my day(?: like| looking like| going to be like)?"
    r"|what does my day look like)")
_RECALL = _re(
    r"what do you (?:remember|know) about me|what do you remember(?: about me)?|what have i told you(?: about me)?"
    r"|(?:tell|show|list) me (?:what you remember|everything you remember|my memories)(?: about me)?"
    r"|what(?:'s| is) in your memory|what do you remember about (?P<q>.+)")
_FORGET = re.compile(rf"^{_POLITE}forget(?: about| that)? (?P<q>.+)$")
_FORGET_SKIP = {"it", "that", "this", "everything", "all", "about it", "that i said", "what i said", "what i told you"}
_REMEMBER = re.compile(r"^(?:(?:can|could|would|will) you )?(?:please )?remember(?: that)? (?P<f>.+)$", re.IGNORECASE)
_REMEMBER_SKIP = re.compile(r"^(?:to|what|when|where|how|why|who|which|if|me|us|the way|whether|i said)\b", re.IGNORECASE)
_WHEN_AT = re.compile(rf"(?<= )(?=(?:in|at|on|after|tomorrow|tonight|today|this|next|\d|{'|'.join(WEEKDAYS)})\b)")


def _clean_original(text: str) -> str:
    """The utterance with its case kept, minus the 'Jarvis' lead and trailing punctuation/politeness."""
    t = re.sub(r"^\s*(?:(?:hey|ok|okay)\s+)?jarvis\b[\s,.:!-]*", "", text.strip(), flags=re.IGNORECASE)
    t = t.strip().rstrip(" .!?,;")
    for _ in range(3):
        t = re.sub(r"(?:[\s,]+(?:please|thanks|thank you)|\s*,\s*(?:sir|jarvis))$", "", t,
                   flags=re.IGNORECASE).rstrip(" .!?,;")
    return t


def _timer_action(seconds: float) -> tuple[str, ...]:
    return _call("set_timer", minutes=int(seconds // 60), seconds=round(seconds % 60, 3) if seconds % 1 else int(seconds % 60))


def _match_timer(t: str) -> FastPath | None:
    for rx in (_TIMER_FOR, _TIMER_ADJ, _TIMER_BARE):
        m = rx.match(t)
        if m:
            secs = parse_duration(m.group("d"))
            if secs is not None:
                return FastPath("timer", "Setting the timer, sir.", _timer_action(secs), speak_result=True)
            return None
    return None


def _split_reminder(rest: str, now: datetime) -> tuple[str, str] | None:
    """('call mom', 'at 6pm') from 'to call mom at 6pm' / 'in 20 minutes to call mom' (None if no clear time)."""
    m = re.match(r"^(?:to|that|about) (?P<body>.+)$", rest)
    if m:  # text first, time last: try the earliest split whose tail is purely a time
        body = m.group("body")
        for pos in sorted({x.start() for x in _WHEN_AT.finditer(body)}):
            text, when = body[:pos].strip(), body[pos:].strip()
            if text and parse_when(when, now, strict=True) is not None:
                return text, when
        return None
    for sep in re.finditer(r" (?:to|that|about) ", rest):  # time first
        when, text = rest[:sep.start()].strip(), rest[sep.end():].strip()
        if when and text and parse_when(when, now, strict=True) is not None:
            return text, when
    return None


def _match_reminder(t: str, now: datetime) -> FastPath | None:
    m = _REMIND_ME.match(t) or _SET_REMINDER.match(t)
    if not m:
        return None
    rest = re.sub(r"^for ", "", m.group("rest"))
    parts = _split_reminder(rest, now)
    if parts is None:
        return None
    text, when = parts
    return FastPath("reminder", "Setting the reminder, sir.", _call("set_reminder", text=text, when=when),
                    speak_result=True)


def _match_assistant(text: str, now: datetime) -> FastPath | None:
    """Memory, timers, reminders and the briefing."""
    t = _normalize_time(text)
    if not t:
        return None
    m = _REMEMBER.match(_clean_original(text))
    if m:
        fact = m.group("f").strip()
        if fact and not _REMEMBER_SKIP.match(fact):
            return FastPath("remember", "Noted, sir.", _call("remember", fact=fact), speak_result=True)
        return None
    m = _RECALL.match(t)
    if m:
        return FastPath("recall", "Let me think, sir.", _call("recall", query=(m.groupdict().get("q") or "").strip()),
                        speak_result=True)
    m = _FORGET.match(t)
    if m:
        q = m.group("q").strip()
        if q in _FORGET_SKIP:
            return None
        return FastPath("forget", "Forgotten, sir.", _call("forget", query=q), speak_result=True)
    fp = _match_timer(t)
    if fp is not None:
        return fp
    fp = _match_reminder(t, now)
    if fp is not None:
        return fp
    if _REM_LIST.match(t):
        return FastPath("reminders", "Let me check, sir.", _call("list_reminders"), speak_result=True)
    m = _REM_CANCEL.match(t)
    if m:
        parts = [(m.group("all") or "").strip(), m.group("q") or "", m.group("kind"), m.group("q2") or ""]
        query = " ".join(x for x in parts if x)
        return FastPath("reminders", "Cancelling, sir.", _call("cancel_reminder", query=query), speak_result=True)
    if _BRIEFING.match(t):
        return FastPath("briefing", "One moment, sir.", _call("briefing"), speak_result=True)
    return None


def _known_window_app(name: str) -> bool:
    return name in APP_MAP or name in SPECIAL or name in _EXTRA_EXE


def _call(tool: str, **args: str) -> tuple[str, ...]:
    return ("call", tool, json.dumps(args))


def format_time(now: datetime) -> str:
    return now.strftime("%I:%M %p").lstrip("0")


def format_date(now: datetime) -> str:
    return f"{now.strftime('%A')}, {now.day} {now.strftime('%B %Y')}"


def match(text: str, now: datetime | None = None) -> FastPath | None:
    """Return a FastPath if the whole utterance is a known trivial command, else None."""
    t = normalize(text)
    if not t:
        return None
    now = now or datetime.now()
    if _STOP.match(t):
        return FastPath("stop", "", stop_speaking=True)
    if _TIME.match(t):
        return FastPath("time", f"It's {format_time(now)}, sir.")
    if _DAY.match(t):
        return FastPath("day", f"It's {now.strftime('%A')}, sir.")
    if _DATE.match(t):
        return FastPath("date", f"Today is {format_date(now)}, sir.")
    fp = _match_assistant(text, now)
    if fp is not None:
        return fp
    if _LOCK.match(t):
        return FastPath("lock", "Locking the PC, sir.", ("lock_pc",))
    if _MIN_ALL.match(t):
        return FastPath("window", "Clearing the desktop, sir.", _call("minimize_all"))
    for rx, verb, past in ((_MINIMIZE, "minimize", "Minimising"), (_MAXIMIZE, "maximize", "Maximising")):
        m = rx.match(t)
        if m:
            app = m.group(1).strip()
            if not _known_window_app(app):
                return None
            return FastPath("window", f"{past} {app.title()}, sir.", _call("window_action", app=app, action=verb))
    # "close X" is deliberately not a fast path: it needs approval, which the agent loop handles.
    for rx, act, reply in ((_M_PAUSE, "play_pause", "Pausing, sir."), (_M_PLAY, "play_pause", "Resuming, sir."),
                           (_M_NEXT, "next", "Next track, sir."), (_M_PREV, "previous", "Previous track, sir."),
                           (_M_STOP, "stop", "Stopping the music, sir.")):
        if rx.match(t):
            return FastPath("music", reply, _call("music_control", action=act))
    if _M_WHAT.match(t):
        return FastPath("music", "Let me see, sir.", _call("spotify_now_playing"), speak_result=True)
    m = _PLAY_SPOTIFY.match(t)
    if m:
        song = m.group(1).strip()
        return FastPath("music", f"Playing {song} on Spotify, sir.", _call("spotify_play", query=song),
                        speak_result=True)
    if _OPEN_WHAT.match(t):
        return FastPath("clarify", "Open what, sir?")
    m = _CLAUDE_STATUS.match(t)
    if m:
        pm = _PROJECT_IN.search(m.group("rest") or "")
        project = pm.group("project").strip() if pm else ""
        return FastPath("claude_status", "Let me check, sir.", _call("claude_status", project=project),
                        speak_result=True)
    m = _FIND_LOCAL.match(t)
    if m:
        q, place = m.group("q").strip(), m.group("place")
        if q and q not in ("it", "that", "this"):
            return FastPath("find_local", f"Looking for {q}.", _call("find_on_pc", name=q, where=_PLACES[place]),
                            speak_result=True)
    m = _SEARCH.match(t)
    if m:
        q = (m.group("q1") or m.group("q2") or m.group("q3") or "").strip()
        if q and q not in ("chrome", "it", "that") and not _SEARCH_OTHER_SITE.search(q):
            return FastPath("search", f"Searching the web for {q}.", _call("open_chrome", search=q))
        return None
    m = _OPEN.match(t)
    if m:
        name = m.group(1).strip()
        if name in APP_MAP or name in SPECIAL:
            return FastPath("open_app", f"Opening {name.title()}, sir.", ("open_app", name))
        return None
    if _VOL_UP.match(t):
        return FastPath("volume", "Volume up, sir.", ("volume", "up"))
    if _VOL_DOWN.match(t):
        return FastPath("volume", "Volume down, sir.", ("volume", "down"))
    if _VOL_MUTE.match(t):
        return FastPath("volume", "Toggling mute, sir.", ("volume", "mute"))
    return None
