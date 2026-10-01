"""Claude Code sessions learned from the local transcripts (~/.claude/projects/<encoded-cwd>/<id>.jsonl).

The transcript line format is internal to Claude Code and changes between versions, so everything here
parses defensively: every line is tried as JSON, unknown shapes are skipped, nothing raises.
Only the head (title, first message) and the tail (last messages) of each file are read, and results are
cached by (path, mtime, size).
"""

from __future__ import annotations

import difflib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import paths

HEAD_BYTES = 64 * 1024
TAIL_BYTES = 256 * 1024
MAX_SESSIONS = 300  # newest transcripts parsed per scan
BUSY_SECONDS = 20.0  # modified more recently than this and mid-turn = Claude is working
AMBIGUITY_MARGIN = 0.12
MIN_RELEVANCE = 0.3

_TITLE_KINDS = {  # event type -> (field, priority); higher priority wins
    "custom-title": ("customTitle", 3),
    "ai-title": ("aiTitle", 2),
    "summary": ("summary", 1),
    "title": ("title", 1),
}
_NOISE_PREFIXES = ("<command-name>", "<command-message>", "<command-args>", "<local-command",
                   "caveat:", "<system-reminder>", "[request interrupted", "<ide_", "<bash-", "<user-prompt")


# ---- locations ------------------------------------------------------------------------------------------
def projects_dir() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR")
    return (Path(base) if base else Path.home() / ".claude") / "projects"


def encode_project(path: str) -> str:
    """Claude Code's folder naming: every non-alphanumeric character becomes '-'."""
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def decode_project(name: str) -> str:
    """Best-effort inverse of encode_project (lossy: '-' inside folder names is read as a separator)."""
    m = re.match(r"^([A-Za-z])--(.*)$", name)
    if m:
        return f"{m.group(1)}:\\" + m.group(2).replace("-", "\\")
    if name.startswith("-"):
        return "/" + name[1:].replace("-", "/")
    return name


def basename(path: str) -> str:
    parts = [p for p in re.split(r"[\\/]", path) if p]
    return parts[-1] if parts else path


def _norm_path(path: str) -> str:
    return re.sub(r"[\\/]+", "/", path).rstrip("/").lower()


# ---- data -----------------------------------------------------------------------------------------------
@dataclass
class Session:
    id: str
    path: str
    project_dir: str
    project_name: str
    title: str = ""
    first_message: str = ""
    last_activity: float = 0.0  # file mtime
    last_assistant_text: str = ""
    last_tool: str = ""  # name of the last tool_use
    message_count: int = 0  # real user + assistant messages (extrapolated for very large files)
    last_kind: str = ""  # tool_use | tool_result | assistant_text | user_text of the last message
    last_user_ts: float | None = None  # timestamp of the last real user message (turn start)

    @property
    def label(self) -> str:
        """Short name for speaking: the title, else the start of the first message."""
        return spoken_title(self.title or self.first_message or "untitled session")


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "..."


_MD = re.compile(r"[*_`#>]+")
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def speakable(text: str) -> str:
    """Plain one-line text: markdown marks and code fences dropped, whitespace collapsed."""
    text = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return " ".join(_MD.sub("", text).split())


def first_sentences(text: str, count: int = 2, max_words: int = 40) -> str:
    """The first `count` sentences of `text` (speakable), cut to `max_words` words."""
    parts = [p for p in _SENT_END.split(speakable(text)) if p]
    out = " ".join(parts[:count])
    words = out.split()
    if len(words) > max_words:
        out = " ".join(words[:max_words]).rstrip(",;:") + "..."
    return out


def spoken_title(text: str, limit: int = 60) -> str:
    text = speakable(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0] or text[:limit]
    return cut.rstrip(",;:.") + "..."


def humanize_age(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return "just now"
    m = s // 60
    if m < 60:
        return "1 minute ago" if m == 1 else f"{m} minutes ago"
    h = m // 60
    if h < 24:
        return "1 hour ago" if h == 1 else f"{h} hours ago"
    d = h // 24
    if d == 1:
        return "yesterday"
    if d < 14:
        return f"{d} days ago"
    if d < 60:
        return f"{d // 7} weeks ago"
    mo = d // 30
    return f"{mo} months ago" if mo < 24 else "years ago"


def format_line(index: int, s: Session, now: float | None = None) -> str:
    """'1. Login bug, Jarvis, 2 hours ago'"""
    now = time.time() if now is None else now
    return f"{index}. {s.label}, {s.project_name}, {humanize_age(now - s.last_activity)}"


def format_list(sessions: list[Session], now: float | None = None) -> str:
    return "\n".join(format_line(i, s, now) for i, s in enumerate(sessions, 1))


# ---- parsing --------------------------------------------------------------------------------------------
def _blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    parts = [b.get("text", "") for b in _blocks(content) if b.get("type") == "text" and isinstance(b.get("text"), str)]
    return "\n".join(p.strip() for p in parts if p.strip())


def _is_noise(text: str) -> bool:
    low = text.lstrip().lower()
    return not low or low.startswith(_NOISE_PREFIXES)


def _parse_ts(value: Any) -> float | None:
    if isinstance(value, (int, float)) and value > 0:
        return float(value) / 1000.0 if value > 1e11 else float(value)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


@dataclass
class _Scan:
    cwd: str = ""
    titles: dict[int, str] = field(default_factory=dict)
    first_user: str = ""
    last_assistant_text: str = ""
    last_tool: str = ""
    last_kind: str = ""
    last_user_ts: float | None = None
    messages: int = 0


def _scan_lines(lines: list[str], scan: _Scan) -> None:
    """Fold transcript lines (oldest first) into `scan`. Unknown or broken lines are ignored."""
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        kind = ev.get("type")
        if kind in _TITLE_KINDS:
            key, prio = _TITLE_KINDS[kind]
            val = ev.get(key)
            if isinstance(val, str) and val.strip():
                scan.titles[prio] = val.strip()
            continue
        if kind not in ("user", "assistant") or ev.get("isMeta") or ev.get("isSidechain"):
            continue
        if isinstance(ev.get("cwd"), str) and ev["cwd"] and not scan.cwd:
            scan.cwd = ev["cwd"]
        msg = ev.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        text = _text_of(content)
        blocks = _blocks(content)
        if kind == "user":
            if any(b.get("type") == "tool_result" for b in blocks) and not text:
                scan.last_kind = "tool_result"
                continue
            if _is_noise(text):
                continue
            scan.messages += 1
            scan.last_kind = "user_text"
            scan.last_user_ts = _parse_ts(ev.get("timestamp")) or scan.last_user_ts
            if not scan.first_user:
                scan.first_user = text
        else:
            tools = [str(b.get("name", "")) for b in blocks if b.get("type") == "tool_use"]
            if not text and not tools:
                continue  # thinking-only block
            scan.messages += 1
            if text:
                scan.last_assistant_text = text
                scan.last_kind = "assistant_text"
            if tools:
                scan.last_tool = tools[-1] or scan.last_tool
                scan.last_kind = "tool_use"


def _read_head_tail(path: Path, size: int) -> tuple[list[str], list[str], bool]:
    """(head lines, tail lines, whole file read). Partial first/last lines are dropped."""
    with path.open("rb") as fh:
        head = fh.read(HEAD_BYTES)
        if size <= HEAD_BYTES:
            return head.decode("utf-8", "replace").splitlines(), [], True
        head_lines = head.decode("utf-8", "replace").splitlines()[:-1]  # last line is cut off
        start = max(size - TAIL_BYTES, HEAD_BYTES)
        fh.seek(start)
        data = fh.read()
    tail_lines = data.decode("utf-8", "replace").splitlines()
    if start > HEAD_BYTES:
        tail_lines = tail_lines[1:]  # first line is cut off
    return head_lines, tail_lines, start <= HEAD_BYTES


def parse_session(path: str | Path, mtime: float | None = None, size: int | None = None) -> Session | None:
    """Session from one transcript file, or None if unreadable or without real messages."""
    p = Path(path)
    try:
        st = p.stat()
        mtime = st.st_mtime if mtime is None else mtime
        size = st.st_size if size is None else size
        head, tail, whole = _read_head_tail(p, size)
    except OSError:
        return None
    scan = _Scan()
    _scan_lines(head, scan)
    head_msgs = scan.messages
    first_user, cwd = scan.first_user, scan.cwd
    if not whole:
        scan.messages = 0
        _scan_lines(tail, scan)
        counted = head_msgs + scan.messages
        read = HEAD_BYTES + TAIL_BYTES
        scan.messages = int(counted * size / read) if size > read else counted
    scan.first_user = first_user or scan.first_user
    scan.cwd = cwd or scan.cwd
    if scan.messages <= 0:
        return None
    for title_prio in (3, 2, 1):
        if title_prio in scan.titles:
            title = scan.titles[title_prio]
            break
    else:
        title = ""
    project_dir = scan.cwd or decode_project(p.parent.name)
    return Session(
        id=p.stem, path=str(p), project_dir=project_dir, project_name=basename(project_dir),
        title=speakable(title), first_message=_clip(speakable(scan.first_user), 400), last_activity=mtime,
        last_assistant_text=scan.last_assistant_text, last_tool=scan.last_tool,
        message_count=scan.messages, last_kind=scan.last_kind, last_user_ts=scan.last_user_ts)


_cache: dict[str, tuple[float, int, Session | None]] = {}


def load_session(path: str | Path) -> Session | None:
    """parse_session cached by (path, mtime, size)."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    key = str(path)
    hit = _cache.get(key)
    if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2]
    s = parse_session(path, st.st_mtime, st.st_size)
    _cache[key] = (st.st_mtime, st.st_size, s)
    return s


def clear_cache() -> None:
    _cache.clear()


def scan_sessions(max_sessions: int = MAX_SESSIONS) -> list[Session]:
    """All sessions, newest first (only the `max_sessions` newest transcripts are parsed)."""
    root = projects_dir()
    files: list[tuple[float, Path]] = []
    try:
        for d in root.iterdir():
            if not d.is_dir():
                continue
            for f in d.glob("*.jsonl"):
                try:
                    files.append((f.stat().st_mtime, f))
                except OSError:
                    continue
    except OSError:
        return []
    files.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _, f in files[:max_sessions]:
        s = load_session(f)
        if s is not None:
            out.append(s)
    return out


# ---- projects -------------------------------------------------------------------------------------------
_STOP = {"the", "a", "an", "my", "of", "in", "on", "to", "for", "at", "and", "please", "project", "projects",
         "folder", "directory", "repo", "repository", "session", "sessions", "conversation", "chat", "claude",
         "about", "one", "that", "this", "with", "from"}


def tokens(text: str) -> list[str]:
    """Lowercase word tokens; camelCase split, filler words dropped."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def _sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a)):
        return 0.9
    if len(a) >= 4 and len(b) >= 4:
        sm = difflib.SequenceMatcher(None, a, b)
        if sm.real_quick_ratio() >= 0.8 and sm.quick_ratio() >= 0.8:
            r = sm.ratio()
            if r >= 0.8:
                return r
    return 0.0


def _best(token: str, pool: list[str]) -> float:
    return max((_sim(token, t) for t in pool), default=0.0)


def _dir_score(q: list[str], path: str) -> float:
    """How well query tokens cover the folder name (weight 1), its parent (0.8) and grandparent (0.5).
    Zero unless the folder name itself matches at least one token."""
    parts = [p for p in re.split(r"[\\/]", path) if p]
    pools = [tokens(parts[-1])] if parts else [[]]
    pools += [tokens(parts[-2])] if len(parts) > 1 else []
    pools += [tokens(parts[-3])] if len(parts) > 2 else []
    weights = (1.0, 0.8, 0.5)
    own = sum(_best(t, pools[0]) for t in q)
    if own <= 0:
        compact = "".join(q)
        return 1.0 if compact and compact == "".join(pools[0]) else 0.0
    total = 0.0
    for t in q:
        total += max(_best(t, pool) * w for pool, w in zip(pools, weights))
    return total / len(q)


def _is_inside(child: str, parent: str) -> bool:
    c, p = _norm_path(child), _norm_path(parent)
    return c != p and c.startswith(p + "/")


def known_roots() -> list[Path]:
    out = []
    for name in ("desktop", "documents"):
        try:
            out.append(paths.known_folder(name))
        except Exception:  # noqa: BLE001
            continue
    return out


def projects(sessions: list[Session] | None = None, only_existing: bool = True) -> list[str]:
    """Project folders: from Claude's own history (newest first) plus subfolders of Desktop and Documents."""
    sessions = scan_sessions() if sessions is None else sessions
    seen: dict[str, str] = {}
    for s in sessions:
        seen.setdefault(_norm_path(s.project_dir), s.project_dir)
    out = [d for d in seen.values() if not only_existing or os.path.isdir(d)]
    for root in known_roots():
        try:
            subs = sorted(root.iterdir())
        except OSError:
            continue
        for d in subs:
            if d.name.startswith(".") or not d.is_dir():
                continue
            if _norm_path(str(d)) not in seen:
                seen[_norm_path(str(d))] = str(d)
                out.append(str(d))
    return out


def resolve_project(spoken: str, sessions: list[Session] | None = None) -> str | None:
    """Best matching project folder for what the user said, or None.
    Handles "the jarvis project", "ai on pc" (a folder holding the Jarvis project resolves to that
    project), "jarvis pc assistant"."""
    spoken = (spoken or "").strip().strip("\"'")
    if not spoken:
        return None
    if re.search(r"[\\/]|^[A-Za-z]:|^~", spoken):
        try:
            p = paths.resolve_path(spoken)
            if p.is_dir():
                return str(p)
        except (OSError, ValueError):
            pass
    q = tokens(spoken)
    if not q:
        return None
    sessions = scan_sessions() if sessions is None else sessions
    latest: dict[str, float] = {}
    for s in sessions:
        k = _norm_path(s.project_dir)
        latest[k] = max(latest.get(k, 0.0), s.last_activity)
    now = time.time()
    best: tuple[float, str] | None = None
    for d in projects(sessions):
        sc = _dir_score(q, d)
        if sc < 0.5:
            continue
        act = latest.get(_norm_path(d))
        sc += 0.1 if act is not None else 0.0
        if act is not None:
            sc += 0.05 * math.exp(-(now - act) / 86400 / 14)
        if best is None or sc > best[0]:
            best = (sc, d)
    if best is None:
        return None
    chosen = best[1]
    if _norm_path(chosen) not in latest:  # a bare folder (e.g. "Ai on pc"): prefer the project inside it
        inside = [(latest[k], k) for k in latest if _is_inside(k, chosen)]
        if inside:
            newest = max(inside)[1]
            for s in sessions:
                if _norm_path(s.project_dir) == newest:
                    return s.project_dir
    return chosen


# ---- finding sessions -------------------------------------------------------------------------------------
@dataclass
class Ranked:
    session: Session
    score: float


def _relevance(q: list[str], s: Session) -> float:
    pools = [(tokens(s.title), 1.0), (tokens(s.first_message), 0.6), (tokens(s.project_name), 0.5),
             (tokens(s.last_assistant_text[:400]), 0.3)]
    return sum(max(_best(t, pool) * w for pool, w in pools) for t in q) / len(q)


def find_sessions(query: str = "", project: str | None = None, limit: int = 3,
                  sessions: list[Session] | None = None, now: float | None = None) -> list[Ranked]:
    """Rank sessions by topic words (title, first message, project, last reply) plus a recency boost.
    `project` is a folder or what the user called it; an unknown project gives []. Empty query = newest first."""
    now = time.time() if now is None else now
    sessions = scan_sessions() if sessions is None else sessions
    if project and project.strip():
        folder = resolve_project(project, sessions)
        if folder is None:
            return []
        want = _norm_path(folder)
        sessions = [s for s in sessions if _norm_path(s.project_dir) == want]
    q = tokens(query)
    out: list[Ranked] = []
    for s in sessions:
        age_days = max(0.0, now - s.last_activity) / 86400
        recency = 0.2 * math.exp(-age_days / 7)
        if q:
            rel = _relevance(q, s)
            if rel < MIN_RELEVANCE:
                continue
            out.append(Ranked(s, rel + recency))
        else:
            out.append(Ranked(s, recency))
    out.sort(key=lambda r: (r.score, r.session.last_activity), reverse=True)
    return out[:limit]


def is_ambiguous(ranked: list[Ranked], margin: float = AMBIGUITY_MARGIN) -> bool:
    """True when the top two matches score within `margin` of each other."""
    return len(ranked) >= 2 and ranked[0].score - ranked[1].score < margin


# ---- status helpers -------------------------------------------------------------------------------------
def looks_busy(s: Session, now: float | None = None) -> bool:
    """Modified a moment ago and the last line is mid-turn (tool call, tool result or the user's prompt)."""
    now = time.time() if now is None else now
    return now - s.last_activity < BUSY_SECONDS and s.last_kind in ("tool_use", "tool_result", "user_text")


def turn_seconds(transcript_path: str, now: float | None = None) -> float | None:
    """Seconds since the last real user message in a transcript (how long the current turn ran), or None."""
    if not transcript_path:
        return None
    s = parse_session(transcript_path)
    if s is None or s.last_user_ts is None:
        return None
    return max(0.0, (time.time() if now is None else now) - s.last_user_ts)


__all__ = ["Session", "Ranked", "scan_sessions", "load_session", "parse_session", "projects", "resolve_project",
           "find_sessions", "is_ambiguous", "format_line", "format_list", "humanize_age"]
