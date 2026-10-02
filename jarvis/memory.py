"""Long-term memory: short facts about the user, kept in %APPDATA%/Jarvis/memory.json.

A fact is {id, text, created, source: "explicit"|"inferred"}. Near duplicates update the existing fact.
Relevant facts reach the model as a "[Known about the user: ...]" block on the newest user message
(see Agent), never in the static system prompt, so the Ollama KV cache stays valid.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from . import store

MAX_FACTS = 200
BLOCK_FACTS = 8     # facts per injected block
CORE_FACTS = 3      # core facts (name, preferences) always included
MAX_FACT_CHARS = 300
EXPLICIT, INFERRED = "explicit", "inferred"

_STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "and", "or", "my", "i", "me", "that",
         "this", "it", "user", "users", "he", "she", "they", "his", "her", "their", "in", "on", "at", "for",
         "with", "as", "do", "does", "am", "im", "ive", "s", "about", "remember", "you", "your", "what",
         "know", "tell", "please", "has", "have", "had", "mine", "our", "we", "so", "if", "from"}
_CORE = re.compile(r"\b(name|call me|called|prefers?|favou?rite|likes?|loves?|hates?|dislikes?|always|never|lives?|"
                   r"works?|main|born|birthday|wife|husband|girlfriend|boyfriend|partner|kids?|children|dog|cat)\b",
                   re.IGNORECASE)
_NAME = re.compile(r"\b(?:my name is|name is|call me|i am called|i'm called|is called|called)\s+([A-Za-z][\w'-]*)",
                   re.IGNORECASE)
_SECRET = re.compile(r"\b(pass(?:word|code|phrase)|pin code|\bpin\b|api[ _-]?key|secret|token|private key|"
                     r"credit card|card number|cvv|ssn|social security)\b|\b(?:sk|pk|ghp|xox[bp])[-_][A-Za-z0-9_-]{12,}",
                     re.IGNORECASE)


class FactError(ValueError):
    """The fact cannot be stored (empty, too secret)."""


def stem(word: str) -> str:
    for suffix in ("ing", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    """Lowercase content words (apostrophes dropped, stop words removed, light stemming)."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower().replace("'", "").replace("’", ""))
    return [stem(w) for w in words if w not in _STOP]


def _same(a: set[str], b: set[str]) -> bool:
    if not a or not b:
        return False
    inter = len(a & b)
    if inter / len(a | b) >= 0.7:
        return True
    small = a if len(a) <= len(b) else b
    return len(small) >= 3 and small <= (a & b)


def _match(q: set[str], fact_tokens: set[str]) -> float:
    """How well `fact_tokens` covers the query tokens (0..1); a 4+ letter prefix counts as a hit."""
    if not q:
        return 0.0
    hits = 0
    for t in q:
        if t in fact_tokens or (len(t) >= 4 and any(f.startswith(t) or t.startswith(f) for f in fact_tokens if len(f) >= 4)):
            hits += 1
    return hits / len(q)


def second_person(text: str) -> str:
    """'my main project is Jarvis' -> 'your main project is Jarvis' (for speaking facts back)."""
    t = " ".join(text.split())
    t = re.sub(r"\bcall me ([\w'-]+)", r"you like to be called \1", t, flags=re.I)
    for pat, rep in ((r"\bi am\b", "you are"), (r"\bi'm\b", "you're"), (r"\bi've\b", "you've"),
                     (r"\bi'll\b", "you'll"), (r"\bi'd\b", "you'd"), (r"\bmyself\b", "yourself"),
                     (r"\bmine\b", "yours"), (r"\bmy\b", "your"), (r"\bme\b", "you"), (r"\bi\b", "you")):
        t = re.sub(pat, rep, t, flags=re.I)
    return t[:1].upper() + t[1:] if t else t


class Memory:
    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.time,
                 max_facts: int = MAX_FACTS) -> None:
        self.path = path or store.app_dir() / "memory.json"
        self._clock = clock
        self.max_facts = max_facts
        self._lock = threading.RLock()

    # ---- storage ----------------------------------------------------------------------------------
    def facts(self) -> list[dict[str, Any]]:
        data = store.read_json(self.path, [])
        if isinstance(data, dict):
            data = data.get("facts", [])
        out = []
        for f in data if isinstance(data, list) else []:
            if isinstance(f, dict) and str(f.get("text", "")).strip():
                out.append({"id": str(f.get("id") or uuid.uuid4().hex[:8]), "text": str(f["text"]).strip(),
                            "created": float(f.get("created") or 0.0),
                            "source": INFERRED if f.get("source") == INFERRED else EXPLICIT})
        return out

    def _save(self, facts: list[dict[str, Any]]) -> None:
        store.write_json(self.path, facts)

    # ---- add / forget -----------------------------------------------------------------------------
    def add(self, text: str, source: str = EXPLICIT) -> tuple[dict[str, Any], bool]:
        """Store a fact. Returns (fact, created); created is False when it updated a near duplicate."""
        text = " ".join((text or "").split()).strip(" .")
        if not text:
            raise FactError("There is nothing to remember.")
        if len(text) > MAX_FACT_CHARS:
            raise FactError("That is too long to remember as one fact; give me the short version.")
        if _SECRET.search(text):
            raise FactError("I don't store passwords, keys or other secrets.")
        source = INFERRED if source == INFERRED else EXPLICIT
        with self._lock:
            facts = self.facts()
            new_tokens = set(tokens(text))
            for f in facts:
                if f["text"].lower() == text.lower() or _same(new_tokens, set(tokens(f["text"]))):
                    f.update(text=text, created=self._clock(),
                             source=EXPLICIT if EXPLICIT in (f["source"], source) else INFERRED)
                    self._save(facts)
                    return f, False
            fact = {"id": uuid.uuid4().hex[:8], "text": text, "created": self._clock(), "source": source}
            facts.append(fact)
            while len(facts) > self.max_facts:  # drop the oldest inferred fact first, else the oldest
                pool = [f for f in facts if f["source"] == INFERRED] or facts
                facts.remove(min(pool, key=lambda f: f["created"]))
            self._save(facts)
            return fact, True

    def forget(self, query: str) -> list[dict[str, Any]]:
        """Remove the best matching fact (all facts tied for best). Returns what was removed."""
        q = set(tokens(query))
        with self._lock:
            facts = self.facts()
            scored = [(_match(q, set(tokens(f["text"]))), f) for f in facts]
            best = max((s for s, _ in scored), default=0.0)
            if not q or best < 0.5:
                return []
            gone = [f for s, f in scored if s == best]
            self._save([f for f in facts if f not in gone])
            return gone

    # ---- search -----------------------------------------------------------------------------------
    def ranked(self, query: str) -> list[tuple[float, dict[str, Any]]]:
        q = set(tokens(query))
        out = [(_match(q, set(tokens(f["text"]))), f) for f in self.facts()]
        return sorted((x for x in out if x[0] > 0), key=lambda x: (-x[0], -x[1]["created"]))

    def search(self, query: str = "", limit: int = BLOCK_FACTS) -> list[dict[str, Any]]:
        """Relevant facts for `query`; everything (newest first) for an empty query or a short list."""
        facts = sorted(self.facts(), key=lambda f: -f["created"])
        if not set(tokens(query)):
            return facts
        if len(facts) <= 5:
            return facts
        return [f for _, f in self.ranked(query)][:limit]

    def core(self, limit: int = CORE_FACTS) -> list[dict[str, Any]]:
        core = [f for f in self.facts() if _CORE.search(f["text"])]
        core.sort(key=lambda f: (not _NAME.search(f["text"]), f["source"] != EXPLICIT, -f["created"]))
        return core[:limit]

    def name(self) -> str:
        for f in sorted(self.facts(), key=lambda f: -f["created"]):
            m = _NAME.search(f["text"])
            if m:
                return m.group(1).capitalize() if m.group(1).islower() else m.group(1)
        return ""

    def block(self, query: str, limit: int = BLOCK_FACTS) -> str:
        """'[Known about the user: ...]\\n' for the newest user message, or '' when nothing is stored."""
        facts = self.facts()
        if not facts:
            return ""
        if len(facts) <= limit:
            chosen = sorted(facts, key=lambda f: f["created"])
        else:
            chosen = self.core()
            for _, f in self.ranked(query):
                if len(chosen) >= limit:
                    break
                if f not in chosen:
                    chosen.append(f)
        return "[Known about the user: " + "; ".join(f["text"] for f in chosen[:limit]) + "]\n"

    def spoken(self, facts: list[dict[str, Any]], limit: int = 10) -> str:
        """Facts as one speakable text, in the second person."""
        shown = facts[:limit]
        text = ". ".join(second_person(f["text"]).rstrip(".") for f in shown) + "."
        extra = len(facts) - len(shown)
        return text + (f" And {extra} more." if extra > 0 else "")


_instances: dict[Path, Memory] = {}


def default_memory() -> Memory:
    """The shared Memory for the current app folder (APPDATA is read each call, so tests can redirect it)."""
    path = store.app_dir() / "memory.json"
    return _instances.setdefault(path, Memory(path))


def known_block(query: str) -> str:
    """Agent hook: the injection block for this user message (never raises)."""
    try:
        return default_memory().block(query)
    except Exception:  # noqa: BLE001 - memory must never break a turn
        return ""


__all__ = ["Memory", "FactError", "default_memory", "known_block", "tokens", "second_person",
           "EXPLICIT", "INFERRED"]
