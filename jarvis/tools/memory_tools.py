"""Long-term memory tools: remember, recall, forget (see memory.py)."""

from __future__ import annotations

import re

from ..memory import FactError, default_memory, second_person
from .registry import ToolError, tool


@tool("Remember a lasting fact or preference about the user (stored on this PC). Use when they say "
      "'remember ...' or clearly state something durable about themselves.")
def remember(fact: str, source: str = "explicit") -> str:
    """Store one short fact.

    Args:
        fact: one short statement, e.g. "Prefers Spotify for music", "Main project is Jarvis", "Name is Nikolay"
        source: "explicit" when the user said remember, "inferred" when you picked it up from what they said
    """
    try:
        _, created = default_memory().add(fact, source)
    except FactError as exc:
        raise ToolError(str(exc))
    return "Noted" if created else "Updated that"


@tool("Look up what is remembered about the user. Empty query lists everything.")
def recall(query: str = "") -> str:
    """Recall stored facts.

    Args:
        query: topic to look up, or empty for everything
    """
    mem = default_memory()
    facts = mem.search(query)
    if not facts:
        return "I don't have anything stored about that" if query.strip() else "I don't remember anything yet"
    return "You told me: " + mem.spoken(facts)


@tool("Forget a remembered fact. Query: words from the fact, e.g. 'my favourite colour'.")
def forget(query: str) -> str:
    """Remove the best matching fact.

    Args:
        query: words from the fact to forget
    """
    if re.fullmatch(r"\s*(?:all|everything|all of it|it all)\s*", query, re.I):
        raise ToolError("Ask the user which fact to forget; never wipe all memory at once.")
    gone = default_memory().forget(query)
    if not gone:
        return "I don't remember anything like that"
    said = [second_person(g["text"]).rstrip(".") for g in gone]
    return "Done. I've forgotten that " + " and that ".join((s[:1].lower() + s[1:] if s.startswith("You") else s) for s in said)
