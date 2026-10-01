"""Pure helpers that decide whether the user is finished talking."""

from __future__ import annotations

import re

# A sentence ending on one of these is almost certainly cut off mid-thought:
# "open chrome with", "play some", "create a folder called".
DANGLING = {
    "a", "an", "the", "my", "your", "his", "her", "our", "their", "this", "that", "some",
    "to", "with", "as", "on", "in", "into", "at", "for", "from", "of", "by", "about", "and", "or", "but",
    "called", "named", "say", "saying", "like", "using", "via", "under", "inside", "then", "so",
    "um", "uh", "erm", "hmm",
}


def sounds_unfinished(text: str) -> bool:
    """True if the transcript looks cut off (trailing dangling word, ellipsis or dash)."""
    t = text.strip()
    if not t:
        return False
    if t.endswith(("...", "…", "-", "—", ",")):
        return True
    words = re.findall(r"[a-z']+", t.lower())
    return bool(words) and words[-1] in DANGLING and not t.endswith(("?", "!"))


def asks_question(reply: str) -> bool:
    """True if Jarvis's spoken reply ends by asking the user something."""
    return reply.strip().rstrip("\"')]").endswith("?")


# Whisper's usual mishearings of "Claude". Rare words are always fixed; "cloud(s)" only
# where the sentence is clearly about the assistant (so "cloud storage" stays intact).
_CLAUDE_ALWAYS = re.compile(r"\b(?:clod|claud|clawed|clyde|klaud|klod|clause)\b", re.IGNORECASE)
_CLOUD_CODE = re.compile(r"\bclouds?\s+code\b", re.IGNORECASE)
_CLOUD_CONTEXT = re.compile(
    r"\b(open|ask|tell|have|start|launch|message|text|asking|telling)\s+clouds?\b"
    r"|\bclouds?(?=\s+(session|sessions|chat|chats|conversation|terminal|desktop|app|ai)\b)", re.IGNORECASE)


def fix_names(text: str) -> str:
    """Correct names Whisper commonly mishears (Claude)."""
    t = _CLOUD_CODE.sub("Claude Code", text)
    t = _CLAUDE_ALWAYS.sub("Claude", t)
    return _CLOUD_CONTEXT.sub(lambda m: re.sub(r"(?i)clouds?", "Claude", m.group(0)), t)
