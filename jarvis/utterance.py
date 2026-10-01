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
