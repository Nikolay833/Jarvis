"""Tool: the user's briefing (see briefing.py)."""

from __future__ import annotations

from datetime import datetime

from .. import briefing as brief
from .. import reminders as rem
from .context import ctx
from .registry import ToolError, tool


@tool("Give the user's briefing: greeting, date, weather, today's reminders and Claude Code activity. "
      "Returns the text to say; say it as given.")
async def briefing() -> str:
    cfg = ctx.config
    if not cfg.briefing.enabled:
        raise ToolError("The briefing is switched off in the config.")
    now = datetime.now()
    text = await brief.build(cfg, now, held=rem.take_held())
    brief.mark_done(cfg.briefing, now)
    return text
