"""Glue between the Assistant and memory / reminders / briefing (kept out of __main__.py on purpose).

`Extras(assistant)` wires the memory block into the agent, runs the reminder scheduler and speaks the
automatic morning briefing on the first wake of the day.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from . import briefing, fastpath, memory
from . import reminders as rem

log = logging.getLogger("jarvis.extras")

BRIEFING_BUILD_TIMEOUT = 15.0


class Extras:
    def __init__(self, assistant: Any) -> None:
        self.a = assistant
        self.cfg = assistant.cfg
        self.scheduler: rem.Scheduler | None = None
        self.memory = memory.default_memory()
        self.memory.max_facts = self.cfg.memory.max_facts
        if self.cfg.memory.enabled:
            assistant.agent.memory_block = memory.known_block

    # ---- reminders ------------------------------------------------------------------------------
    def start(self, bg: list[asyncio.Task]) -> None:
        """Start the reminder scheduler (call once the assistant is online)."""
        if not self.cfg.reminders.enabled:
            return
        self.scheduler = rem.Scheduler(rem.default_store(), self.announce_reminder, hold_missed=self.briefing_pending)
        rem.set_active(self.scheduler)
        bg.append(asyncio.create_task(self.scheduler.run()))

    async def announce_reminder(self, text: str) -> None:
        """A gentle chime (when nobody is mid-turn), then the spoken announcement."""
        a = self.a
        if self.cfg.reminders.chime and a.voice and not a.turn_lock.locked():
            from .audio.mic import parse_device

            await asyncio.to_thread(rem.play_reminder_chime, parse_device(self.cfg.audio.output_device))
        await a.announce(text)

    def briefing_pending(self) -> bool:
        """True while the automatic briefing is still due today (missed reminders then go into it)."""
        try:
            return briefing.due_today(self.cfg.briefing, datetime.now())
        except Exception:  # noqa: BLE001
            return False

    # ---- briefing -------------------------------------------------------------------------------
    async def first_wake_briefing(self, req: Any, text: str) -> None:
        """On the first wake-word/hotkey activation of the day: speak the briefing, then the turn goes on.

        Called inside run_turn (the turn lock is held) with what the user asked."""
        if req.kind != "activate" or req.source not in ("wake", "hotkey"):
            return
        now = datetime.now()
        try:
            if not briefing.due_today(self.cfg.briefing, now):
                return
            fp = fastpath.match(text) if self.cfg.agent.fast_paths else None
            if fp is not None and fp.kind == "briefing":
                return  # they asked for it themselves; the tool marks the day
            held = rem.take_held()
            try:
                text_out = await asyncio.wait_for(briefing.build(self.cfg, now, held=held), BRIEFING_BUILD_TIMEOUT)
            except BaseException:
                rem.restore_held(held)
                raise
            briefing.mark_done(self.cfg.briefing, now)
        except Exception:  # noqa: BLE001 - a broken briefing must never block the request
            log.warning("automatic briefing failed", exc_info=True)
            return
        log.info("automatic briefing: %s", text_out)
        self.a.agent.add_exchange("(automatic morning briefing)", text_out)
        await self.a.say_safe(text_out)
