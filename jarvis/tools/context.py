"""Shared runtime context for tools (config, bus, speaking). Set once at startup."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..config import Config

IS_WINDOWS = sys.platform == "win32"
# Hide console windows for child processes on Windows.
NO_WINDOW = 0x08000000 if IS_WINDOWS else 0


async def _noop_speak(text: str) -> None:
    return None


@dataclass
class ToolContext:
    config: Config = field(default_factory=Config)
    bus: Any = None
    # Speak an out-of-turn announcement (job finished). Replaced by the assistant.
    speak: Callable[[str], Awaitable[None]] = _noop_speak


ctx = ToolContext()


def set_context(config: Config, bus: Any = None, speak: Callable[[str], Awaitable[None]] | None = None) -> ToolContext:
    ctx.config = config
    ctx.bus = bus
    ctx.speak = speak or _noop_speak
    return ctx


def emit(type: str, **fields: Any) -> None:
    if ctx.bus is not None:
        ctx.bus.emit_nowait(type, **fields)
