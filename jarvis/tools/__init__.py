"""Tools. Import `load_all()` to register every tool on the global registry."""

from __future__ import annotations

from .registry import Registry, Tool, ToolError, registry, tool


def load_all() -> Registry:
    from . import (apps, briefing_tool, chrome, claude_chat, claude_code, claude_history,  # noqa: F401
                   claude_sessions_tools, claude_terminal, files, maps_tools, memory_tools, reminder_tools,
                   spotify, system, windows)  # (register on import)

    return registry


__all__ = ["Registry", "Tool", "ToolError", "registry", "tool", "load_all"]
