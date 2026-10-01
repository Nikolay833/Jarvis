"""Tools. Import `load_all()` to register every tool on the global registry."""

from __future__ import annotations

from .registry import Registry, Tool, ToolError, registry, tool


def load_all() -> Registry:
    from . import apps, claude_code, claude_history, files, system  # noqa: F401  (register on import)

    return registry


__all__ = ["Registry", "Tool", "ToolError", "registry", "tool", "load_all"]
