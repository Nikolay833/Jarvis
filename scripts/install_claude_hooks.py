"""Install (or remove) the Jarvis hooks in Claude Code's user settings (~/.claude/settings.json).

    python scripts/install_claude_hooks.py                # install / update
    python scripts/install_claude_hooks.py --uninstall    # remove only Jarvis entries

Registers `"<venv python>" -m jarvis.claude_hook` for Stop, Notification (one entry per notification type,
exact-string matchers), PostToolUse / PostToolUseFailure (edits and shell commands, for progress reports) and
PermissionRequest (voice approval; timeout 60 s, the hook itself waits at most 45 s and prints nothing on any
failure so Claude's own prompt appears). Existing settings and other hooks are kept; a timestamped backup is written first.
Idempotent: entries whose command contains "jarvis.claude_hook" are replaced. Needs no jarvis imports.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

MARKER = "jarvis.claude_hook"
NOTIFICATION_TYPES = ("permission_prompt", "agent_needs_input", "elicitation_dialog")
TIMEOUT = 10
PERMISSION_TIMEOUT = 60
TOOL_MATCHER = "Bash|PowerShell|Edit|Write|MultiEdit|NotebookEdit"


def default_python() -> str:
    root = Path(__file__).resolve().parent.parent
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        if (root / rel).is_file():
            return str(root / rel)
    return sys.executable


def settings_path() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR")
    return (Path(base) if base else Path.home() / ".claude") / "settings.json"


def build_command(python: str) -> str:
    """Quoted interpreter path (it may contain spaces) plus the module.

    Forward slashes: Claude Code on Windows may run hooks through Git Bash, where backslashes
    are fragile; Windows itself accepts forward slashes in the path either way."""
    return f'"{python.replace(chr(92), "/")}" -m {MARKER}'


def _ours(entry: Any) -> bool:
    return isinstance(entry, dict) and any(
        isinstance(h, dict) and MARKER in str(h.get("command", "")) for h in entry.get("hooks", []) or [])


def _strip_ours(entries: Any) -> list[Any]:
    """Entries without our hooks; foreign hooks that share an entry with ours are kept."""
    out = []
    for e in entries if isinstance(entries, list) else []:
        if not _ours(e):
            out.append(e)
            continue
        rest = [h for h in e.get("hooks", []) if not (isinstance(h, dict) and MARKER in str(h.get("command", "")))]
        if rest:
            out.append({**e, "hooks": rest})
    return out


def _hook(command: str, matcher: str, timeout: int = TIMEOUT) -> dict[str, Any]:
    return {"matcher": matcher, "hooks": [{"type": "command", "command": command, "timeout": timeout}]}


def merge(settings: dict[str, Any], command: str) -> dict[str, Any]:
    """Settings with the Jarvis hooks added (old Jarvis entries replaced)."""
    out = dict(settings)
    hooks = dict(out.get("hooks") or {})
    hooks["Stop"] = _strip_ours(hooks.get("Stop")) + [_hook(command, "")]
    hooks["Notification"] = _strip_ours(hooks.get("Notification")) + [_hook(command, t) for t in NOTIFICATION_TYPES]
    for event in ("PostToolUse", "PostToolUseFailure"):
        hooks[event] = _strip_ours(hooks.get(event)) + [_hook(command, TOOL_MATCHER)]
    hooks["PermissionRequest"] = _strip_ours(hooks.get("PermissionRequest")) + [
        _hook(command, "", PERMISSION_TIMEOUT)]
    out["hooks"] = hooks
    return out


def remove(settings: dict[str, Any]) -> dict[str, Any]:
    """Settings without any Jarvis hook entries; empty events (and an empty hooks map) are dropped."""
    out = dict(settings)
    hooks = out.get("hooks")
    if not isinstance(hooks, dict):
        return out
    hooks = dict(hooks)
    for event in list(hooks):
        if not isinstance(hooks[event], list):
            continue
        had_ours = any(_ours(e) for e in hooks[event])
        kept = _strip_ours(hooks[event])
        if kept:
            hooks[event] = kept
        elif had_ours:
            del hooks[event]
    if hooks:
        out["hooks"] = hooks
    else:
        out.pop("hooks", None)
    return out


def load(path: Path) -> dict[str, Any]:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("settings.json is not a JSON object")
    return data


def backup(path: Path) -> Path | None:
    if not path.is_file():
        return None
    dest = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, dest)
    return dest


def run(path: Path, python: str, uninstall: bool = False) -> str:
    try:
        settings = load(path)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Cannot read {path}: {exc}. Fix or remove the file and run again.")
    new = remove(settings) if uninstall else merge(settings, build_command(python))
    if new == settings and path.is_file():
        return f"No change needed in {path}."
    bak = backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    note = f" Backup: {bak}." if bak else ""
    return (f"Removed the Jarvis hooks from {path}." if uninstall else f"Installed the Jarvis hooks in {path}.") + note


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Install the Jarvis hooks into Claude Code settings")
    ap.add_argument("--uninstall", action="store_true", help="remove only the Jarvis hook entries")
    ap.add_argument("--settings", help="settings.json to edit (default ~/.claude/settings.json)")
    ap.add_argument("--python", help="python used to run the hook (default: this repo's .venv)")
    args = ap.parse_args(argv)
    print(run(Path(args.settings) if args.settings else settings_path(), args.python or default_python(),
              args.uninstall))
    return 0


if __name__ == "__main__":
    sys.exit(main())
