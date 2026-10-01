"""Risk classification of tool calls. Pure functions, no I/O."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

SAFE = "safe"
RISKY = "risky"


@dataclass(frozen=True)
class Assessment:
    risk: str
    reason: str = ""

    @property
    def is_risky(self) -> bool:
        return self.risk == RISKY


_I = re.IGNORECASE

# (pattern, reason). Any hit makes a PowerShell command risky.
_POWERSHELL_RISKY: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, _I), r)
    for p, r in [
        (r"\bRemove-\w+", "deletes things"),
        (r"(?<![\w-])(rm|ri|del|erase|rmdir|rd)(?![\w-])", "deletes files"),
        (r"\bClear-(Disk|RecycleBin|Content|Item)\b", "clears data"),
        (r"\bFormat-Volume\b|(?<![\w-])format\s+[a-z]:", "formats a disk"),
        (r"\b(Initialize|Clear)-Disk\b|\bdiskpart\b", "changes disks"),
        (r"\bStop-\w+|\btaskkill\b|(?<![\w-])(kill|spps|spsv)(?![\w-])", "stops processes or services"),
        (r"\b(Restart|Stop)-Computer\b|(?<![\w-])shutdown(?:\.exe)?(?![\w-])", "shuts down or restarts"),
        (r"\b(Set|New|Remove)-ItemProperty\b|\bHK(LM|CU|CR|U|CC):|\bregistry::", "touches the registry"),
        (r"(?<![\w-])reg(?:\.exe)?\s+(add|delete|import|load|unload)\b|\bregedit\b", "writes the registry"),
        (r"\b(Invoke-WebRequest|iwr|Invoke-RestMethod|irm|curl|wget)\b[^|]*\|\s*(iex|Invoke-Expression)\b", "downloads and runs code"),
        (r"(?<![\w-])iex(?![\w-])|\bInvoke-Expression\b", "runs dynamic code"),
        (r"-EncodedCommand\b|(?<![\w-])-(e|ec|en|enc|enco\w*)\s", "runs encoded code"),
        (r"\b(Install|Uninstall|Update)-(Module|Package|Script|WindowsFeature)\b", "installs software"),
        (r"\b(winget|choco|scoop)\s+(install|uninstall|upgrade)\b", "installs software"),
        (r"\b(pip3?|npm|yarn|pnpm|cargo)\s+(install|uninstall|add)\b", "installs packages"),
        (r"\bmsiexec\b|\bStart-Process\b[^|;]*-Verb\s+RunAs\b", "installs or elevates"),
        (r"\bSet-ExecutionPolicy\b", "changes execution policy"),
        (r"\b(Move|Rename|Set|Add|Out|Copy|New)-(Item|Content|File)\b", "modifies files"),
        (r"(?<![\w-])(mv|mi|move|ren|rni|cp|cpi|copy|sc|ac|ni)(?![\w-])", "modifies files"),
        (r"\b(Disable|Enable|Unregister|Register|Revoke)-\w+", "changes system settings"),
        (r"\b(Set|Add)-MpPreference\b|\bnetsh\b|\bbcdedit\b|\bschtasks\b|\bsc(?:\.exe)?\s+(delete|config|stop)\b", "changes system settings"),
        (r"\bnet\s+(user|localgroup)\b|\b(New|Set)-Local(User|Group)\b", "changes accounts"),
        # output redirection to a file (not 2>&1, not > $null)
        (r"(?<![\w\-=<>])\d?>>?(?![&>])(?!\s*\$null)", "writes to a file"),
    ]
]

_EXECUTABLE_EXTS = {
    ".exe", ".bat", ".cmd", ".ps1", ".msi", ".vbs", ".vbe", ".js", ".jse",
    ".wsf", ".scr", ".com", ".jar", ".reg", ".lnk", ".hta", ".msc",
}


def classify_powershell(command: str) -> Assessment:
    for pattern, reason in _POWERSHELL_RISKY:
        if pattern.search(command):
            return Assessment(RISKY, reason)
    return Assessment(SAFE)


def classify_call(name: str, args: dict[str, Any], base_risk: str = SAFE) -> Assessment:
    """Final risk of one tool call: the tool's base risk, raised by its arguments."""
    if base_risk == RISKY:
        return Assessment(RISKY, "tool is always risky")
    if name == "run_powershell":
        return classify_powershell(str(args.get("command", "")))
    if name == "open_path":
        ext = os.path.splitext(str(args.get("path", "")))[1].lower()
        if ext in _EXECUTABLE_EXTS:
            return Assessment(RISKY, "runs a program")
    return Assessment(SAFE)


def _clip(text: str, n: int = 120) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "..."


def describe_call(name: str, args: dict[str, Any]) -> str:
    """Short human summary used in 'Sir, I'm about to <summary>'."""
    if name == "run_powershell":
        return f"run a PowerShell command: {_clip(args.get('command', ''))}"
    if name == "delete_path":
        return f"permanently delete {_clip(args.get('path', ''))}"
    if name == "move_path":
        return f"move {_clip(args.get('src', ''))} to {_clip(args.get('dst', ''))}"
    if name == "open_path":
        return f"open {_clip(args.get('path', ''))}"
    if name == "claude_code_run":
        raw = str(args.get("folder", ""))
        parts = [p for p in re.split(r"[\\/]", raw) if p]
        folder = parts[-1] if parts else raw
        return f"start Claude Code in {_clip(folder, 60)} to: {_clip(args.get('prompt', ''), 100)}"
    pretty = ", ".join(f"{k}={_clip(v, 40)}" for k, v in args.items())
    return f"call {name}({pretty})"
