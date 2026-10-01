"""App and URL launching."""

from __future__ import annotations

import subprocess
import webbrowser

from .context import IS_WINDOWS, NO_WINDOW
from .registry import ToolError, tool

# spoken name -> Start-Process target
APP_MAP: dict[str, str] = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "vscode": "code",
    "vs code": "code",
    "visual studio code": "code",
    "code": "code",
    "explorer": "explorer",
    "file explorer": "explorer",
    "files": "explorer",
    "terminal": "wt",
    "windows terminal": "wt",
    "wt": "wt",
    "spotify": "spotify",
    "notepad": "notepad",
}


# Apps that need a custom launch command.
SPECIAL: dict[str, str] = {
    "discord": "Start-Process \"$env:LOCALAPPDATA\\Discord\\Update.exe\" -ArgumentList '--processStart','Discord.exe'",
}


def _q(text: str) -> str:
    """Quote for a PowerShell single-quoted string."""
    return text.replace("'", "''")


def build_open_app_command(name: str) -> str:
    """PowerShell command to launch an app: known mapping, else Start Menu lookup."""
    key = " ".join(name.lower().split())
    if key in SPECIAL:
        return SPECIAL[key]
    if key in APP_MAP:
        return f"Start-Process '{APP_MAP[key]}'"
    q = _q(name.strip())
    return (
        f"$a = Get-StartApps | Where-Object {{ $_.Name -like '*{q}*' }} | Select-Object -First 1; "
        f"if ($a) {{ Start-Process \"shell:AppsFolder\\$($a.AppID)\" }} "
        f"else {{ Write-Error 'No app named {q}'; exit 1 }}"
    )


@tool("Open an application by name, e.g. chrome, edge, vscode, explorer, terminal, spotify, discord, notepad, "
      "or any other installed app.")
def open_app(name: str) -> str:
    """Open an app.

    Args:
        name: Spoken app name.
    """
    if not IS_WINDOWS:
        raise ToolError("open_app is only supported on Windows")
    cmd = build_open_app_command(name)
    res = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
        capture_output=True, text=True, timeout=20, creationflags=NO_WINDOW,
    )
    if res.returncode != 0:
        raise ToolError((res.stderr or res.stdout).strip()[:300] or f"could not open {name}")
    return f"Opened {name}"


@tool("Open a web address (http or https) in the default browser.")
def open_url(url: str) -> str:
    """Open a URL.

    Args:
        url: Full address; https:// is added if missing.
    """
    u = url.strip()
    if "://" not in u:
        u = "https://" + u
    if not u.lower().startswith(("http://", "https://")):
        raise ToolError("only http and https addresses can be opened")
    if not webbrowser.open(u):
        raise ToolError("no browser available")
    return f"Opened {u}"
