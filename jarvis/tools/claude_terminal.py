"""Open a visible terminal running interactive Claude Code with the user's request.

The user watches and steers Claude there; Jarvis does not read the answer back.
The prompt goes through a small PowerShell script plus a text file, so quotes, ampersands
and newlines in the request survive (no command-line quoting through cmd or wt).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from ..paths import resolve_path
from .claude_code import resolve_binary
from .context import IS_WINDOWS, ctx
from .registry import ToolError, tool


def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def claude_invocation(binary: str) -> str:
    """How to call Claude from PowerShell. Prefer the npm .ps1 shim next to claude.cmd:
    it hands arguments to node unchanged, while the .cmd shim re-parses them through cmd."""
    p = Path(binary)
    if p.suffix.lower() == ".cmd" and p.with_suffix(".ps1").is_file():
        return "& " + _ps_quote(str(p.with_suffix(".ps1")))
    return "& " + _ps_quote(binary)


def build_script(invocation: str, folder: str, prompt_file: str | None, mode: str, session: str) -> str:
    """PowerShell script text that starts Claude in `folder` with the prompt from `prompt_file`."""
    flags = ""
    if mode == "continue":
        flags = " --continue"
    elif mode == "resume" and session:
        flags = " --resume " + _ps_quote(session)
    lines = [
        "$Host.UI.RawUI.WindowTitle = 'Claude (Jarvis)'",
        "Set-Location -LiteralPath " + _ps_quote(folder),
    ]
    if prompt_file:
        lines.append("$p = Get-Content -Raw -Encoding UTF8 -LiteralPath " + _ps_quote(prompt_file))
        lines.append(f"{invocation}{flags} $p")
    else:
        lines.append(f"{invocation}{flags}")
    return "\r\n".join(lines) + "\r\n"


def launch_command(script: str, folder: str) -> list[str]:
    """Windows Terminal tab if available, else a plain PowerShell console."""
    ps = ["powershell.exe", "-NoLogo", "-NoExit", "-ExecutionPolicy", "Bypass", "-File", script]
    wt = shutil.which("wt.exe") or shutil.which("wt")
    if wt:
        return [wt, "-w", "new", "-d", folder, "--title", "Claude (Jarvis)", *ps]
    return ps


@tool("Open a visible terminal window running Claude (Claude Code) and give it the user's request, so Claude "
      "does the work on screen while the user watches. Use this when the user asks you to ask Claude, tell "
      "Claude, or have Claude do something. Jarvis does not hear Claude's answer.")
def claude_terminal(prompt: str = "", folder: str = "", mode: str = "new", session: str = "") -> str:
    """Start interactive Claude in a terminal.

    Args:
        prompt: What to send to Claude, in the user's words. Empty just opens Claude.
        folder: Folder to run Claude in (a project folder for coding work). Empty means the home folder.
        mode: "new" for a fresh conversation, "continue" to continue the most recent conversation in that folder, or "resume" with a session id.
        session: Claude session id, only with mode "resume".
    """
    if not IS_WINDOWS:
        raise ToolError("opening a Claude terminal is only supported on Windows")
    binary = resolve_binary(ctx.config.claude_code.binary)
    where = resolve_path(folder) if folder.strip() else Path.home()
    if not Path(where).is_dir():
        raise ToolError(f"folder not found: {where}")
    mode = (mode or "new").strip().lower()
    if mode not in ("new", "continue", "resume"):
        mode = "new"
    tmp = Path(tempfile.gettempdir()) / "jarvis-claude"
    tmp.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    prompt_file = None
    if prompt.strip():
        prompt_file = tmp / f"prompt-{stamp}.txt"
        prompt_file.write_text(prompt.strip(), encoding="utf-8")
    script = tmp / f"start-{stamp}.ps1"
    # BOM so Windows PowerShell 5 reads non-ASCII paths in the script correctly.
    script.write_text(build_script(claude_invocation(binary), str(where),
                                   str(prompt_file) if prompt_file else None, mode, session),
                      encoding="utf-8-sig")
    cmd = launch_command(str(script), str(where))
    flags = 0 if os.path.basename(cmd[0]).lower().startswith("wt") else subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    subprocess.Popen(cmd, cwd=str(where), creationflags=flags)
    what = {"new": "a new", "continue": "the last", "resume": "that"}[mode]
    return f"Opened Claude in a terminal ({what} conversation) in {where}" + (" with your request." if prompt_file else ".")
