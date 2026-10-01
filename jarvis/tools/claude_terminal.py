"""Open a visible terminal running interactive Claude Code with the user's request.

The user watches and steers Claude there; Jarvis does not read the answer back.
The prompt goes through a small PowerShell script plus a text file, so quotes, ampersands
and newlines in the request survive (no command-line quoting through cmd or wt).
"""

from __future__ import annotations

import os
import re
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


_PLAIN_ARG = re.compile(r"^--?[A-Za-z][\w-]*$")


def _ps_arg(arg: str) -> str:
    """Flags like --resume stay bare; values (ids, names with spaces) are single-quoted."""
    return arg if _PLAIN_ARG.match(arg) else _ps_quote(arg)


def build_args(mode: str = "new", session: str = "", name: str = "") -> list[str]:
    """Claude CLI arguments for a session: --continue, --resume <id> and/or --name <name>."""
    args: list[str] = []
    if mode == "continue":
        args.append("--continue")
    elif mode == "resume" and session:
        args += ["--resume", session]
    if name.strip():
        args += ["--name", name.strip()]
    return args


def build_script(invocation: str, folder: str, prompt_file: str | None, args: list[str] | None = None) -> str:
    """PowerShell script text that starts Claude (with `args`) in `folder`, prompt read from `prompt_file`."""
    flags = "".join(" " + _ps_arg(a) for a in (args or []))
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


def prepare_launch(binary: str, where: Path, prompt: str, args: list[str], tmp: Path) -> tuple[list[str], bool]:
    """Write the prompt file and start script under `tmp`. Returns (command line, uses_windows_terminal)."""
    tmp.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    prompt_file = None
    if prompt.strip():
        prompt_file = tmp / f"prompt-{stamp}.txt"
        prompt_file.write_text(prompt.strip(), encoding="utf-8")
    script = tmp / f"start-{stamp}.ps1"
    # BOM so Windows PowerShell 5 reads non-ASCII paths in the script correctly.
    script.write_text(build_script(claude_invocation(binary), str(where),
                                   str(prompt_file) if prompt_file else None, args), encoding="utf-8-sig")
    cmd = launch_command(str(script), str(where))
    return cmd, os.path.basename(cmd[0]).lower().startswith("wt")


def open_claude_terminal(folder: str, prompt: str = "", args: list[str] | None = None) -> Path:
    """Open a visible terminal running interactive Claude in `folder` with CLI `args` (--resume <id>,
    --continue, --name <name>) and the prompt. Returns the folder used. Windows only."""
    if not IS_WINDOWS:
        raise ToolError("opening a Claude terminal is only supported on Windows")
    binary = resolve_binary(ctx.config.claude_code.binary)
    where = resolve_path(folder) if folder.strip() else Path.home()
    if not Path(where).is_dir():
        raise ToolError(f"folder not found: {where}")
    cmd, is_wt = prepare_launch(binary, Path(where), prompt, list(args or []),
                                Path(tempfile.gettempdir()) / "jarvis-claude")
    flags = 0 if is_wt else subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    subprocess.Popen(cmd, cwd=str(where), creationflags=flags)
    return Path(where)


@tool("Open a visible terminal window running Claude (Claude Code) and give it a simple request, so Claude does "
      "the work on screen while the user watches. For a plain 'ask Claude X' with no named project or session. "
      "For existing or named sessions and projects use claude_open_session, claude_continue or "
      "claude_new_session instead. Jarvis does not hear Claude's answer.")
def claude_terminal(prompt: str = "", folder: str = "", mode: str = "new", session: str = "") -> str:
    """Start interactive Claude in a terminal.

    Args:
        prompt: What to send to Claude, in the user's words. Empty just opens Claude.
        folder: Folder to run Claude in (a project folder for coding work). Empty means the home folder.
        mode: "new" for a fresh conversation, "continue" to continue the most recent conversation in that folder, or "resume" with a session id.
        session: Claude session id, only with mode "resume".
    """
    mode = (mode or "new").strip().lower()
    if mode not in ("new", "continue", "resume"):
        mode = "new"
    where = open_claude_terminal(folder, prompt, build_args(mode, session))
    what = {"new": "a new", "continue": "the last", "resume": "that"}[mode]
    return f"Opened Claude in a terminal ({what} conversation) in {where}" + (" with your request." if prompt.strip() else ".")
