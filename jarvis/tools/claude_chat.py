"""Normal chat with Claude through the Claude Code CLI (uses the user's subscription).

Not for coding: chats run in an empty dedicated folder with file/shell tools disallowed.
Sessions are stored in %APPDATA%/Jarvis/claude_chats.json, transcripts in claude_chats/<name>.md.
The user's message goes to the CLI on stdin (not argv): no quoting problems with the Windows
claude.cmd shim and no length limit.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from .claude_code import resolve_binary
from .context import NO_WINDOW, ctx
from .registry import ToolError, tool

TIMEOUT = 120.0
CONTINUE_WINDOW = 30 * 60  # seconds: a chat used within this is continued when no name is given
DISALLOWED = "Bash,Edit,Write,MultiEdit,NotebookEdit"
ALLOWED = "WebSearch,WebFetch"
SYSTEM_PROMPT = ("You are being used as a general chat assistant through a voice assistant called Jarvis. "
                 "Answer conversationally and concisely, in 2 to 4 sentences, unless the user asks for detail. "
                 "Plain spoken text only: no markdown, no lists, no code blocks. You cannot edit files or run commands here.")
_STOP_WORDS = {"a", "an", "the", "to", "of", "and", "is", "it", "me", "my", "i", "can", "you", "please"}


# ---- storage ----------------------------------------------------------------
def app_dir() -> Path:
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "Jarvis"


def chat_dir() -> Path:
    return app_dir() / "claude-chat"


def store_path() -> Path:
    return app_dir() / "claude_chats.json"


def load_store() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(store_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_store(store: dict[str, dict[str, Any]]) -> None:
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .") or "chat"


def name_from_message(message: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9']+", message) if w.lower() not in _STOP_WORDS]
    return " ".join(words[:4]).lower() or "chat"


def unique_name(store: dict[str, Any], base: str) -> str:
    name, n = base, 2
    while name in store:
        name, n = f"{base} {n}", n + 1
    return name


def find_chat(store: dict[str, dict[str, Any]], query: str) -> str | None:
    """Chat name matching `query`: exact, else unique substring (case-insensitive)."""
    q = query.strip().lower()
    for n in store:
        if n.lower() == q:
            return n
    hits = [n for n in store if q and q in n.lower()]
    return hits[0] if len(hits) == 1 else None


def pick_recent(store: dict[str, dict[str, Any]], now: float | None = None) -> str | None:
    """Most recently used chat if used within the continue window."""
    now = time.time() if now is None else now
    if not store:
        return None
    name = max(store, key=lambda n: float(store[n].get("last_used", 0)))
    return name if now - float(store[name].get("last_used", 0)) <= CONTINUE_WINDOW else None


def new_entry(now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    return {"session_id": str(uuid.uuid4()), "created": now, "last_used": now, "cwd": str(chat_dir()),
            "started": False}


def ago(ts: float, now: float | None = None) -> str:
    secs = int((time.time() if now is None else now) - ts)
    if secs < 90:
        return "just now"
    if secs < 3600:
        return f"{secs // 60} minutes ago"
    if secs < 86400:
        h = secs // 3600
        return f"{h} hour{'s' if h != 1 else ''} ago"
    d = secs // 86400
    return f"{d} day{'s' if d != 1 else ''} ago"


# ---- command + output ---------------------------------------------------------
def build_command(binary: str, session_id: str, resume: bool) -> list[str]:
    """`claude -p` (message on stdin). New chat: --session-id <uuid>; existing: --resume <id>."""
    return [binary, "-p", "--output-format", "json",
            "--resume" if resume else "--session-id", session_id,
            "--append-system-prompt", SYSTEM_PROMPT,
            "--disallowedTools", DISALLOWED,
            "--allowedTools", ALLOWED]


def parse_result(stdout: str) -> tuple[str, str, bool]:
    """(answer text, session_id, is_error) from `--output-format json` output."""
    text = stdout.strip()
    try:
        data: Any = json.loads(text)
    except ValueError:
        return text, "", False  # not JSON: treat the raw output as the answer
    if isinstance(data, list):  # verbose/stream output: take the result event
        data = next((e for e in reversed(data) if isinstance(e, dict) and e.get("type") == "result"), {})
    if not isinstance(data, dict):
        return text, "", False
    is_err = bool(data.get("is_error")) or str(data.get("subtype", "success")).startswith("error")
    return str(data.get("result") or "").strip(), str(data.get("session_id") or ""), is_err


async def run_cli(cmd: list[str], cwd: str, message: str, timeout: float = TIMEOUT) -> tuple[int, str, str]:
    """Run the CLI without blocking the loop. Replaced in tests."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, creationflags=NO_WINDOW)
    try:
        out, err = await asyncio.wait_for(proc.communicate(message.encode("utf-8")), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolError(f"Claude did not answer within {int(timeout)} seconds")
    return proc.returncode or 0, out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace")


def append_transcript(name: str, question: str, answer: str, when: float | None = None) -> None:
    d = app_dir() / "claude_chats"
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(when or time.time()))
    with (d / f"{safe_filename(name)}.md").open("a", encoding="utf-8") as fh:
        fh.write(f"## {stamp}\n\n**Q:** {question}\n\n**A:** {answer}\n\n")


_lock = asyncio.Lock()


async def _ask(name: str, message: str) -> str:
    cfg = ctx.config.claude_code
    binary = resolve_binary(cfg.binary)
    cwd = chat_dir()
    cwd.mkdir(parents=True, exist_ok=True)
    async with _lock:
        store = load_store()
        entry = store.get(name) or new_entry()
        store[name] = entry
        resume = bool(entry.get("started"))
        sid = entry["session_id"]
        code, out, err = await run_cli(build_command(binary, sid, resume), str(cwd), message)
        answer, new_sid, is_err = parse_result(out)
        if resume and (code != 0 or is_err) and "no conversation found" in (answer + err + out).lower():
            sid = str(uuid.uuid4())  # Claude lost the session: start the chat over under the same name
            code, out, err = await run_cli(build_command(binary, sid, False), str(cwd), message)
            answer, new_sid, is_err = parse_result(out)
        if code != 0 or is_err or not answer:
            raise ToolError((answer or err.strip() or out.strip() or f"claude exited with code {code}")[:300])
        entry.update(session_id=new_sid or sid, started=True, last_used=time.time(), cwd=str(cwd))
        save_store(store)
        try:
            append_transcript(name, message, answer)
        except OSError:
            pass
    return answer


# ---- tools -----------------------------------------------------------------------
@tool("Ask Claude a question in the background and read the answer back to the user. Use ONLY when the "
      "user wants the answer spoken back ('ask Claude and tell me', 'what does Claude say'). For plain "
      "'ask Claude' / 'tell Claude' / 'have Claude do', use claude_terminal instead. Continues the recent "
      "chat unless a session name is given. Relay Claude's answer faithfully; shorten only if very long.")
async def claude_chat(message: str, session: str = "") -> str:
    """Chat with Claude.

    Args:
        message: What to say to Claude.
        session: Optional chat name; empty continues the chat used in the last 30 minutes, else starts a new one.
    """
    if not message.strip():
        raise ToolError("what should I ask Claude?")
    store = load_store()
    if session.strip():
        name = find_chat(store, session) or unique_name(store, session.strip())
    else:
        name = pick_recent(store) or unique_name(store, name_from_message(message))
    return await _ask(name, message)


@tool("Start a new, separate Claude chat with a name (optionally sending the first message). Use for "
      "'new Claude chat' or 'start a fresh conversation with Claude'.")
async def claude_chat_new(name: str, message: str = "") -> str:
    """New Claude chat.

    Args:
        name: Name for the chat.
        message: Optional first message to send.
    """
    store = load_store()
    nm = unique_name(store, name.strip() or "chat")
    if message.strip():
        return await _ask(nm, message)
    store[nm] = new_entry()
    save_store(store)
    return f"Started a new Claude chat called {nm}. Say 'ask Claude' to talk in it."


@tool("List the saved Claude chats with when each was last used.")
def claude_chat_list() -> str:
    store = load_store()
    if not store:
        return "No Claude chats yet"
    rows = sorted(store.items(), key=lambda kv: float(kv[1].get("last_used", 0)), reverse=True)
    return "\n".join(f"{n}, last used {ago(float(e.get('last_used', 0)))}" for n, e in rows[:10])


@tool("Permanently delete a saved Claude chat (its name and saved transcript). Needs approval.", risk="risky")
def claude_chat_delete(name: str) -> str:
    """Delete a Claude chat.

    Args:
        name: Chat name.
    """
    store = load_store()
    found = find_chat(store, name)
    if not found:
        raise ToolError(f"no Claude chat called '{name}'")
    del store[found]
    save_store(store)
    try:
        (app_dir() / "claude_chats" / f"{safe_filename(found)}.md").unlink()
    except OSError:
        pass
    return f"Deleted the Claude chat {found}"


_QA = re.compile(r"\*\*Q:\*\* (.*?)\n\n\*\*A:\*\* (.*?)(?=\n\n## |\Z)", re.S)


def read_exchanges(md: str) -> list[tuple[str, str]]:
    return [(q.strip(), a.strip()) for q, a in _QA.findall(md)]


@tool("Show the latest messages of a chat Jarvis had with Claude (claude_chat); default is the most recent chat. "
      "Not for Claude Code coding sessions (claude_code_history).")
def claude_chat_history(session: str = "", count: int = 4) -> str:
    """Recent Claude chat messages.

    Args:
        session: Chat name; empty means the most recently used chat.
        count: How many of the last question/answer pairs to show.
    """
    store = load_store()
    if not store:
        return "No Claude chats yet"
    if session.strip():
        name = find_chat(store, session)
        if not name:
            raise ToolError(f"no Claude chat called '{session}'. Chats: {', '.join(store)}")
    else:
        name = max(store, key=lambda n: float(store[n].get("last_used", 0)))
    count = max(1, min(int(count), 10))
    pairs: list[tuple[str, str]] = []
    md = app_dir() / "claude_chats" / f"{safe_filename(name)}.md"
    try:
        pairs = read_exchanges(md.read_text(encoding="utf-8"))
    except OSError:
        pass
    if not pairs:  # fall back to Claude's own transcript of the session
        from .claude_history import _read_tail_lines, encode_project, parse_transcript, projects_dir

        f = projects_dir() / encode_project(store[name].get("cwd") or str(chat_dir())) / f"{store[name]['session_id']}.jsonl"
        try:
            msgs, _ = parse_transcript(_read_tail_lines(f))
        except OSError:
            msgs = []
        lines = [f"{'you' if r == 'user' else 'claude'}: {' '.join(t.split())[:400]}" for r, t in msgs[-count * 2:]]
        return f"Chat {name}:\n" + ("\n".join(lines) if lines else "(no messages saved)")
    lines = [f"you: {' '.join(q.split())[:300]}\nclaude: {' '.join(a.split())[:400]}" for q, a in pairs[-count:]]
    return f"Chat {name}:\n" + "\n".join(lines)
