"""Voice control of Claude Code sessions: list, open, continue, start, check status, ask.

Sessions are learned from Claude's own transcripts (see jarvis/claude_sessions.py). Terminals are opened
through claude_terminal.open_claude_terminal.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from .. import claude_sessions as cs
from ..claude_watch import watch
from . import claude_chat
from .claude_code import resolve_binary
from .claude_terminal import open_claude_terminal
from .context import ctx
from .registry import ToolError, tool

OFFER_TTL = 300.0  # seconds a "which one?" list stays valid for the follow-up choice
ASK_TIMEOUT = 180.0
MAX_ANSWER_CHARS = 2000
_offer: dict[str, object] = {"ids": [], "at": 0.0}


# ---- helpers --------------------------------------------------------------------------------------------------
def _project_folder(project: str, sessions: list[cs.Session]) -> str:
    """Folder for what the user called the project; ToolError if it is unknown. Empty in, empty out."""
    if not project.strip():
        return ""
    folder = cs.resolve_project(project, sessions)
    if folder is None:
        raise ToolError(f"I don't know a project called '{project.strip()}'")
    return folder


def _spoken_list(ranked: list[cs.Ranked]) -> str:
    return ", ".join(
        cs.format_line(i, r.session).rstrip(".") for i, r in enumerate(ranked, 1))


def _pick(topic: str, project: str, choice: int = 0) -> tuple[cs.Session | None, str]:
    """Choose one session. Returns (session, "") or (None, question to ask the user)."""
    sessions = cs.scan_sessions()
    if choice > 0 and time.time() - float(_offer["at"]) < OFFER_TTL and _offer["ids"]:
        ids = list(_offer["ids"])  # type: ignore[arg-type]
        if choice > len(ids):
            raise ToolError(f"there are only {len(ids)} sessions in that list")
        found = next((s for s in sessions if s.id == ids[choice - 1]), None)
        if found:
            return found, ""
    _project_folder(project, sessions)  # fail early on an unknown project
    ranked = cs.find_sessions(topic, project or None, limit=3, sessions=sessions)
    if not ranked:
        where = f" in {project.strip()}" if project.strip() else ""
        what = f" about '{topic.strip()}'" if topic.strip() else ""
        raise ToolError(f"I found no Claude session{what}{where}")
    if choice > 0:
        if choice > len(ranked):
            raise ToolError(f"there are only {len(ranked)} matching sessions")
        return ranked[choice - 1].session, ""
    if topic.strip() and cs.is_ambiguous(ranked):
        _offer["ids"] = [r.session.id for r in ranked]
        _offer["at"] = time.time()
        return None, f"I found {len(ranked)} sessions: {_spoken_list(ranked)}. Which one, sir?"
    return ranked[0].session, ""


def derive_name(prompt: str, words: int = 4) -> str:
    """Short session name (2 to 4 words) from the first words of a prompt; '' if nothing usable."""
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", prompt or "")
    skip = {"please", "can", "could", "you", "would", "will", "to", "the", "a", "an", "and", "i", "me", "my", "we",
            "us", "now", "just", "claude", "jarvis", "want", "need", "like", "that", "this", "it", "of", "for"}
    kept = [t for t in toks if t.lower() not in skip][:words]
    return " ".join(kept).lower() if len(kept) >= 1 else ""


# ---- tools ----------------------------------------------------------------------------------------------------
@tool("List the user's recent Claude Code sessions (newest first) in a spoken-friendly way: topic, project, how "
      "long ago. Use for 'list my Claude sessions' or 'what sessions do I have in <project>'.")
def claude_sessions(project: str = "", count: int = 5) -> str:
    """List recent Claude sessions.

    Args:
        project: Project name or folder to limit the list to. Empty means all projects.
        count: How many sessions to list.
    """
    count = max(1, min(int(count), 10))
    sessions = cs.scan_sessions()
    _project_folder(project, sessions)
    ranked = cs.find_sessions("", project or None, limit=count, sessions=sessions)
    if not ranked:
        return "I found no Claude sessions" + (f" in {project.strip()}." if project.strip() else ".")
    head = f"Your latest {len(ranked)} Claude sessions:" if len(ranked) > 1 else "Your latest Claude session:"
    return head + "\n" + cs.format_list([r.session for r in ranked])


@tool("Open an existing Claude Code session in a terminal, found by topic and/or project (resumes it, optionally "
      "sending a prompt). Empty topic with a project means the most recent session there. If several sessions "
      "match it returns a question listing them: ask it aloud, then call again with choice=N.")
def claude_open_session(topic: str = "", project: str = "", prompt: str = "", choice: int = 0) -> str:
    """Resume a Claude session.

    Args:
        topic: What the session was about, e.g. "login bug", or "latest" for the most recent one. Leave empty only if the user named no topic; the tool then asks which session.
        project: Project name or folder, e.g. "Jarvis".
        prompt: Optional message to send into the session right away.
        choice: 1, 2 or 3 to pick from the list offered by an earlier call; 0 otherwise.
    """
    latest = topic.strip().lower() in _LATEST_WORDS
    if latest:
        topic = ""
    if not topic.strip() and not project.strip() and not int(choice or 0) and not latest:
        newest = cs.scan_sessions()[:1]
        if not newest:
            return "I found no Claude sessions, sir. Shall I start a new one?"
        n = newest[0]
        return (f"Which session, sir? Your latest is '{n.label}' in {n.project_name}, "
                f"{cs.humanize_age(time.time() - n.last_activity)}. Shall I open that one, or start a new one?")
    s, question = _pick(topic, project, int(choice))
    if s is None:
        return question
    if not Path(s.project_dir).is_dir():
        raise ToolError(f"the project folder of that session no longer exists: {s.project_dir}")
    open_claude_terminal(s.project_dir, prompt, ["--resume", s.id])
    _offer["ids"] = []
    return f"Opened the session '{s.label}' in {s.project_name}" + (" with your message." if prompt.strip() else ".")


_LATEST_WORDS = {"last", "latest", "recent", "most recent", "previous", "newest", "last one", "the last one"}


@tool("Continue the most recent Claude Code conversation in a project (like 'continue where I left off in "
      "Jarvis'), in a terminal, optionally sending a prompt.")
def claude_continue(project: str = "", prompt: str = "") -> str:
    """Continue the latest Claude session in a project.

    Args:
        project: Project name or folder. Empty means the project of the most recent session.
        prompt: Optional message to send right away.
    """
    sessions = cs.scan_sessions()
    folder = _project_folder(project, sessions)
    if not folder and sessions:
        folder = sessions[0].project_dir
    if not folder:
        raise ToolError("I found no Claude sessions to continue")
    open_claude_terminal(folder, prompt, ["--continue"])
    return f"Continuing the last Claude session in {cs.basename(folder)}" + (" with your message." if prompt.strip() else ".")


@tool("Start a NEW Claude Code session in a terminal, in a project (or the home folder), optionally with a first "
      "prompt. Use for 'new Claude session for X' or 'ask Claude to do X in project Y'. The session gets a short name.")
def claude_new_session(project: str = "", prompt: str = "", name: str = "") -> str:
    """Start a new Claude session.

    Args:
        project: Project name or folder. Empty means the home folder.
        prompt: What Claude should do, in the user's words.
        name: Optional short session name; derived from the prompt when empty.
    """
    sessions = cs.scan_sessions()
    folder = _project_folder(project, sessions)
    name = name.strip() or derive_name(prompt)
    args = ["--name", name] if name else []
    where = open_claude_terminal(folder, prompt, args)
    return ("Started a new Claude session" + (f" named '{name}'" if name else "")
            + f" in {cs.basename(str(where))}" + (" with your request." if prompt.strip() else "."))


@tool("Report what Claude Code is doing or whether it finished: for the matching or most recently active session "
      "gives project, how long ago it was active, busy or waiting or done, what Claude last said, the last tool, "
      "and any pending permission request. Use for 'what did Claude say', 'what is Claude doing', 'is Claude done'.")
def claude_status(project: str = "", topic: str = "") -> str:
    """Status of a Claude session.

    Args:
        project: Project name or folder. Empty means any project.
        topic: What the session is about. Empty means the most recently active one.
    """
    sessions = cs.scan_sessions()
    _project_folder(project, sessions)
    ranked = cs.find_sessions(topic, project or None, limit=1, sessions=sessions)
    if not ranked:
        return "I found no Claude session to report on."
    s = ranked[0].session
    now = time.time()
    pend = watch.pending_for(s.id)
    if pend is not None:
        state = "waiting for your permission"
    elif cs.looks_busy(s, now):
        state = "busy working"
    else:
        state = "idle, its last turn is finished" if s.last_kind == "assistant_text" else "idle"
    parts = [f"Claude in {s.project_name} ('{s.label}') is {state}; last active {cs.humanize_age(now - s.last_activity)}."]
    if s.last_tool:
        parts.append(f"Last tool: {s.last_tool}.")
    said = cs.first_sentences(s.last_assistant_text, 2, 45)
    if said:
        parts.append(f"Last said: {said}")
    if pend is not None and pend.message:
        parts.append(f"Permission request: {cs.speakable(pend.message)[:200]}")
    return "\n".join(parts)


@tool("Ask an existing Claude session a question in the background and read its answer back. Use ONLY when the "
      "user wants the answer spoken ('ask that session X and tell me'); otherwise prefer claude_open_session.")
async def claude_ask_session(topic: str = "", project: str = "", question: str = "") -> str:
    """Ask a Claude session a question headlessly.

    Args:
        topic: What the session is about. Empty means the most recent one.
        project: Project name or folder.
        question: The question, in the user's words.
    """
    if not question.strip():
        raise ToolError("what should I ask the session?")
    s, ask = _pick(topic, project)
    if s is None:
        return ask
    if not Path(s.project_dir).is_dir():
        raise ToolError(f"the project folder of that session no longer exists: {s.project_dir}")
    binary = resolve_binary(ctx.config.claude_code.binary)
    cmd = [binary, "-p", "--resume", s.id, "--output-format", "json"]
    code, out, err = await claude_chat.run_cli(cmd, s.project_dir, question.strip(), ASK_TIMEOUT)
    answer, _sid, is_err = claude_chat.parse_result(out)
    if code != 0 or is_err or not answer:
        raise ToolError((answer or err.strip() or "Claude gave no answer")[:200])
    return f"Answer from '{s.label}' in {s.project_name}: " + cs.speakable(answer)[:MAX_ANSWER_CHARS]
