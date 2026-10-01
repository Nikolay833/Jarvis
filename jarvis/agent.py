"""Conversation loop: memory, tool calls, confirmations."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable

from . import safety
from .llm import LLM, LLMError
from .tools.registry import Registry

log = logging.getLogger("jarvis.agent")

MAX_TOOL_RESULT_CHARS = 4000

SYSTEM_PROMPT = """You are Jarvis, a personal assistant living on the user's Windows PC.
Persona: a polite, dry-witted British butler. Address the user as "sir".
Your replies are spoken aloud, so: one to three short sentences, plain words, no markdown, no lists, no emoji, no URLs read out in full.
Use the tools to act on the PC; never claim to have done something you did not do with a tool.
If a request is ambiguous and a wrong guess could do harm, ask one short question first.
Risky tools (deleting, installing, stopping things, running Claude Code) ask the user for approval automatically; just call the tool.
For coding work in a project folder, use claude_code_run. For simple PC questions, use system_info or run_powershell.
After using tools, report the outcome briefly. If a tool fails, say so plainly and suggest the next step.
Current date and time: {now}."""

_YES = {"yes", "yeah", "yep", "yup", "sure", "proceed", "approve", "approved", "confirm", "confirmed",
        "affirmative", "ok", "okay", "go", "do", "please"}
_NO = {"no", "nope", "nah", "stop", "cancel", "deny", "denied", "negative", "don't", "dont", "abort", "never"}


def parse_yes_no(text: str) -> bool | None:
    """True for yes, False for no, None if unclear. 'no' wins over 'yes' when both appear."""
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return None
    if any(w in _NO for w in words):
        return False
    if any(w in _YES for w in words) or "go ahead" in " ".join(words):
        return True
    return None


class Confirmer:
    """Asks for approval of a risky action by voice and/or bus click. Timeout means no."""

    def __init__(self, bus: Any, speak: Callable[[str], Awaitable[None]], timeout: float = 30.0,
                 listen: Callable[[], Awaitable[str]] | None = None) -> None:
        self.bus = bus
        self.speak = speak
        self.timeout = timeout
        self.listen = listen  # records one utterance and returns its text
        self._pending: dict[str, asyncio.Future[bool]] = {}

    @property
    def pending(self) -> bool:
        return any(not f.done() for f in self._pending.values())

    def resolve(self, confirm_id: str, approved: bool) -> bool:
        fut = self._pending.get(confirm_id)
        if fut and not fut.done():
            fut.set_result(bool(approved))
            return True
        return False

    def resolve_text(self, text: str) -> bool:
        """Resolve the newest pending confirmation from typed/spoken text. False if unclear."""
        verdict = parse_yes_no(text)
        if verdict is None:
            return False
        for cid in reversed(list(self._pending)):
            if self.resolve(cid, verdict):
                return True
        return False

    async def _voice_loop(self, fut: asyncio.Future[bool]) -> None:
        assert self.listen is not None
        while not fut.done():
            text = await self.listen()
            verdict = parse_yes_no(text or "")
            if verdict is not None and not fut.done():
                fut.set_result(verdict)

    async def ask(self, summary: str) -> bool:
        cid = uuid.uuid4().hex[:8]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending[cid] = fut
        await self.bus.emit("confirm", id=cid, summary=summary)
        voice_task: asyncio.Task | None = None
        try:
            await self.speak(f"Sir, I'm about to {summary}. Shall I proceed?")
            if self.listen is not None and not fut.done():
                voice_task = asyncio.create_task(self._voice_loop(fut))
            try:
                approved = await asyncio.wait_for(asyncio.shield(fut), self.timeout)
            except asyncio.TimeoutError:
                approved = False
        finally:
            if voice_task:
                voice_task.cancel()
                await asyncio.gather(voice_task, return_exceptions=True)
            self._pending.pop(cid, None)
            if not fut.done():
                fut.cancel()
        await self.bus.emit("confirm_resolved", id=cid, approved=approved)
        return approved


class Agent:
    def __init__(self, llm: LLM, registry: Registry, bus: Any, confirmer: Confirmer,
                 max_steps: int = 8, max_history: int = 30) -> None:
        self.llm = llm
        self.registry = registry
        self.bus = bus
        self.confirmer = confirmer
        self.max_steps = max_steps
        self.max_history = max_history
        self.sessions: dict[str, list[dict[str, Any]]] = {}

    # ---- memory ----------------------------------------------------------
    def history(self, session: str) -> list[dict[str, Any]]:
        return self.sessions.setdefault(session, [])

    def reset(self, session: str = "default") -> None:
        self.sessions.pop(session, None)

    def trim(self, session: str) -> None:
        """Keep the newest messages, always starting at a user message (no orphan tool results)."""
        msgs = self.history(session)
        if len(msgs) <= self.max_history:
            return
        msgs = msgs[-self.max_history:]
        for i, m in enumerate(msgs):
            if m.get("role") == "user":
                msgs = msgs[i:]
                break
        else:
            msgs = []
        self.sessions[session] = msgs

    def _system(self) -> dict[str, str]:
        return {"role": "system", "content": SYSTEM_PROMPT.format(now=datetime.now().strftime("%A %d %B %Y, %H:%M"))}

    # ---- main loop -------------------------------------------------------
    async def handle(self, text: str, session: str = "default") -> str:
        msgs = self.history(session)
        msgs.append({"role": "user", "content": text})
        self.trim(session)
        msgs = self.history(session)
        tools = self.registry.schemas()
        reply = ""
        try:
            for _ in range(self.max_steps):
                resp = await self.llm.chat([self._system(), *msgs], tools)
                assistant: dict[str, Any] = {"role": "assistant", "content": resp.content}
                if resp.raw_tool_calls:
                    assistant["tool_calls"] = resp.raw_tool_calls
                elif resp.tool_calls:
                    assistant["tool_calls"] = [
                        {"function": {"name": c.name, "arguments": c.arguments}} for c in resp.tool_calls
                    ]
                msgs.append(assistant)
                if not resp.tool_calls:
                    reply = resp.content or "Very good, sir."
                    break
                for call in resp.tool_calls:
                    result = await self._run_tool(call.name, call.arguments)
                    msgs.append({"role": "tool", "tool_name": call.name, "content": result})
            else:
                reply = "My apologies, sir, that took more steps than I allow myself. Shall I carry on?"
                msgs.append({"role": "assistant", "content": reply})
        except LLMError as exc:
            log.error("LLM failure: %s", exc)
            reply = "I am afraid my thinking engine is unavailable, sir. Is Ollama running?"
            self._drop_dangling_user(session)
        self.trim(session)
        return reply

    def _drop_dangling_user(self, session: str) -> None:
        msgs = self.history(session)
        while msgs and msgs[-1].get("role") != "assistant":
            msgs.pop()

    async def _run_tool(self, name: str, args: dict[str, Any]) -> str:
        tool = self.registry.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        assessment = safety.classify_call(name, args, tool.risk)
        if assessment.is_risky:
            summary = safety.describe_call(name, args)
            if not await self.confirmer.ask(summary):
                return "The user declined (or did not answer in time). The action was NOT performed."
        log.info("tool %s %s", name, json.dumps(args, default=str)[:200])
        result = await self.registry.call(name, args)
        if len(result) > MAX_TOOL_RESULT_CHARS:
            result = result[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"
        return result
