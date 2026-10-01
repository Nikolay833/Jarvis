"""Conversation loop: memory, tool calls, confirmations."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable

from . import safety
from .audio.tts import SentenceBuffer
from .llm import LLM, LLMError, LLMResponse
from .tools.registry import Registry

log = logging.getLogger("jarvis.agent")

MAX_TOOL_RESULT_CHARS = 4000

# Fully static (no clock): the Ollama KV cache can reuse system prompt + tool schemas every turn.
# The current date/time goes in the newest user message instead (see Agent._with_now).
SYSTEM_PROMPT = """You are Jarvis, a personal assistant living on the user's Windows PC.
Persona: a polite, dry-witted British butler. Address the user as "sir".
Your replies are spoken aloud: answer in one or two short sentences unless the user asks for detail. Plain words only: no markdown, no lists, no emoji, no URLs read out in full.
The newest user message starts with a line like "[Now: Wednesday 01 October 2026, 18:42]" giving the current date and time; use it when needed and never read it back unprompted.
Use the tools to act on the PC; never claim to have done something you did not do with a tool.
If a request is ambiguous and a wrong guess could do harm, ask one short question first.
Risky tools (deleting, installing, stopping things, running Claude Code) ask the user for approval automatically; just call the tool.
For coding work in a project folder, use claude_code_run. For simple PC questions, use system_info or run_powershell.
After using tools, report the outcome in one short sentence. If a tool fails, say so plainly and suggest the next step."""

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


class _Out:
    """Lazy speaker stream for one turn: opened on the first sentence, so silent turns open none."""

    def __init__(self, speaker: Any) -> None:
        self.speaker = speaker
        self.stream: Any = None
        self.pushed = 0
        self.first_audio_at: float | None = None

    def push(self, sentence: str) -> None:
        if self.stream is None:
            self.stream = self.speaker.start_stream()
        self.pushed += 1
        self.stream.push(sentence)

    async def finish(self) -> None:
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                await stream.finish()
            finally:
                if self.first_audio_at is None:
                    self.first_audio_at = getattr(stream, "first_audio_at", None)


class Agent:
    def __init__(self, llm: LLM, registry: Registry, bus: Any, confirmer: Confirmer,
                 max_steps: int = 8, max_history: int = 30, stream_replies: bool = True) -> None:
        self.llm = llm
        self.registry = registry
        self.bus = bus
        self.confirmer = confirmer
        self.max_steps = max_steps
        self.max_history = max_history
        self.stream_replies = stream_replies
        self.timing: dict[str, float | None] = {}
        self.first_audio_at: float | None = None  # perf_counter of the first spoken audio, if streamed
        self.streamed = False  # True if the last handle() already spoke its reply via the stream
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
        return {"role": "system", "content": SYSTEM_PROMPT}

    @staticmethod
    def now_prefix(now: datetime | None = None) -> str:
        return (now or datetime.now()).strftime("[Now: %A %d %B %Y, %H:%M]\n")

    @staticmethod
    def _with_now(msgs: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
        """Copy of `msgs` with `prefix` on the newest user message. History itself stays unprefixed."""
        out = list(msgs)
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                out[i] = {**out[i], "content": prefix + str(out[i].get("content", ""))}
                break
        return out

    def add_exchange(self, user_text: str, reply: str, session: str = "default") -> None:
        """Record a turn answered without the LLM (fast path) so follow-ups have context."""
        msgs = self.history(session)
        msgs.append({"role": "user", "content": user_text})
        if reply:
            msgs.append({"role": "assistant", "content": reply})
        else:
            msgs.pop()
        self.trim(session)

    async def _step(self, wire: list[dict[str, Any]], tools: list[dict[str, Any]],
                    out: "_Out | None") -> LLMResponse:
        """One LLM step. Streams when possible, speaking each finished sentence of a plain reply."""
        chat_stream = getattr(self.llm, "chat_stream", None)
        if chat_stream is None or out is None or not self.stream_replies:
            resp = await self.llm.chat(wire, tools)
            if self.timing.get("first_token") is None:
                self.timing["first_token"] = time.perf_counter()
            return resp
        buf = SentenceBuffer()
        final: LLMResponse | None = None
        try:
            async for item in chat_stream(wire, tools):
                if isinstance(item, LLMResponse):
                    final = item
                    break
                if self.timing.get("first_token") is None:
                    self.timing["first_token"] = time.perf_counter()
                for sentence in buf.feed(item):
                    out.push(sentence)
        except LLMError:
            if out.pushed:
                raise  # already speaking: a retry would repeat words
            log.warning("streaming failed, falling back to non-streaming chat", exc_info=True)
            return await self.llm.chat(wire, tools)
        if final is None:
            raise LLMError("stream ended without a final message")
        if final.tool_calls:
            buf.discard()  # tool step: do not speak the unfinished remainder
        else:
            for sentence in buf.flush():
                out.push(sentence)
        return final

    # ---- main loop -------------------------------------------------------
    async def handle(self, text: str, session: str = "default", speaker: Any = None) -> str:
        """Run one turn and return the reply text.

        With a `speaker` (needs `start_stream()`), the reply is spoken sentence by sentence while
        the LLM generates; `self.streamed` is then True and the caller must not speak it again.
        """
        msgs = self.history(session)
        msgs.append({"role": "user", "content": text})
        self.trim(session)
        msgs = self.history(session)
        tools = self.registry.schemas()
        prefix = self.now_prefix()
        out = _Out(speaker) if speaker is not None else None
        self.timing = {"start": time.perf_counter(), "first_token": None}
        self.streamed = False
        self.first_audio_at = None
        reply = ""
        try:
            for step in range(1, self.max_steps + 1):
                t0 = time.perf_counter()
                resp = await self._step([self._system(), *self._with_now(msgs, prefix)], tools, out)
                took = time.perf_counter() - t0
                if resp.tool_calls:
                    log.info("llm step %d: tool calls %s (%.1f s)", step, [c.name for c in resp.tool_calls], took)
                else:
                    log.info("llm step %d: reply (%.1f s)", step, took)
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
                    if out is not None and not out.pushed:
                        out.push(reply)  # empty content: speak the fallback
                    break
                if out is not None:
                    await out.finish()  # free the speaker (a confirmation may need it)
                for call in resp.tool_calls:
                    result = await self._run_tool(call.name, call.arguments)
                    msgs.append({"role": "tool", "tool_name": call.name, "content": result})
            else:
                reply = "My apologies, sir, that took more steps than I allow myself. Shall I carry on?"
                msgs.append({"role": "assistant", "content": reply})
                if out is not None:
                    out.push(reply)
        except LLMError as exc:
            log.error("LLM failure: %s", exc)
            reply = "I am afraid my thinking engine is unavailable, sir. Is Ollama running?"
            self._drop_dangling_user(session)
            if out is not None:
                out.push(reply)
        finally:
            if out is not None:
                await out.finish()
                self.streamed = out.pushed > 0
                self.first_audio_at = out.first_audio_at
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
