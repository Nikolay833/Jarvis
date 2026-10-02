"""Conversation loop: memory, tool calls, confirmations."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable

from . import safety
from .audio.tts import SentenceBuffer
from . import paths
from .llm import LLM, LLMError, LLMResponse, ToolCallStreamFilter, extract_text_tool_calls
from .tools.registry import Registry

log = logging.getLogger("jarvis.agent")

MAX_TOOL_RESULT_CHARS = 4000

# Fully static (no clock): the Ollama KV cache can reuse system prompt + tool schemas every turn.
# The current date/time goes in the newest user message instead (see Agent._with_now).
SYSTEM_PROMPT = """You are Jarvis, a personal assistant living on the user's Windows PC.
Persona: warm, quick, dry-witted and British, like a trusted friend who happens to run the house. You may call the user "sir", but at most once in a reply and often not at all.
Your replies are spoken aloud, so talk like a person in a real conversation: contractions, natural rhythm, varied wording (never the same stock phrase twice in a row), a little personality. For commands, one short sentence is plenty. For questions, chat, advice or feelings, talk freely and naturally, a few sentences, like you would to a friend; ask a follow-up question when it genuinely helps. Plain spoken words only: no markdown, no bullet points, no emoji, no URLs read out in full, no numbered "1." lists; when listing things say "the first is…, then…, and finally…".
The newest user message starts with a line like "[Now: Wednesday 01 October 2026, 18:42]" giving the current date and time; use it when needed and never read it back unprompted.
The newest user message may also have a line like "[Known about the user: ...]" listing facts you remembered; use them naturally and never read the list back unprompted.
Use the tools to act on the PC; never claim to have done something you did not do with a tool.
Memory: when the user says "remember X", call remember with a short fact (e.g. "Prefers Spotify for music"). When they clearly state a lasting personal fact or preference ("I prefer Spotify", "my main project is Jarvis", "call me Nikolay"), also call remember with source "inferred" and acknowledge in a few words; do not store moods, one-off requests or guesses, and never store passwords, keys or other secrets. recall answers "what do you remember", forget removes a fact.
Timers and reminders (announced out loud when due): set_timer, set_reminder(text, when) with the user's own words for when ("in 20 minutes", "at 6pm", "tomorrow at 9"; ask when if it is missing), list_reminders, cancel_reminder. For "give me my briefing" call briefing and say its text as given.
If a request is ambiguous and a wrong guess could do harm, ask one short question first.
Risky tools (deleting, installing, stopping things, running Claude Code) ask the user for approval automatically; just call the tool.
For coding work in a project folder, use claude_code_run. For simple PC questions, use system_info or run_powershell.
When the user asks you to do something on the PC, call the tool in this same response; never say you will do it later. Only reply in text after the tool results arrive, reporting what actually happened.
To read out the last messages of a Claude conversation word for word, use claude_code_history. To make folders or files, use create_folder and write_file. To find files or folders on the PC, use find_on_pc (where "~" for the whole user folder, or desktop/documents/downloads) and speak its answer without reading full paths aloud; if the user says "search for X" without saying web or PC, ask which they mean.
For music use music_control (pause, resume, next, previous, stop), spotify_now_playing and spotify_play; for windows use window_action (minimize, maximize, restore, focus, close) and minimize_all; for Chrome profiles or searching in Chrome use open_chrome (chrome_profiles lists profiles).
Only use Claude tools when the user mentions Claude or a Claude session; a bare "open X" means an app, folder or file, and if X is unclear ask what to open. Claude Code sessions: to open or resume an existing session ("open the login bug session in Jarvis") use claude_open_session(topic, project); for "my last/latest Claude session" pass topic "latest"; if it returns a question listing sessions, say it exactly as given, and after the user answers call it again with choice=N. "Continue where I left off in X" -> claude_continue. "New Claude session for X" or "ask Claude to do X in project Y" -> claude_new_session (the task goes in prompt). "What did Claude say", "what is Claude doing", "is Claude done", "did Claude finish" -> claude_status (never claude_code_status, which only covers jobs Jarvis started). "List my Claude sessions" -> claude_sessions. "Ask that session X and tell me the answer" -> claude_ask_session (only when the answer should be spoken; otherwise open the session). Confirm in one short, natural sentence, e.g. "Opening the login bug session now."
For a plain "ask Claude" or "tell Claude" request with no project or session, use claude_terminal (opens a visible terminal running Claude with the request), then say in one short sentence that Claude is on it. Only if the user explicitly wants you to read Claude's answer back, use claude_chat instead and relay the answer faithfully; claude_chat_list and claude_chat_history manage those chats. claude_code_run is for silent background coding jobs.
After using tools, report the outcome in one short sentence. If a tool fails, say so plainly and suggest the next step."""

NUDGE = "(system) You announced an action but did not call a tool. Call the appropriate tool now. Do not reply with text only."
MAX_NUDGES = 2

# Slow tools: say this short line before running, so the user is not left in silence.
# Tools that open or message Claude. They only run when the user actually talked about Claude
# (this turn, or the previous Jarvis reply was about Claude, e.g. "which session?" -> "the second").
CLAUDE_ACTION_TOOLS = {"claude_open_session", "claude_continue", "claude_new_session", "claude_terminal",
                       "claude_ask_session", "claude_chat", "claude_chat_new", "claude_code_run"}
_CLAUDE_WORDS = re.compile(r"\b(claude|claude's|session|sessions)\b", re.IGNORECASE)
CLAUDE_GUARD_MSG = ("Not done: the user did not mention Claude, so do not open or message Claude. "
                    "Ask the user in one short sentence what they want, e.g. 'Open what, sir?'")

SLOW_TOOL_NOTICE = {"claude_chat": "Asking Claude, sir.", "claude_chat_new": "Asking Claude, sir.",
                    "claude_ask_session": "Asking that session, sir."}

_LEAD = r"(?:(?:very well|certainly|of course|right|sure|ok(?:ay)?|alright|understood|absolutely|splendid)[,.!]?\s+)?(?:sir[,.!]?\s+)?"
_ANNOUNCE = re.compile(
    r"(?:^|[.!;:]\s+|\n)" + _LEAD +
    r"(?:i['\u2019]ll|i will|i shall|i['\u2019]m going to|i am going to|i['\u2019]m about to|let me|let['\u2019]s|"
    r"one moment|just a moment|one second|just a second|right away|on it|checking|creating|deleting|opening|"
    r"making|searching|looking|fetching|reading|writing|starting|launching|running|allow me|"
    r"i['\u2019]m (?:checking|creating|deleting|opening|making|searching|looking|fetching|reading|writing|starting)"
    r")\b(?!\s+(?:know|be\b|need|have\b|ask|tell))",
    re.IGNORECASE)


def looks_like_announcement(text: str) -> bool:
    """True if `text` promises an action instead of reporting a result (and is not a question)."""
    text = (text or "").strip()
    if not text or text.endswith("?"):
        return False
    return _ANNOUNCE.search(text) is not None


@functools.lru_cache(maxsize=1)
def full_system_prompt() -> str:
    """Static system prompt incl. the user's folders; computed once so the KV cache stays valid."""
    try:
        return SYSTEM_PROMPT + "\n" + paths.folder_hint()
    except Exception:  # noqa: BLE001
        return SYSTEM_PROMPT

_YES = {"yes", "yeah", "yep", "yup", "sure", "proceed", "approve", "approved", "confirm", "confirmed",
        "affirmative", "ok", "okay", "absolutely", "certainly", "correct", "accept", "accepted"}
_YES_PHRASES = ("go ahead", "do it", "go for it", "of course", "carry on", "make it so", "sounds good")
_NO = {"no", "nope", "nah", "stop", "cancel", "deny", "denied", "negative", "don't", "dont", "abort", "never",
       "not", "wait", "reject", "rejected", "decline", "declined"}
_NO_PHRASES = ("hold on", "hang on", "leave it", "forget it")


def parse_yes_no(text: str) -> bool | None:
    """True for yes, False for no, None if unclear. Any no-word wins, so "do not do it" is no."""
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return None
    joined = " ".join(words)
    if any(w in _NO for w in words) or any(p in joined for p in _NO_PHRASES):
        return False
    if any(w in _YES for w in words) or any(p in joined for p in _YES_PHRASES):
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
            if fut.done():
                return
            if verdict is not None:
                fut.set_result(verdict)
            elif text:
                await self.speak("Sorry sir, was that a yes or a no?")

    async def ask(self, summary: str) -> bool:
        cid = uuid.uuid4().hex[:8]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending[cid] = fut
        await self.bus.emit("confirm", id=cid, summary=summary)
        voice_task: asyncio.Task | None = None
        try:
            await self.speak(f"Sir, I'm about to {summary}. Shall I proceed? Say yes or no.")
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
                 max_steps: int = 8, max_history: int = 30, stream_replies: bool = True,
                 memory_block: Callable[[str], str] | None = None) -> None:
        self.llm = llm
        self.registry = registry
        self.bus = bus
        self.confirmer = confirmer
        self.max_steps = max_steps
        self.max_history = max_history
        self.stream_replies = stream_replies
        # user text -> "[Known about the user: ...]\n" (or ""): goes on the newest user message only
        self.memory_block = memory_block
        self.timing: dict[str, float | None] = {}
        self.first_audio_at: float | None = None  # perf_counter of the first spoken audio, if streamed
        self.streamed = False  # True if the last handle() already spoke its reply via the stream
        self.sessions: dict[str, list[dict[str, Any]]] = {}
        self._turn_text = ""  # the user's words this turn (for the Claude guard)
        self._prev_reply = ""  # Jarvis's previous reply (a question about Claude lets "the second one" through)

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
        return {"role": "system", "content": full_system_prompt()}

    @staticmethod
    def now_prefix(now: datetime | None = None) -> str:
        return (now or datetime.now()).strftime("[Now: %A %d %B %Y, %H:%M]\n")

    def known_prefix(self, text: str) -> str:
        if self.memory_block is None:
            return ""
        try:
            return self.memory_block(text) or ""
        except Exception:  # noqa: BLE001 - memory must never break a turn
            log.warning("memory lookup failed", exc_info=True)
            return ""

    @staticmethod
    def _with_now(msgs: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
        """Copy of `msgs` with `prefix` on the newest user message. History itself stays unprefixed."""
        out = [{k: v for k, v in m.items() if k != "_nudge"} if "_nudge" in m else m for m in msgs]
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user" and "_nudge" not in msgs[i]:
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
            return self._recover_calls(resp)
        buf = SentenceBuffer()
        guard = ToolCallStreamFilter()
        final: LLMResponse | None = None
        try:
            async for item in chat_stream(wire, tools):
                if isinstance(item, LLMResponse):
                    final = item
                    break
                if self.timing.get("first_token") is None:
                    self.timing["first_token"] = time.perf_counter()
                for sentence in buf.feed(guard.feed(item)):
                    out.push(sentence)
        except LLMError:
            if out.pushed:
                raise  # already speaking: a retry would repeat words
            log.warning("streaming failed, falling back to non-streaming chat", exc_info=True)
            return self._recover_calls(await self.llm.chat(wire, tools))
        if final is None:
            raise LLMError("stream ended without a final message")
        final = self._recover_calls(final)
        if final.tool_calls:
            buf.discard()  # tool step: do not speak the unfinished remainder
        else:
            for sentence in buf.feed(guard.flush()):
                out.push(sentence)
            for sentence in buf.flush():
                out.push(sentence)
        return final

    def _recover_calls(self, resp: LLMResponse) -> LLMResponse:
        """Tool calls the model wrote as text become real calls (known tool names only)."""
        if resp.tool_calls or not resp.content:
            return resp
        content, calls = extract_text_tool_calls(resp.content, self.registry.names())
        if not calls:
            return resp
        log.info("recovered tool call(s) written as text: %s", [c.name for c in calls])
        return LLMResponse(content=content, tool_calls=calls, raw_tool_calls=[])

    # ---- main loop -------------------------------------------------------
    def _claude_in_context(self) -> bool:
        if _CLAUDE_WORDS.search(self._turn_text):
            return True
        return bool(self._prev_reply and _CLAUDE_WORDS.search(self._prev_reply))

    async def handle(self, text: str, session: str = "default", speaker: Any = None) -> str:
        """Run one turn and return the reply text.

        With a `speaker` (needs `start_stream()`), the reply is spoken sentence by sentence while
        the LLM generates; `self.streamed` is then True and the caller must not speak it again.
        """
        msgs = self.history(session)
        self._turn_text = text
        msgs.append({"role": "user", "content": text})
        self.trim(session)
        msgs = self.history(session)
        tools = self.registry.schemas()
        prefix = self.now_prefix() + self.known_prefix(text)
        out = _Out(speaker) if speaker is not None else None
        self.timing = {"start": time.perf_counter(), "first_token": None}
        self.streamed = False
        self.first_audio_at = None
        reply = ""
        executed: list[str] = []
        nudged: list[dict[str, Any]] = []  # nudge exchanges, removed from history when the turn ends
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
                if (not resp.tool_calls and tools and len(nudged) < 2 * MAX_NUDGES
                        and looks_like_announcement(resp.content)):
                    log.info("nudged model to call a tool (reply was: %s)", resp.content[:200])
                    nudge = {"role": "user", "content": NUDGE, "_nudge": True}
                    msgs.append(nudge)
                    nudged += [assistant, nudge]
                    continue
                if not resp.tool_calls:
                    reply = resp.content or ("Done, sir." if executed else "Apologies sir, I wasn't able to do that.")
                    if out is not None and not out.pushed:
                        out.push(reply)  # empty content: speak the fallback
                    break
                if out is not None:
                    await out.finish()  # free the speaker (a confirmation may need it)
                for call in resp.tool_calls:
                    executed.append(call.name)
                    result = await self._run_tool(call.name, call.arguments, out)
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
            if nudged:
                msgs[:] = [m for m in msgs if not any(m is n for n in nudged)]
            log.info("turn done: tools called %s", executed if executed else "none (no tools called)")
            if out is not None:
                await out.finish()
                self.streamed = out.pushed > 0
                self.first_audio_at = out.first_audio_at
        self.trim(session)
        self._prev_reply = reply
        return reply

    def _drop_dangling_user(self, session: str) -> None:
        msgs = self.history(session)
        while msgs and msgs[-1].get("role") != "assistant":
            msgs.pop()

    def _state(self, state: str) -> None:
        emit = getattr(self.bus, "emit_nowait", None)
        if emit is not None:
            emit("state", state=state)

    async def _run_tool(self, name: str, args: dict[str, Any], out: "_Out | None" = None) -> str:
        tool = self.registry.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        if name in CLAUDE_ACTION_TOOLS and not self._claude_in_context():
            log.info("blocked %s: user did not mention Claude ('%s')", name, self._turn_text[:80])
            return CLAUDE_GUARD_MSG
        assessment = safety.classify_call(name, args, tool.risk)
        if assessment.is_risky:
            summary = safety.describe_call(name, args)
            if not await self.confirmer.ask(summary):
                return "The user declined (or did not answer in time). The action was NOT performed."
        log.info("tool %s %s", name, json.dumps(args, default=str)[:200])
        notice = SLOW_TOOL_NOTICE.get(name)
        if notice and out is not None and (name != "claude_chat_new" or args.get("message")):
            out.push(notice)  # spoken while the tool runs; the later reply reuses this stream
            self._state("thinking")
        result = await self.registry.call(name, args)
        if notice and out is not None:
            self._state("speaking")
        if len(result) > MAX_TOOL_RESULT_CHARS:
            result = result[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"
        return result
