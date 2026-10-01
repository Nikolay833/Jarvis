"""Ollama /api/chat client with tool calling (non-streaming)."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

log = logging.getLogger("jarvis.llm")

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove qwen3 <think> blocks (also an unclosed one, or a stray closing tag)."""
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


class ThinkStripper:
    """Incremental `strip_think`: feed raw deltas, get visible text. Tags may split across chunks."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        while True:
            low = self._buf.lower()
            if self._inside:
                i = low.find(self.CLOSE)
                if i < 0:
                    # drop thinking text, keep a possible partial closing tag
                    self._buf = self._buf[len(self._buf) - self._partial(low, self.CLOSE):]
                    break
                self._buf = self._buf[i + len(self.CLOSE):]
                self._inside = False
                continue
            o, c = low.find(self.OPEN), low.find(self.CLOSE)
            hits = [i for i in (o, c) if i >= 0]
            if not hits:
                keep = max(self._partial(low, self.OPEN), self._partial(low, self.CLOSE))
                cut = len(self._buf) - keep
                out.append(self._buf[:cut])
                self._buf = self._buf[cut:]
                break
            i = min(hits)
            out.append(self._buf[:i])
            if i == o:
                self._buf = self._buf[i + len(self.OPEN):]
                self._inside = True
            else:  # stray closing tag: ignore it
                self._buf = self._buf[i + len(self.CLOSE):]
        return "".join(out)

    @staticmethod
    def _partial(low: str, tag: str) -> int:
        """Length of the longest suffix of `low` that is a proper prefix of `tag`."""
        for n in range(min(len(tag) - 1, len(low)), 0, -1):
            if tag.startswith(low[-n:]):
                return n
        return 0

    def flush(self) -> str:
        """End of stream: release held-back text unless it is inside an unclosed think block."""
        rest = "" if self._inside else self._buf
        self._buf = ""
        self._inside = False
        return rest


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


# ---- tool calls emitted as text (qwen3 sometimes does this instead of structured tool_calls) ----
_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|tool_call)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_dec = json.JSONDecoder()


def _decode_all(payload: str) -> list[Any]:
    """All JSON values in `payload`, back to back (a list counts as its items). [] if it is not JSON."""
    out: list[Any] = []
    i, n = 0, len(payload)
    while i < n:
        while i < n and payload[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        try:
            val, i = _dec.raw_decode(payload, i)
        except ValueError:
            return []
        out.extend(val if isinstance(val, list) else [val])
    return out


def _as_call(obj: Any, known: set[str], need_args: bool) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name = obj.get("name")
    if not isinstance(name, str) or name not in known:
        return None
    has_args = "arguments" in obj or "parameters" in obj
    if need_args and not has_args:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return ToolCall(name=name, arguments=args)


def extract_text_tool_calls(content: str, known_names: Any) -> tuple[str, list[ToolCall]]:
    """Find tool calls written as text in `content`; return (content without them, calls).

    Understands `<tool_call>{...}</tool_call>`, fenced JSON and bare JSON objects with "name" plus
    "arguments"/"parameters". Only names in `known_names` count. A <tool_call> tag is always removed.
    """
    known = set(known_names or ())
    if not content or not known:
        return content, []
    calls: list[ToolCall] = []

    def from_payload(payload: str, need_args: bool) -> list[ToolCall] | None:
        vals = _decode_all(payload)
        got = [_as_call(v, known, need_args) for v in vals]
        return got if got and all(got) else None  # type: ignore[return-value]

    def tag_sub(m: re.Match[str]) -> str:
        got = from_payload(m.group(1), False)
        if got:
            calls.extend(got)
        return ""

    def fence_sub(m: re.Match[str]) -> str:
        got = from_payload(m.group(1), True)
        if got is None:
            return m.group(0)
        calls.extend(got)
        return ""

    text = _TAG_RE.sub(tag_sub, content)
    text = _FENCE_RE.sub(fence_sub, text)
    # bare JSON objects anywhere in the remaining text
    out: list[str] = []
    i = 0
    while i < len(text):
        j = text.find("{", i)
        if j < 0:
            break
        try:
            val, end = _dec.raw_decode(text, j)
        except ValueError:
            out.append(text[i:j + 1])
            i = j + 1
            continue
        call = _as_call(val, known, True)
        if call is None:
            out.append(text[i:end])
        else:
            calls.append(call)
            out.append(text[i:j])
        i = end
    out.append(text[i:])
    if not calls and text == content:
        return content, []
    return "".join(out).strip(), calls


class ToolCallStreamFilter:
    """Hold back text that starts a text-form tool call so it is never spoken.

    Feed visible deltas; get the safe-to-speak part. After `<tool_call>` or a line starting with
    `{"name"` (also after a ```json fence) everything is suppressed. Works across chunk splits.
    """

    TAG = "<tool_call>"
    LINE_STARTS = ('{"name"', "```json{\"name\"", "```{\"name\"")

    def __init__(self) -> None:
        self.triggered = False
        self._pending = ""
        self._at_start = True

    def _tag_hold(self, text: str) -> int:
        low = text.lower()
        for n in range(min(len(self.TAG) - 1, len(low)), 0, -1):
            if self.TAG.startswith(low[-n:]):
                return n
        return 0

    def feed(self, chunk: str) -> str:
        if self.triggered:
            return ""
        self._pending += chunk
        out: list[str] = []
        while self._pending and not self.triggered:
            i = self._pending.lower().find(self.TAG)
            if i >= 0:
                out.append(self._pending[:i])
                self._pending = ""
                self.triggered = True
                break
            if self._at_start:
                norm = re.sub(r"\s+", "", self._pending)
                if not norm:
                    out.append(self._pending)
                    self._pending = ""
                    break
                if any(norm.startswith(c) for c in self.LINE_STARTS):
                    self._pending = ""
                    self.triggered = True
                    break
                if any(c.startswith(norm) for c in self.LINE_STARTS):
                    break  # undecided: wait for more text
                self._at_start = False
                continue
            nl = self._pending.find("\n")
            if nl >= 0:
                out.append(self._pending[:nl + 1])
                self._pending = self._pending[nl + 1:]
                self._at_start = True
                continue
            keep = self._tag_hold(self._pending)
            cut = len(self._pending) - keep
            out.append(self._pending[:cut])
            self._pending = self._pending[cut:]
            break
        return "".join(out)

    def flush(self) -> str:
        rest, self._pending = ("" if self.triggered else self._pending), ""
        return rest


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_tool_calls: list[dict[str, Any]] = field(default_factory=list)


class LLM(Protocol):
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse: ...

    # Optional: chat_stream(messages, tools) -> async iterator of str (visible text deltas),
    # ending with one LLMResponse (the complete step). The agent falls back to `chat` without it.


def parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except ValueError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(name=str(fn.get("name", "")), arguments=args))
    return calls


def normalize_keep_alive(value: int | str) -> int | str:
    """Ollama takes a number of seconds (-1 = forever) or a duration string such as "30m".

    A bare numeric string ("-1", "300") must become an int: Ollama rejects "-1" without a unit.
    """
    if isinstance(value, str):
        v = value.strip()
        if re.fullmatch(r"-?\d+", v):
            return int(v)
        return v
    return value


def finalize_response(content: str, raw_calls: list[dict[str, Any]],
                      tools: list[dict[str, Any]] | None) -> LLMResponse:
    """Build the step result; with no structured tool calls, look for text-form calls to known tools."""
    calls = parse_tool_calls({"tool_calls": raw_calls})
    if not calls and tools:
        names = {t.get("function", {}).get("name") for t in tools}
        content, calls = extract_text_tool_calls(content, names)
        if calls:
            log.info("recovered %d tool call(s) written as text: %s", len(calls), [c.name for c in calls])
    return LLMResponse(content=content, tool_calls=calls, raw_tool_calls=list(raw_calls))


class OllamaClient:
    def __init__(self, url: str, model: str, think: bool | None = False,
                 timeout: float = 120.0, num_ctx: int = 8192, keep_alive: int | str = -1,
                 max_reply_tokens: int = 0) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.think = think
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.keep_alive = normalize_keep_alive(keep_alive)
        self.max_reply_tokens = max_reply_tokens
        self._supports_think = True
        self._client: Any = None

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(base_url=self.url, timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def build_payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"num_ctx": self.num_ctx, "temperature": 0.4},
        }
        if self.max_reply_tokens > 0:
            payload["options"]["num_predict"] = self.max_reply_tokens
        if tools:
            payload["tools"] = tools
        if self.think is not None and self._supports_think:
            payload["think"] = self.think
        return payload

    async def warm_up(self) -> float:
        """Load the model into VRAM with an empty chat request. Returns seconds taken."""
        import httpx

        payload = {"model": self.model, "messages": [], "stream": False, "keep_alive": self.keep_alive,
                   "options": {"num_ctx": self.num_ctx}}  # same num_ctx, or Ollama reloads the model
        t0 = time.perf_counter()
        try:
            resp = await self._http().post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach Ollama at {self.url}: {exc!r}") from exc
        if resp.status_code != 200:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        return time.perf_counter() - t0

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        import httpx

        payload = self.build_payload(messages, tools)
        try:
            resp = await self._http().post("/api/chat", json=payload)
            if resp.status_code == 400 and "think" in payload and "think" in resp.text.lower():
                # Model or Ollama version does not know `think`: retry without it.
                self._supports_think = False
                payload.pop("think")
                resp = await self._http().post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach Ollama at {self.url}: {exc}") from exc
        if resp.status_code != 200:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        message = resp.json().get("message", {})
        return finalize_response(strip_think(message.get("content") or ""), message.get("tool_calls") or [], tools)

    async def chat_stream(self, messages: list[dict[str, Any]],
                          tools: list[dict[str, Any]] | None = None) -> AsyncIterator[str | LLMResponse]:
        """Stream one step. Yields visible text deltas (think blocks removed), then one LLMResponse."""
        import httpx

        payload = self.build_payload(messages, tools)
        payload["stream"] = True
        stripper = ThinkStripper()
        guard = ToolCallStreamFilter()
        visible: list[str] = []
        raw_calls: list[dict[str, Any]] = []
        try:
            for attempt in (1, 2):
                async with self._http().stream("POST", "/api/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        if (attempt == 1 and resp.status_code == 400 and "think" in payload
                                and "think" in body.lower()):
                            self._supports_think = False
                            payload.pop("think")
                            continue
                        raise LLMError(f"Ollama error {resp.status_code}: {body[:300]}")
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except ValueError as exc:
                            raise LLMError(f"bad stream line from Ollama: {line[:100]!r}") from exc
                        if chunk.get("error"):
                            raise LLMError(f"Ollama error: {str(chunk['error'])[:300]}")
                        message = chunk.get("message") or {}
                        raw_calls.extend(message.get("tool_calls") or [])
                        delta = message.get("content") or ""
                        if delta:
                            text = stripper.feed(delta)
                            if text:
                                visible.append(text)
                                text = guard.feed(text)
                                if text:
                                    yield text
                        if chunk.get("done"):
                            break
                    break
        except httpx.HTTPError as exc:
            raise LLMError(f"Cannot reach Ollama at {self.url}: {exc}") from exc
        tail = stripper.flush()
        if tail:
            visible.append(tail)
            tail = guard.feed(tail)
            if tail:
                yield tail
        tail = guard.flush()
        if tail:
            yield tail
        yield finalize_response("".join(visible).strip(), raw_calls, tools)
